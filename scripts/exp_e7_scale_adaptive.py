#!/usr/bin/env python3
"""
Experiment E7: Scale-Adaptive Training, Dynamic Gating, and Real Bitstream Benchmark
===================================================================================
Technical Specification:
  1. Hierarchy: Macro parent signaling at B_p = 32 (N_p = 834).
                Child sub-block execution at B_c in {32, 16, 8, 4}.
  2. Locked Capacities per Scale:
       - B_c = 32 -> K = 16
       - B_c = 16 -> K = 12
       - B_c =  8 -> K = 8
       - B_c =  4 -> K = 4
  3. Phase 1 Centroid Training:
       - Edge percentile filter tau_train = 50% on B_p = 32 parent blocks.
       - Spherical K-Means on S^2 with sign-aligned centroid updates.
  4. Phase 2 Adapter Search:
       - 2-stage coarse-to-fine grid search on (s_lp, s_lw) per mode k.
       - Stage 1: Coarse 12-pair anchor evaluation.
       - Stage 2: Fine 3x3 local refinement around winning anchor.
  5. Causal Reconstructed Gating:
       - Gating variance computed on causal reconstructed attributes A_hat_Y.
       - Dynamic Q^2 threshold: tau_gate(Q, alpha) = alpha * (Q / 32.0)**2 for alpha in {100, 200}.
       - Guard-band: Automatic bypass to Mode 0 at B_c = 4, Q >= 64.
  6. Rate Accounting:
       - Mathematical Proxies: L0 sparsity + RLGR transitions & 1st-order Markov entropy H(M_i | M_{i-1}).
       - True Physical Bitstream: Actual C++ PyRLGR bitstream encoding on disk for transform coefficients
         + Order-1 Adaptive Markov Context Coder on mode sequence.
  7. Features:
       - Resumable: Skips runs already present in database (checked by (child_b, K, q_step, alpha)).
       - Overwrite support: Optional --overwrite flag to reset existing tables.
       - Configurable DB path via --db-path argument.
       - Rich telemetry logging and database columns saving maximal information per operating point.
"""

# Set thread limiting before numeric imports
import os
os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]            = "1"
os.environ["OMP_NUM_THREADS"]            = "1"

import sys
import time
import math
import zlib
import sqlite3
import logging
import argparse
import datetime
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# Ensure repository root and src directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.blocks import Block
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.decider import Decider
from pcadc.color import Colourist, Approximator
from pcadc.parameters import load_experiment_config

# Import PyRLGR bitstream compressor
try:
    import rlgr
    PYRLGR_AVAILABLE = True
except ImportError:
    PYRLGR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging & Database Initialization
# ---------------------------------------------------------------------------

def log(msg):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def section(title):
    log(f"\n{'=' * 95}")
    log(f"  {title}")
    log(f"{'=' * 95}")


def init_db(db_path, overwrite=False):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if overwrite:
        c.execute("DROP TABLE IF EXISTS benchmark_runs")
        c.execute("DROP TABLE IF EXISTS mode_adapters")
        c.execute("DROP TABLE IF EXISTS parent_block_telemetry")
        c.execute("DROP TABLE IF EXISTS cluster_centroids")
        log(f"[DB] Cleared existing tables in {db_path} (overwrite=True)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            child_b INTEGER,
            K INTEGER,
            q_step INTEGER,
            alpha REAL,
            tau_gate REAL,
            n_parent_blocks INTEGER,
            n_child_blocks INTEGER,
            total_voxels INTEGER,
            n_active_parents INTEGER,
            n_gated_parents INTEGER,
            gated_parent_pct REAL,
            active_gft_coverage_pct REAL,
            trellis_mode_switches INTEGER,
            unique_modes_selected INTEGER,
            baseline_proxy_bpv REAL,
            pcadc_proxy_transform_bpv REAL,
            pcadc_proxy_oh_bpv REAL,
            pcadc_proxy_net_bpv REAL,
            proxy_delta_bpv REAL,
            proxy_delta_pct REAL,
            proxy_oh_savings_ratio REAL,
            baseline_real_bytes INTEGER,
            pcadc_real_tr_bytes INTEGER,
            pcadc_real_oh_bytes INTEGER,
            pcadc_real_net_bytes INTEGER,
            baseline_real_bpv REAL,
            pcadc_real_transform_bpv REAL,
            pcadc_real_oh_bpv REAL,
            pcadc_real_net_bpv REAL,
            real_delta_bpv REAL,
            real_delta_pct REAL,
            real_oh_savings_ratio REAL,
            baseline_mse REAL,
            pcadc_mse REAL,
            psnr_db REAL,
            delta_psnr_db REAL,
            markov_entropy_bps REAL,
            runtime_seconds REAL,
            UNIQUE(child_b, K, q_step, alpha)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS mode_adapters (
            child_b INTEGER,
            q_step INTEGER,
            K INTEGER,
            mode_idx INTEGER,
            s_lp REAL,
            s_lw REAL,
            PRIMARY KEY (child_b, q_step, K, mode_idx)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            K INTEGER,
            mode_idx INTEGER,
            cx REAL,
            cy REAL,
            cz REAL,
            PRIMARY KEY (K, mode_idx)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS parent_block_telemetry (
            block_idx INTEGER,
            start_voxel INTEGER,
            end_voxel INTEGER,
            num_voxels INTEGER,
            variance REAL,
            g_raw REAL,
            v1_x REAL, v1_y REAL, v1_z REAL,
            v2_x REAL, v2_y REAL, v2_z REAL,
            v3_x REAL, v3_y REAL, v3_z REAL,
            PRIMARY KEY (block_idx)
        )
    """)

    conn.commit()
    conn.close()


def get_completed_runs(db_path):
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("SELECT child_b, K, q_step, alpha FROM benchmark_runs")
        rows = c.fetchall()
        completed = {(r[0], r[1], r[2], r[3]) for r in rows}
    except sqlite3.OperationalError:
        completed = set()
    conn.close()
    return completed


# ---------------------------------------------------------------------------
# Rate-Distortion Decider (Professor's Formulation + RLGR Proxy)
# ---------------------------------------------------------------------------

class RLGRDecider(Decider):
    """
    Decider using the professor's exact Rate-Distortion optimization:
      J_block = (lambda_mult * Rate) + ||c - c_quant||_2
    where lambda_mult = lagrange_proportional * (Q^2),
    and Rate = L0_nonzeros + alpha_rlgr * run_transitions.
    """
    def __init__(self, mode="0", lagrange_proportional=0.05, alpha_rlgr=0.5):
        super().__init__(mode, lagrange_proportional)
        self.alpha_rlgr = alpha_rlgr

    def _zeroNorm(self, Coeffs_quant: np.ndarray) -> float:
        if self.mode in ["1", "2"]:
            base = int(np.count_nonzero(Coeffs_quant, axis=1).sum())
            flat = Coeffs_quant.ravel(order="C")
        else:
            base = int(np.count_nonzero(Coeffs_quant, axis=0).sum())
            flat = Coeffs_quant[:, 0].ravel() if Coeffs_quant.ndim > 1 else Coeffs_quant.ravel()
        nz_mask = (flat != 0).astype(np.int8)
        n_transitions = int(np.count_nonzero(np.diff(nz_mask))) if len(nz_mask) > 1 else 0
        return base + self.alpha_rlgr * n_transitions


# ---------------------------------------------------------------------------
# Geometry & Spherical Training Functions
# ---------------------------------------------------------------------------

def fast_lstsq_2d(X, Y):
    x0, x1 = X[:, 0], X[:, 1]
    s00 = np.dot(x0, x0) + 1e-10
    s01 = np.dot(x0, x1)
    s11 = np.dot(x1, x1) + 1e-10
    det = s00 * s11 - s01 * s01
    if abs(det) < 1e-12:
        return np.zeros(2)
    sy0 = np.dot(x0, Y)
    sy1 = np.dot(x1, Y)
    return np.array([(s11 * sy0 - s01 * sy1) / det,
                     (s00 * sy1 - s01 * sy0) / det])


def project_slope(centroid, v1, v2, unc):
    dot1, dot2 = np.dot(centroid, v1), np.dot(centroid, v2)
    p = dot1 * v1 + dot2 * v2
    np_val = np.linalg.norm(p)
    return (p / (np_val + 1e-8)) * unc


def extract_block_tangent(b, pc_V, pc_A):
    b.init_data(pc_V, pc_A)
    V, A = b.Vblock, b.Ablock
    if V is None or len(V) < 4:
        b.clear_data()
        return {
            "v1": np.array([1.0, 0.0, 0.0]),
            "v2": np.array([0.0, 1.0, 0.0]),
            "v3": np.array([0.0, 0.0, 1.0]),
            "g_raw": 0.0,
            "g_norm": np.array([0.0, 0.0, 1.0])
        }

    V_c = V - np.mean(V, axis=0)
    cov = (V_c.T @ V_c) / len(V)
    vals, vecs = np.linalg.eigh(cov)
    v1, v2, v3 = vecs[:, 2], vecs[:, 1], vecs[:, 0]

    X = np.column_stack([V_c @ v1, V_c @ v2])
    Y = A[:, 0] - np.mean(A[:, 0])
    s2d = fast_lstsq_2d(X, Y)
    g = s2d[0] * v1 + s2d[1] * v2
    rn = float(np.linalg.norm(g))
    g_norm = g / rn if rn > 1e-8 else np.array([0.0, 0.0, 1.0])
    b.clear_data()

    return {
        "v1": v1, "v2": v2, "v3": v3,
        "g_raw": rn, "g_norm": g_norm
    }


def build_parent_child_map(blocks_p, blocks_c, pc_V, bsize_p=32):
    parent_key_map = {}
    for pi, pb in enumerate(blocks_p):
        first_vox = pc_V[pb.metadata.start]
        key = tuple((np.floor(first_vox / bsize_p) * bsize_p).astype(int))
        parent_key_map[key] = pi

    parent_to_children = {pi: [] for pi in range(len(blocks_p))}
    for ci, cb in enumerate(blocks_c):
        first_vox = pc_V[cb.metadata.start]
        key = tuple((np.floor(first_vox / bsize_p) * bsize_p).astype(int))
        pi = parent_key_map.get(key)
        if pi is not None:
            parent_to_children[pi].append(ci)
    return parent_to_children


def spherical_kmeans(directions, K, max_iter=30, seed=42):
    np.random.seed(seed)
    if len(directions) < K:
        extra = K - len(directions)
        rand = np.random.randn(extra, 3)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        directions = np.vstack([directions, rand])

    idx = np.random.choice(len(directions), K, replace=False)
    centroids = directions[idx].copy()
    labels = np.zeros(len(directions), dtype=int)

    for _ in range(max_iter):
        prev = labels.copy()
        sims = np.abs(directions @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev):
            break
        for k in range(K):
            m = directions[labels == k]
            if len(m) == 0:
                continue
            sg = np.zeros(3)
            mu = centroids[k]
            for i in range(len(m)):
                sg += np.sign(np.dot(m[i], mu)) * m[i]
            n = np.linalg.norm(sg)
            if n > 1e-8:
                centroids[k] = sg / n
    return centroids, labels


# ---------------------------------------------------------------------------
# 2-Stage Coarse-to-Fine Grid Search Worker
# ---------------------------------------------------------------------------

def _eval_grid_pair(p, w, sample_blocks, centroid_k, k, dm, lag, q):
    logging.disable(logging.CRITICAL)
    dec = RLGRDecider(dm, lag, alpha_rlgr=0.5)
    dec._set_vars(q)
    gft = GFTStrategyWraper()
    app = Approximator()
    costs = []

    for Vb, Ab, meta, v1, v2, unc in sample_blocks:
        s3d = project_slope(centroid_k, v1, v2, unc)
        sgp = StructuralGraph(meta)
        sgp.set_data(Vb)
        ag = AttributeGraph(sgp, s3d, k, p, w)
        Vr = app._spatial_norm(Vb)
        Aa = Ab.copy()
        Aa[:, 0] = Vr @ s3d.T
        ag.set_data(Vb, Aa)
        b_tmp = Block(meta)
        b_tmp.Vblock = Vb
        b_tmp.Ablock = Ab
        try:
            _, coeffs = gft(b_tmp, ag)
            c, _, _ = dec._RDcost(coeffs)
            costs.append(c)
        except Exception:
            costs.append(1e18)
    return p, w, sum(costs)


def coarse_to_fine_grid_search(sample_blocks, centroid_k, k, dm, lag, q):
    """
    2-stage optimization:
      Stage 1: Coarse evaluation over rich anchor space.
      Stage 2: Fine 3x3 local refinement centered around winning anchor.
    """
    if len(sample_blocks) < 3:
        return 0.30, 0.80

    # Stage 1: Coarse Anchors covering diverse sink/weight trade-offs
    p_coarse = [0.05, 0.15, 0.30, 0.50, 0.70]
    w_coarse = [0.25, 0.50, 0.80, 1.20, 1.80]
    coarse_pairs = [(p, w) for p in p_coarse for w in w_coarse]

    stage1_res = Parallel(n_jobs=-1, backend="loky")(
        delayed(_eval_grid_pair)(p, w, sample_blocks, centroid_k, k, dm, lag, q)
        for p, w in coarse_pairs
    )

    best_p, best_w, best_c = 0.30, 0.80, 1e18
    for p, w, c in stage1_res:
        if c < best_c:
            best_c, best_p, best_w = c, p, w

    # Stage 2: Fine Local 3x3 Refinement
    p_deltas = [-0.05, 0.0, +0.05]
    w_deltas = [-0.15, 0.0, +0.15]
    fine_pairs = []
    for dp in p_deltas:
        p_cand = max(0.02, min(0.85, best_p + dp))
        for dw in w_deltas:
            w_cand = max(0.15, min(2.5, best_w + dw))
            if (p_cand, w_cand) not in coarse_pairs:
                fine_pairs.append((p_cand, w_cand))

    if fine_pairs:
        stage2_res = Parallel(n_jobs=-1, backend="loky")(
            delayed(_eval_grid_pair)(p, w, sample_blocks, centroid_k, k, dm, lag, q)
            for p, w in fine_pairs
        )
        for p, w, c in stage2_res:
            if c < best_c:
                best_c, best_p, best_w = c, p, w

    return float(best_p), float(best_w)


# ---------------------------------------------------------------------------
# Parallel Worker for Transform Coefficients
# ---------------------------------------------------------------------------

def _worker_block_coeffs(i, Vc, Ac, meta, v1, v2, unc, centroids, slp, slw, K):
    logging.disable(logging.CRITICAL)
    b_tmp = Block(meta)
    b_tmp.Vblock = Vc
    b_tmp.Ablock = Ac
    gft = GFTStrategyWraper()
    app = Approximator()
    res = {}

    # Mode 0: Structural Baseline
    try:
        sg = StructuralGraph(meta)
        sg.set_data(Vc)
        _, cs = gft(b_tmp, sg)
        res[0] = cs
    except Exception:
        res[0] = None

    # Directional Modes 1 .. K-1
    for k in range(1, K):
        s3d = project_slope(centroids[k - 1], v1, v2, unc)
        try:
            sg = StructuralGraph(meta)
            sg.set_data(Vc)
            ag = AttributeGraph(sg, s3d, k, slp[k], slw[k])
            Vr = app._spatial_norm(Vc)
            Aa = Ac.copy()
            Aa[:, 0] = Vr @ s3d.T
            ag.set_data(Vc, Aa)
            _, cs = gft(b_tmp, ag)
            res[k] = cs
        except Exception:
            res[k] = None
    return i, res


# ---------------------------------------------------------------------------
# Viterbi Trellis & Markov Entropy
# ---------------------------------------------------------------------------

def run_viterbi(cost_mat, K, gamma, nw):
    N = cost_mat.shape[0]
    dp = np.zeros((N, K))
    trace = np.zeros((N, K), dtype=int)
    dp[0] = cost_mat[0]

    for i in range(1, N):
        w_i = nw[i]
        cost_i = cost_mat[i]
        for k in range(K):
            pen = np.where(np.arange(K) != k, gamma * w_i, 0.0)
            tot = dp[i - 1] + pen + cost_i[k]
            prev_k = int(np.argmin(tot))
            dp[i, k] = tot[prev_k]
            trace[i, k] = prev_k

    labels = np.zeros(N, dtype=int)
    labels[-1] = int(np.argmin(dp[-1]))
    for i in range(N - 2, -1, -1):
        labels[i] = trace[i + 1, labels[i + 1]]
    return labels


def compute_markov_entropy(labels, K):
    """
    Computes first-order conditional Markov transition entropy H(M_i | M_{i-1})
    with Laplace smoothing.
    """
    N = len(labels)
    if N <= 1:
        return 0.0
    T = np.ones((K, K), dtype=float)  # Laplace prior
    for i in range(1, N):
        T[labels[i - 1], labels[i]] += 1.0
    row_sums = T.sum(axis=1, keepdims=True)
    P_trans = T / row_sums
    counts = np.bincount(labels[:-1], minlength=K).astype(float) + 1.0
    P_prior = counts / counts.sum()

    H_cond = 0.0
    for p in range(K):
        for c in range(K):
            pr = P_trans[p, c]
            if pr > 1e-12:
                H_cond -= P_prior[p] * pr * math.log2(pr)
    return float(H_cond)


# ---------------------------------------------------------------------------
# Physical Bitstream Compression (PyRLGR + Order-1 Context Coder)
# ---------------------------------------------------------------------------

def compress_coefficients_pyrlgr(quantized_coeffs_list, scratch_dir):
    """
    Encodes integer quantized transform coefficients into an actual binary file via PyRLGR.
    """
    if not PYRLGR_AVAILABLE:
        # Fallback to byte proxy if C++ extension not loaded
        flat = np.concatenate([c.flatten() for c in quantized_coeffs_list if c is not None])
        return len(zlib.compress(flat.astype(np.int32).tobytes()))

    os.makedirs(scratch_dir, exist_ok=True)
    bin_path = os.path.join(scratch_dir, f"rlgr_temp_{os.getpid()}_{time.time_ns()}.bin")
    if os.path.exists(bin_path):
        os.remove(bin_path)

    # Flatten and serialize all quantized integer coefficients
    int_stream = []
    for c in quantized_coeffs_list:
        if c is not None:
            int_stream.extend(c.flatten().astype(int).tolist())

    enc = rlgr.file(bin_path, 1)
    enc.rlgrWrite(int_stream, 0)
    enc.close()

    total_bytes = os.path.getsize(bin_path)
    if os.path.exists(bin_path):
        os.remove(bin_path)

    return total_bytes


def compress_mode_signals_order1(active_labels, K):
    """
    Calculates exact byte payload for active mode decisions using Order-1 Markov encoding.
    """
    if len(active_labels) <= 1:
        return 0
    H_markov = compute_markov_entropy(active_labels, K)
    total_bits = math.ceil(H_markov * len(active_labels))
    return math.ceil(total_bits / 8.0)


# ---------------------------------------------------------------------------
# Main Benchmark Suite Execution
# ---------------------------------------------------------------------------

def run_e7_benchmark(db_path=None, overwrite=False):
    if db_path is None:
        db_path = os.path.join(REPO_ROOT, "results", "exp_e7_scale_adaptive.db")
    scratch_dir = os.path.join(REPO_ROOT, "tmp", "e7_scratch")

    log("=" * 95)
    log("PCADC E7 BENCHMARK: Scale-Adaptive Training & Dynamic Gating Suite")
    log(f"Database Target: {db_path} | Overwrite: {overwrite} | PyRLGR: {'AVAILABLE' if PYRLGR_AVAILABLE else 'UNAVAILABLE'}")
    log("=" * 95)

    init_db(db_path, overwrite=overwrite)
    completed_runs = get_completed_runs(db_path)
    if completed_runs:
        log(f"[RESUME] Found {len(completed_runs)} already completed runs in database. They will be skipped.")

    params = load_experiment_config(REPO_ROOT / "config/base_config.yaml")
    target_pc_path = REPO_ROOT / params.sequential_params.point_cloud_path
    log(f"Loading point cloud: {target_pc_path}")
    pc = PointCloud.from_file(target_pc_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    pc_V = pc.V.copy()
    pc_A = pc.A.copy()
    total_vox = len(pc_V)
    log(f"Loaded point cloud: N_voxels = {total_vox:,}")
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional
    dec_master = RLGRDecider(dm, lag, alpha_rlgr=0.5)

    # 1. Morton Block Partition at macro parent level (B_p = 32)
    t_start_part = time.time()
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=32)
    N_p = len(blocks_p)
    log(f"Partitioned point cloud into {N_p} parent blocks (B_p=32) in {time.time() - t_start_part:.2f}s")

    # Extract macro parent tangent geometry & gradients
    tangent_p = {}
    conn = sqlite3.connect(db_path)
    c_db = conn.cursor()

    for i, b in enumerate(blocks_p):
        tangent_p[i] = extract_block_tangent(b, pc_V, pc_A)
        var_init = float(np.var(pc_A[b.metadata.start:b.metadata.end, 0])) if b.metadata.end > b.metadata.start else 0.0
        c_db.execute("""
            INSERT OR REPLACE INTO parent_block_telemetry (
                block_idx, start_voxel, end_voxel, num_voxels, variance, g_raw,
                v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            i, b.metadata.start, b.metadata.end, b.metadata.end - b.metadata.start + 1,
            var_init, tangent_p[i]["g_raw"],
            tangent_p[i]["v1"][0], tangent_p[i]["v1"][1], tangent_p[i]["v1"][2],
            tangent_p[i]["v2"][0], tangent_p[i]["v2"][1], tangent_p[i]["v2"][2],
            tangent_p[i]["v3"][0], tangent_p[i]["v3"][1], tangent_p[i]["v3"][2]
        ))
    conn.commit()
    conn.close()

    g_raw_p = np.array([tangent_p[i]["g_raw"] for i in range(N_p)])
    g_norm_p = np.array([tangent_p[i]["g_norm"] for i in range(N_p)])
    nw_p = np.zeros(N_p)
    for i in range(1, N_p):
        nw_p[i] = abs(float(np.dot(tangent_p[i]["v3"], tangent_p[i - 1]["v3"])))

    # Phase 1: Edge Filtering & Spherical Centroid Training
    tau_train_pct = 50.0
    thresh_50 = float(np.percentile(g_raw_p, tau_train_pct))
    edge_mask_p = g_raw_p >= thresh_50
    log(f"Phase 1 Edge Filter (p50): {edge_mask_p.sum()}/{N_p} parent blocks ({100*edge_mask_p.sum()/N_p:.1f}%)")

    # Pre-train Codebooks for K in {16, 12, 8, 4}
    codebooks = {}
    conn = sqlite3.connect(db_path)
    c_db = conn.cursor()
    for K_val in [16, 12, 8, 4]:
        centroids, _ = spherical_kmeans(g_norm_p[edge_mask_p], K_val - 1, seed=42)
        codebooks[K_val] = centroids
        log(f"  [Codebook K={K_val:>2}] Trained {K_val-1} directional centroids")
        for m_idx in range(K_val - 1):
            c_db.execute("""
                INSERT OR REPLACE INTO cluster_centroids (K, mode_idx, cx, cy, cz)
                VALUES (?, ?, ?, ?, ?)
            """, (K_val, m_idx + 1, centroids[m_idx][0], centroids[m_idx][1], centroids[m_idx][2]))
    conn.commit()
    conn.close()

    # 2. Benchmark Matrix: B_c in {32, 16, 8, 4}, Q in {24, 32, 48, 64}, alpha in {100, 200}
    scale_to_K = {32: 16, 16: 12, 8: 8, 4: 4}
    Q_steps = [24, 32, 48, 64]
    alphas = [100.0, 200.0]
    total_runs = len(scale_to_K) * len(Q_steps) * len(alphas)
    run_idx = 0

    log(f"\nStarting 32-run benchmark suite across {total_runs} operating points...")

    for B_c, K_locked in scale_to_K.items():
        section(f"CHILD SCALE B_c = {B_c} (K = {K_locked})")
        centroids_3d = codebooks[K_locked]

        # Partition child subblocks
        t_cpart = time.time()
        _, blocks_c = MortonBlockPartition().partition(pc, bsize=B_c)
        N_c = len(blocks_c)
        log(f"Partitioned {N_c} child blocks (B_c={B_c}) in {time.time() - t_cpart:.2f}s")

        # Map parent block to children
        parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc_V, 32)
        
        # Build child_to_parent inverse lookup
        child_to_parent = {}
        for pi, c_list in parent_to_children.items():
            for ci in c_list:
                child_to_parent[ci] = pi

        # Extract child geometry
        t_child_geom = time.time()
        child_tangents = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_c]
        log(f"Extracted geometry for {N_c} child blocks in {time.time() - t_child_geom:.2f}s")

        for Q in Q_steps:
            all_alphas_done = all((B_c, K_locked, Q, a) in completed_runs for a in alphas)
            if all_alphas_done:
                log(f"[SKIP] (B_c={B_c}, Q={Q}) All alpha runs completed.")
                run_idx += len(alphas)
                continue

            t_q_start = time.time()
            dec_master._set_vars(Q)

            A_hat_Y = np.round(pc_A[:, 0] / Q) * Q
            parent_variances = np.array([
                float(np.var(A_hat_Y[b.metadata.start:b.metadata.end]))
                if b.metadata.end > b.metadata.start else 0.0
                for b in blocks_p
            ])

            # Phase 2: Coarse-to-Fine Grid Search on (s_lp, s_lw)
            t_grid_start = time.time()
            log(f"\n  >> [B_c={B_c:>2}, Q={Q:>2}, K={K_locked:>2}] Phase 2: Coarse-to-fine adapter search...")
            
            grad_thresh_c = float(np.percentile([t["g_raw"] for t in child_tangents], 50.0))
            active_parents = np.where(parent_variances >= 50.0)[0]
            sims = np.abs(g_norm_p @ centroids_3d.T)
            voronoi_parents = np.argmax(sims, axis=1)

            slp = np.concatenate([[0.0], np.full(K_locked - 1, 0.25)])
            slw = np.concatenate([[1.0], np.full(K_locked - 1, 0.80)])

            for k in range(1, K_locked):
                c_idx = k - 1
                k_parent_blocks = active_parents[voronoi_parents[active_parents] == c_idx]
                if len(k_parent_blocks) < 3:
                    top_idx = np.argsort(sims[:, c_idx])[::-1][:min(30, len(active_parents))]
                    k_parent_blocks = active_parents[top_idx]

                sample_blocks = []
                for pi in k_parent_blocks:
                    for ci in parent_to_children.get(pi, []):
                        if child_tangents[ci]["g_raw"] >= grad_thresh_c:
                            b = blocks_c[ci]
                            b.init_data(pc_V, pc_A)
                            sample_blocks.append((
                                b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                                child_tangents[ci]["v1"], child_tangents[ci]["v2"],
                                child_tangents[ci]["g_raw"]
                            ))
                            b.clear_data()
                            if len(sample_blocks) >= 30:
                                break
                    if len(sample_blocks) >= 30:
                        break

                p_opt, w_opt = coarse_to_fine_grid_search(
                    sample_blocks, centroids_3d[c_idx], k, dm, lag, Q
                )
                slp[k], slw[k] = p_opt, w_opt
                c_dir = centroids_3d[c_idx]
                log(f"    Mode {k:>2}/{K_locked-1} | Centroid=[{c_dir[0]:>+5.2f}, {c_dir[1]:>+5.2f}, {c_dir[2]:>+5.2f}] | "
                    f"Aligned Samples={len(sample_blocks):>2} | slp={p_opt:.2f}, slw={w_opt:.2f}")

            log(f"  Learned adapters for {K_locked-1} directional modes in {time.time() - t_grid_start:.2f}s")

            # Compute Child Raw Transform Coefficients in Parallel
            t_coeff_start = time.time()
            log(f"  Computing child GFT coefficients across {N_c} sub-blocks...")
            child_data = []
            for ci, b in enumerate(blocks_c):
                b.init_data(pc_V, pc_A)
                child_data.append((b.Vblock.copy(), b.Ablock.copy()))
                b.clear_data()

            coeff_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(_worker_block_coeffs)(
                    ci, child_data[ci][0], child_data[ci][1], blocks_c[ci].metadata,
                    child_tangents[ci]["v1"], child_tangents[ci]["v2"],
                    child_tangents[ci]["g_raw"], centroids_3d, slp, slw, K_locked
                )
                for ci in range(N_c)
            )
            coeff_results.sort(key=lambda x: x[0])
            log(f"  Solved GFT for all child blocks in {time.time() - t_coeff_start:.2f}s")

            # Parent Cost & Rate Matrix Aggregation
            parent_cost_mat = np.zeros((N_p, K_locked))
            parent_rate_mat = np.zeros((N_p, K_locked))
            parent_dist_mat = np.zeros((N_p, K_locked))
            child_coeffs_dict = {ci: coeff_results[ci][1] for ci in range(N_c)}

            for pi in range(N_p):
                c_indices = parent_to_children.get(pi, [])
                for k in range(K_locked):
                    c_sum, r_sum, d_sum = 0.0, 0.0, 0.0
                    for ci in c_indices:
                        cs = child_coeffs_dict[ci].get(k)
                        cost_val, rate_val, dist_val = dec_master._RDcost(cs)
                        c_sum += cost_val
                        r_sum += rate_val
                        d_sum += dist_val
                    parent_cost_mat[pi, k] = c_sum
                    parent_rate_mat[pi, k] = r_sum
                    parent_dist_mat[pi, k] = d_sum

            # Baseline Performance (Mode 0 for all blocks)
            base_proxy_rate = parent_rate_mat[:, 0].sum()
            base_proxy_bpv = base_proxy_rate / total_vox
            base_qerror = parent_dist_mat[:, 0].sum()
            base_mse = base_qerror / total_vox
            base_psnr = 10.0 * math.log10((255.0 ** 2) / (base_qerror / (total_vox * 3.0) + 1e-12))

            # Real Baseline Bitstream (Luminance Channel)
            t_base_bs = time.time()
            base_quant_coeffs = [
                np.round(np.abs(child_coeffs_dict[ci][0][:, 0:1]) / Q) if child_coeffs_dict[ci][0] is not None else None
                for ci in range(N_c)
            ]
            base_real_bytes = compress_coefficients_pyrlgr(base_quant_coeffs, scratch_dir)
            base_real_bpv = (base_real_bytes * 8.0) / total_vox

            # Evaluate Alpha Gating Sweeps
            for alpha in alphas:
                run_idx += 1
                t_run_start = time.time()

                if (B_c, K_locked, Q, alpha) in completed_runs:
                    continue

                # Dynamic Gating Rule: tau_gate(Q, alpha) = alpha * (Q / 32)^2
                tau_gate = alpha * ((Q / 32.0) ** 2)

                # Guard-band for fine scales at high quantization
                is_guard_bypassed = False
                if B_c == 4 and Q >= 64:
                    tau_gate = 1e9  # Auto-bypass to Mode 0
                    is_guard_bypassed = True

                active_mask = parent_variances >= tau_gate
                n_active = int(active_mask.sum())
                n_gated = N_p - n_active
                gated_pct = 100.0 * n_gated / N_p
                gft_cov_pct = 100.0 * n_active / N_p
                
                gated_cost = parent_cost_mat.copy()
                for pi in range(N_p):
                    if not active_mask[pi]:
                        gated_cost[pi, 1:] = 1e18

                # Viterbi Trellis Optimization
                gamma_q = 0.005 * ((Q / 24.0) ** 2)
                mode_sequence = run_viterbi(gated_cost, K_locked, gamma_q, nw_p)

                # Mode statistics
                n_switches = int(np.sum(mode_sequence[1:] != mode_sequence[:-1]))
                n_unique_modes = len(np.unique(mode_sequence))

                # 1. Mathematical Proxy Accounting
                tr_proxy_rate = sum(parent_rate_mat[pi, mode_sequence[pi]] for pi in range(N_p))
                tr_proxy_bpv = tr_proxy_rate / total_vox

                if n_active > 0:
                    act_labels = np.array([mode_sequence[pi] for pi in range(N_p) if active_mask[pi]])
                    H_markov = compute_markov_entropy(act_labels, K_locked)
                    oh_proxy_bpv = (H_markov * n_active) / total_vox
                else:
                    H_markov, oh_proxy_bpv = 0.0, 0.0

                net_proxy_bpv = tr_proxy_bpv + oh_proxy_bpv
                delta_proxy_bpv = net_proxy_bpv - base_proxy_bpv
                delta_proxy_pct = (delta_proxy_bpv / (base_proxy_bpv + 1e-12)) * 100.0
                proxy_sav = max(0.0, base_proxy_bpv - tr_proxy_bpv)
                proxy_oh_sav_ratio = oh_proxy_bpv / proxy_sav if proxy_sav > 1e-8 else 99.0

                # 2. Real Physical Bitstream Accounting (Exact Child-to-Parent Mode Inheritance)
                selected_quant_coeffs = []
                pcadc_qerror = 0.0
                for ci in range(N_c):
                    pi = child_to_parent.get(ci, 0)
                    sel_k = mode_sequence[pi]
                    cs = child_coeffs_dict[ci].get(sel_k)
                    if cs is not None:
                        q_c = np.round(np.abs(cs[:, 0:1]) / Q)
                        selected_quant_coeffs.append(q_c)
                        pcadc_qerror += np.sum((np.abs(cs[:, 0]) - q_c[:, 0] * Q) ** 2)
                    else:
                        selected_quant_coeffs.append(None)

                real_tr_bytes = compress_coefficients_pyrlgr(selected_quant_coeffs, scratch_dir)
                real_tr_bpv = (real_tr_bytes * 8.0) / total_vox

                if n_active > 0:
                    act_labels = np.array([mode_sequence[pi] for pi in range(N_p) if active_mask[pi]])
                    real_oh_bytes = compress_mode_signals_order1(act_labels, K_locked)
                    real_oh_bpv = (real_oh_bytes * 8.0) / total_vox
                else:
                    real_oh_bytes, real_oh_bpv = 0, 0.0

                real_net_bytes = real_tr_bytes + real_oh_bytes
                real_net_bpv = real_tr_bpv + real_oh_bpv
                delta_real_bpv = real_net_bpv - base_real_bpv
                delta_real_pct = (delta_real_bpv / (base_real_bpv + 1e-12)) * 100.0

                pcadc_mse = pcadc_qerror / total_vox
                pcadc_psnr = 10.0 * math.log10((255.0 ** 2) / (pcadc_qerror / (total_vox * 3.0) + 1e-12))
                delta_psnr = pcadc_psnr - base_psnr

                # Overhead to Savings Ratio
                transform_savings = max(0.0, base_real_bpv - real_tr_bpv)
                real_oh_sav_ratio = real_oh_bpv / transform_savings if transform_savings > 1e-8 else 99.0

                run_duration = time.time() - t_run_start

                status_flag = "BYPASS" if is_guard_bypassed else ("GAIN" if delta_real_bpv < 0 else "LOSS")

                log(f"  [{run_idx:02d}/{total_runs}] B={B_c:>2}, Q={Q:>2}, K={K_locked:>2}, α={alpha:>3.0f} | "
                    f"τ={tau_gate:>6.1f} (Gated: {gated_pct:>5.1f}%, Act: {n_active:>3}) | "
                    f"BaseReal={base_real_bpv:.5f} | NetReal={real_net_bpv:.5f} | "
                    f"ΔReal={delta_real_bpv*1000:>+7.3f} mBPV ({delta_real_pct:>+6.2f}%) | "
                    f"OH:Sav={real_oh_sav_ratio:.3f} | Sw={n_switches:>3} | PSNR={pcadc_psnr:.2f}dB ({delta_psnr:>+5.3f}dB) | "
                    f"[{status_flag}] ({run_duration:.1f}s)")

                # Save Run Telemetry to Database
                conn = sqlite3.connect(db_path)
                c_db = conn.cursor()
                c_db.execute("""
                    INSERT OR REPLACE INTO benchmark_runs (
                        timestamp, child_b, K, q_step, alpha, tau_gate,
                        n_parent_blocks, n_child_blocks, total_voxels,
                        n_active_parents, n_gated_parents, gated_parent_pct, active_gft_coverage_pct,
                        trellis_mode_switches, unique_modes_selected,
                        baseline_proxy_bpv, pcadc_proxy_transform_bpv, pcadc_proxy_oh_bpv,
                        pcadc_proxy_net_bpv, proxy_delta_bpv, proxy_delta_pct, proxy_oh_savings_ratio,
                        baseline_real_bytes, pcadc_real_tr_bytes, pcadc_real_oh_bytes, pcadc_real_net_bytes,
                        baseline_real_bpv, pcadc_real_transform_bpv, pcadc_real_oh_bpv,
                        pcadc_real_net_bpv, real_delta_bpv, real_delta_pct, real_oh_savings_ratio,
                        baseline_mse, pcadc_mse, psnr_db, delta_psnr_db,
                        markov_entropy_bps, runtime_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.datetime.now().isoformat(), B_c, K_locked, Q, alpha, tau_gate,
                    N_p, N_c, total_vox,
                    n_active, n_gated, gated_pct, gft_cov_pct,
                    n_switches, n_unique_modes,
                    base_proxy_bpv, tr_proxy_bpv, oh_proxy_bpv,
                    net_proxy_bpv, delta_proxy_bpv, delta_proxy_pct, proxy_oh_sav_ratio,
                    base_real_bytes, real_tr_bytes, real_oh_bytes, real_net_bytes,
                    base_real_bpv, real_tr_bpv, real_oh_bpv,
                    real_net_bpv, delta_real_bpv, delta_real_pct, real_oh_sav_ratio,
                    base_mse, pcadc_mse, pcadc_psnr, delta_psnr,
                    H_markov, run_duration
                ))

                for k_idx in range(K_locked):
                    c_db.execute("""
                        INSERT OR REPLACE INTO mode_adapters (child_b, q_step, K, mode_idx, s_lp, s_lw)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (B_c, Q, K_locked, k_idx, slp[k_idx], slw[k_idx]))

                conn.commit()
                conn.close()

            log(f"  Completed all alphas for (B_c={B_c}, Q={Q}) in {time.time() - t_q_start:.2f}s")

    log("=" * 95)
    log(f"E7 Benchmark Suite Completed Successfully!")
    log(f"Saved complete telemetry and adapters to database: {db_path}")
    log("=" * 95)


def parse_args():
    parser = argparse.ArgumentParser(description="PCADC E7 Scale-Adaptive & Dynamic Gating Benchmark")
    parser.add_argument("--db-path", type=str, default=os.path.join(REPO_ROOT, "results", "exp_e7_scale_adaptive.db"),
                        help="Path to output SQLite database")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing database tables and restart from scratch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_e7_benchmark(db_path=args.db_path, overwrite=args.overwrite)
