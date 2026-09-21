"""
Experiment E6 Optimized: RA-GFT Finalized Roadmap — Performance-Fixed Version
===============================================================================
Identical algorithm to exp_e6_rdmap.py. Four parallelization bugs fixed:

  FIX 1 — Sliced IPC payloads (Bug #1 — CRITICAL)
     compute_all_raw_coeffs_hierarchical_opt() passes only the child-block
     slices that each parent worker actually needs, instead of the full
     tangent_c / struct_map_c dicts (3480 entries) to every task.
     Was:  ~1.4 GB IPC per call × 32 calls = ~44 GB per child scale
     Now:  ~8 KB IPC per call × 32 calls = ~256 KB per child scale

  FIX 2 — Persistent loky worker pool (Bug #2 — HIGH)
     The (Q, tau_gate) double loop is wrapped in a single
     `parallel_backend("loky", n_jobs=-1)` context that keeps 12 workers
     alive for the full sweep instead of spawning/tearing down 960 times.

  FIX 3 — @profile guard in library files (Bug #3 — HIGH)
     decider.py and transforms.py now use `try: profile except NameError:`
     so line_profiler instrumentation is a no-op outside kernprof.
     Applied directly to src/pcadc/decider.py and src/pcadc/transforms.py.

  FIX 4 — StructuralGraph cached across modes per worker (Bug #4 — MEDIUM)
     _opt_worker_hierarchical_coeffs() builds the structural graph (cdist)
     ONCE per child block and reuses it for all K modes.
     Was:  cdist called K times per child block per worker
     Now:  cdist called once per child block per worker

  FIX 5 — SCIPY_OPENBLAS_NUM_THREADS set (Bug #5 — MEDIUM)
     scipy uses a private OpenBLAS namespace that ignores OPENBLAS_NUM_THREADS.
     Added SCIPY_OPENBLAS_NUM_THREADS=1 to prevent 768-thread oversubscription.

Algorithm is 100% identical to exp_e6_rdmap.py:
  (1) Q-Dependent Grid Search
  (2) RLGR-Correlated Rate Proxy (RLGRDecider)
  (3) Active-Block Slope Training
  (4) Quantization-Scaled Viterbi Penalty gamma(Q) = gamma_0 * (Q/24)^2

Usage:
    python3 scripts/exp_e6_optimized.py [same flags as exp_e6_rdmap.py]
                                        [--db-path results/exp_e6_optimized.db]
"""

# ── FIX 5: set ALL thread-limiting env vars BEFORE any import ────────────────
import os
os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"   # ← NEW: scipy private BLAS
os.environ["OPENBLAS_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]            = "1"
os.environ["OMP_NUM_THREADS"]            = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"]        = "1"
# ─────────────────────────────────────────────────────────────────────────────

import logging
logging.disable(logging.CRITICAL)

import sys, argparse, sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed, parallel_backend   # ← FIX 2: import parallel_backend

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block

from oracle_em_tangent import extract_block_tangent_planes

# Import algorithm functions we do NOT change
from exp_e4_master_benchmark_suite import (
    fit_gradient_3d, spherical_kmeans, build_parent_child_map,
    grid_search, evaluate_hierarchical_decider_on_coeffs,
    run_viterbi, compute_entropy,
    fast_lstsq_2d, project_slope,
)


# ---------------------------------------------------------------------------
# FIX 3 — @profile no-op is already applied to decider.py / transforms.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Improvement 2 (unchanged): RLGR-Correlated Rate Proxy
# ---------------------------------------------------------------------------

class RLGRDecider(Decider):
    """
    Identical to exp_e6_rdmap.py.
    rate_proxy = L0_nonzeros + rlgr_alpha * n_run_transitions
    """
    def __init__(self, mode: str, lagrange_proportional: float,
                 rlgr_alpha: float = 0.5):
        super().__init__(mode, lagrange_proportional)
        self.rlgr_alpha = rlgr_alpha

    def _zeroNorm(self, Coeffs_quant: np.ndarray) -> float:
        if self.mode in ["1", "2"]:
            base = int(np.count_nonzero(Coeffs_quant, axis=1).sum())
            flat = Coeffs_quant.ravel(order="C")
        else:
            base = int(np.count_nonzero(Coeffs_quant, axis=0).sum())
            flat = Coeffs_quant[:, 0].ravel() if Coeffs_quant.ndim > 1 else Coeffs_quant.ravel()
        nz_mask     = (flat != 0).astype(np.int8)
        n_transitions = int(np.count_nonzero(np.diff(nz_mask)))
        return base + self.rlgr_alpha * n_transitions


# ---------------------------------------------------------------------------
# FIX 1 + FIX 4 — Optimized worker: sliced tangent/struct + sg_cache
# ---------------------------------------------------------------------------

def _opt_worker_hierarchical_coeffs(pi, children_data, tangent_c_slice,
                                    struct_map_c_slice, centroids,
                                    slp, slw, v1, v2, unc, K):
    """
    Optimized version of _top_level_worker_hierarchical_coeffs.

    Changes vs original:
      - FIX 1: tangent_c_slice and struct_map_c_slice contain ONLY the
               child blocks that belong to this parent (not all 3480).
      - FIX 4: StructuralGraph (cdist) is built ONCE per child block and
               reused across all K modes via sg_cache.

    Algorithm logic is identical.
    """
    logging.disable(logging.CRITICAL)
    gft = GFTStrategyWraper()
    app = Approximator()
    res = {k: [] for k in range(K)}

    # Mode 0: structural baseline — just look up precomputed coefficients
    for ci, Vc_arr, Ac_arr, meta in children_data:
        res[0].append(struct_map_c_slice.get(ci))   # FIX 1: use sliced dict

    # FIX 4: build StructuralGraph ONCE per child (cdist is O(N²) — expensive)
    sg_cache = {}
    for ci, Vc_arr, Ac_arr, meta in children_data:
        sg = StructuralGraph(meta)
        sg.set_data(Vc_arr)              # <── cdist called exactly once per child
        sg_cache[ci] = sg

    # Modes k=1..K-1: directional GFT
    for k in range(1, K):
        slope_3d = project_slope(centroids[k - 1], v1, v2, unc)

        for ci, Vc_arr, Ac_arr, meta in children_data:
            tc  = tangent_c_slice[ci]   # FIX 1: from sliced dict
            cv1, cv2 = tc["v1"], tc["v2"]

            s2  = np.dot(slope_3d, cv1) * cv1 + np.dot(slope_3d, cv2) * cv2
            rn  = np.linalg.norm(s2)
            if rn < 1e-3:
                s2 = cv1 * 1e-3; rn = 1e-3

            Vc_c = Vc_arr - Vc_arr.mean(0)
            X    = np.column_stack((Vc_c @ cv1, Vc_c @ cv2))
            Y    = Ac_arr[:, 0] - Ac_arr[:, 0].mean()
            s2d_c = fast_lstsq_2d(X, Y)
            g_c   = s2d_c[0] * cv1 + s2d_c[1] * cv2
            unc_c = np.linalg.norm(g_c)
            s3d_c = (s2 / rn) * (unc_c if unc_c > 1e-8 else rn)

            try:
                sg = sg_cache[ci]                  # FIX 4: reuse — no extra cdist
                ag = AttributeGraph(sg, s3d_c, k, slp[k], slw[k])
                Vr = app._spatial_norm(Vc_arr)
                Aa = Ac_arr.copy()
                Aa[:, 0] = Vr @ s3d_c.T
                ag.set_data(Vc_arr, Aa)
                b_tmp = Block(meta)
                b_tmp.Vblock = Vc_arr
                b_tmp.Ablock = Ac_arr
                _, cs = gft(b_tmp, ag)
                res[k].append(cs)
            except Exception:
                res[k].append(None)

    return pi, res


def compute_all_raw_coeffs_hierarchical_opt(parent_blocks, child_blocks,
                                            parent_to_children,
                                            tangent_p, tangent_c, struct_map_c,
                                            centroids, slp, slw,
                                            g3d_parent, g3d_raw_parent,
                                            pc_V, pc_A, K):
    """
    FIX 1: Pre-slice tangent_c and struct_map_c per parent before building tasks.
    Only the relevant child indices are sent to each worker — not the full 3480-entry dicts.
    Everything else is identical to compute_all_raw_coeffs_hierarchical.
    """
    Np = len(parent_blocks)
    parent_coeffs = [{} for _ in range(Np)]

    tasks = []
    for pi, pb in enumerate(parent_blocks):
        child_idxs = parent_to_children[pi]

        children_data = []
        for ci in child_idxs:
            child_blocks[ci].init_data(pc_V, pc_A)
            children_data.append((ci,
                                   child_blocks[ci].Vblock.copy(),
                                   child_blocks[ci].Ablock.copy(),
                                   child_blocks[ci].metadata))
            child_blocks[ci].clear_data()

        # FIX 1: slice to only this parent's children — payload shrinks ~175×
        tangent_c_slice  = {ci: tangent_c[ci]  for ci in child_idxs}
        struct_map_c_slice = {ci: struct_map_c.get(ci) for ci in child_idxs}

        tasks.append((pi, children_data,
                       tangent_c_slice,        # ← FIX 1: was full tangent_c
                       struct_map_c_slice,     # ← FIX 1: was full struct_map_c
                       centroids, slp, slw,
                       tangent_p[pi]["v1"], tangent_p[pi]["v2"],
                       g3d_raw_parent[pi], K))

    # n_jobs=-1 inherits the active parallel_backend context (FIX 2)
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_opt_worker_hierarchical_coeffs)(*t) for t in tasks
    )
    for pi, res in results:
        parent_coeffs[pi] = res
    return parent_coeffs


# ---------------------------------------------------------------------------
# Helpers (identical to exp_e6_rdmap.py)
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def gamma_q(gamma_0: float, q: int) -> float:
    return gamma_0 * (q / 24.0) ** 2


# ---------------------------------------------------------------------------
# Database — same 6-table schema as exp_e6_rdmap.py
# ---------------------------------------------------------------------------

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS gating_results (
        run_id TEXT PRIMARY KEY, child_b INTEGER, q_step INTEGER,
        tau_gate REAL, tau_train_pct REAL, K INTEGER,
        gamma_0 REAL, gamma_q REAL, rlgr_alpha REAL,
        n_parent_blocks INTEGER, n_child_blocks INTEGER,
        n_active_parents INTEGER, n_gated_parents INTEGER,
        gating_ratio_pct REAL, effective_oh_bpv REAL,
        entropy_active_bits REAL, active_edge_compaction_pct REAL,
        baseline_bpv REAL, baseline_psnr_db REAL,
        hier_transform_bpv REAL, net_bpv REAL,
        delta_bpv REAL, delta_pct REAL,
        psnr_db REAL, psnr_delta_db REAL, mse REAL,
        n_viterbi_switches INTEGER, dominant_mode_k INTEGER,
        dominant_mode_pct REAL, n_training_blocks_used INTEGER,
        timestamp TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS cluster_centroids (
        run_prefix TEXT, tau_train_pct REAL, K INTEGER,
        mode_k INTEGER, centroid_x REAL, centroid_y REAL, centroid_z REAL,
        azimuth_deg REAL, elevation_deg REAL,
        n_training_blocks INTEGER, gradient_energy_threshold REAL,
        PRIMARY KEY (run_prefix, mode_k))""")
    c.execute("""CREATE TABLE IF NOT EXISTS cluster_slopes (
        run_id TEXT, child_b INTEGER, q_step INTEGER, tau_gate REAL,
        mode_k INTEGER, slp REAL, slw REAL, n_training_child_blocks INTEGER,
        PRIMARY KEY (run_id, mode_k))""")
    c.execute("""CREATE TABLE IF NOT EXISTS gradient_distribution (
        run_prefix TEXT, block_scale INTEGER, role TEXT,
        n_blocks INTEGER, grad_mean REAL, grad_std REAL, grad_min REAL,
        grad_p25 REAL, grad_p50 REAL, grad_p75 REAL, grad_p90 REAL,
        grad_max REAL, n_edge_blocks INTEGER, edge_pct REAL,
        PRIMARY KEY (run_prefix, block_scale, role))""")
    c.execute("""CREATE TABLE IF NOT EXISTS parent_block_telemetry (
        run_id TEXT, parent_idx INTEGER, variance REAL,
        gradient_energy REAL, is_active INTEGER, chosen_mode_k INTEGER,
        struct_rate REAL, chosen_rate REAL, rate_delta REAL,
        rate_savings_pct REAL, distortion REAL, n_children INTEGER,
        PRIMARY KEY (run_id, parent_idx))""")
    c.execute("""CREATE TABLE IF NOT EXISTS mode_activation_stats (
        run_id TEXT, mode_k INTEGER, n_activations INTEGER,
        activation_pct REAL, mean_rate_savings_pct REAL, mean_distortion REAL,
        PRIMARY KEY (run_id, mode_k))""")
    conn.commit(); conn.close()


def _conn(db_path): return sqlite3.connect(db_path)


def save_centroids(db_path, run_prefix, tau_train_pct, K, centroids, n_training, threshold):
    conn = _conn(db_path); c = conn.cursor()
    for i, cent in enumerate(centroids):
        x, y, z = float(cent[0]), float(cent[1]), float(cent[2])
        az = float(np.degrees(np.arctan2(y, x)))
        el = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
        c.execute("INSERT OR REPLACE INTO cluster_centroids VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (run_prefix, tau_train_pct, K, i+1, x, y, z, az, el, n_training, threshold))
    conn.commit(); conn.close()


def save_slopes(db_path, run_id, child_b, q, tau_gate, K, slp, slw, n_train):
    conn = _conn(db_path); c = conn.cursor()
    for k in range(K):
        c.execute("INSERT OR REPLACE INTO cluster_slopes VALUES (?,?,?,?,?,?,?,?)",
                  (run_id, child_b, q, tau_gate, k, float(slp[k]), float(slw[k]), n_train))
    conn.commit(); conn.close()


def save_gradient_dist(db_path, run_prefix, scale, role, g3d_raw, n_edge, edge_pct):
    conn = _conn(db_path); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO gradient_distribution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (run_prefix, scale, role, len(g3d_raw),
               float(np.mean(g3d_raw)), float(np.std(g3d_raw)),
               float(np.min(g3d_raw)),
               float(np.percentile(g3d_raw, 25)), float(np.percentile(g3d_raw, 50)),
               float(np.percentile(g3d_raw, 75)), float(np.percentile(g3d_raw, 90)),
               float(np.max(g3d_raw)), n_edge, edge_pct))
    conn.commit(); conn.close()


def save_gating_result(db_path, rec):
    conn = _conn(db_path); c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO gating_results VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        rec["run_id"], rec["child_b"], rec["q_step"],
        rec["tau_gate"], rec["tau_train_pct"], rec["K"],
        rec["gamma_0"], rec["gamma_q"], rec["rlgr_alpha"],
        rec["n_parent_blocks"], rec["n_child_blocks"],
        rec["n_active_parents"], rec["n_gated_parents"],
        rec["gating_ratio_pct"], rec["effective_oh_bpv"],
        rec["entropy_active_bits"], rec["active_edge_compaction_pct"],
        rec["baseline_bpv"], rec["baseline_psnr_db"],
        rec["hier_transform_bpv"], rec["net_bpv"],
        rec["delta_bpv"], rec["delta_pct"],
        rec["psnr_db"], rec["psnr_delta_db"], rec["mse"],
        rec["n_viterbi_switches"], rec["dominant_mode_k"], rec["dominant_mode_pct"],
        rec["n_training_blocks_used"], rec["timestamp"]
    ))
    conn.commit(); conn.close()


def save_parent_telemetry(db_path, run_id, rows):
    conn = _conn(db_path)
    conn.executemany("INSERT OR REPLACE INTO parent_block_telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()


def save_mode_stats(db_path, run_id, vl, rate_mat, dist_mat, K, Np):
    conn = _conn(db_path); c = conn.cursor()
    vl_arr = np.array(vl)
    for k in range(K):
        idxs  = np.where(vl_arr == k)[0]
        n_act = len(idxs)
        act_pct = 100.0 * n_act / Np if Np > 0 else 0.0
        if n_act > 0 and k > 0:
            sav   = [(rate_mat[i, 0] - rate_mat[i, k]) / (rate_mat[i, 0] + 1e-12) * 100.0
                     for i in idxs if not np.isinf(rate_mat[i, k])]
            dists = [dist_mat[i, k] for i in idxs if not np.isinf(dist_mat[i, k])]
            mean_sav  = float(np.mean(sav))   if sav   else 0.0
            mean_dist = float(np.mean(dists)) if dists else 0.0
        else:
            mean_sav  = 0.0
            mean_dist = float(np.mean([dist_mat[i, 0] for i in idxs])) if n_act > 0 else 0.0
        c.execute("INSERT OR REPLACE INTO mode_activation_stats VALUES (?,?,?,?,?,?)",
                  (run_id, k, n_act, act_pct, mean_sav, mean_dist))
    conn.commit(); conn.close()


# ---------------------------------------------------------------------------
# Algorithm helpers (identical to exp_e6_rdmap.py)
# ---------------------------------------------------------------------------

def compute_parent_variances(blocks_p, pc_V, pc_A):
    variances = np.zeros(len(blocks_p))
    for pi, pb in enumerate(blocks_p):
        pb.init_data(pc_V, pc_A)
        lum = pb.Ablock[:, 0]
        variances[pi] = float(np.var(lum)) if len(lum) > 1 else 0.0
        pb.clear_data()
    return variances


def edge_focused_centroids(g3d_norm, g3d_raw, K_hybrid, tau_train_pct):
    if tau_train_pct > 0.0:
        threshold = float(np.percentile(g3d_raw, tau_train_pct))
        edge_mask = g3d_raw >= threshold
    else:
        threshold = 0.0
        edge_mask = np.ones(len(g3d_norm), dtype=bool)
    n_edge = int(edge_mask.sum())
    log(f"  Edge filter: {n_edge}/{len(g3d_norm)} blocks ({100*n_edge/len(g3d_norm):.1f}%) "
        f"at p{tau_train_pct:.0f} threshold={threshold:.4f}")
    if n_edge < K_hybrid * 3:
        log(f"  Warning: too few edge blocks — using all {len(g3d_norm)}.")
        edge_mask = np.ones(len(g3d_norm), dtype=bool)
        threshold = 0.0
    centroids, _ = spherical_kmeans(g3d_norm[edge_mask], K_hybrid)
    return centroids, edge_mask, threshold


def apply_implicit_gate(cost_mat, parent_variances, tau_gate, Np, K):
    active_mask = parent_variances >= tau_gate
    gated = cost_mat.copy()
    for pi in range(Np):
        if not active_mask[pi]:
            gated[pi, 1:] = np.inf
    return gated, active_mask


def build_child_active_training_mask(parent_variances, parent_to_children,
                                     g3d_raw_c, tau_gate, tau_train_pct, Nc):
    child_parent_active = np.zeros(Nc, dtype=bool)
    for pi, var in enumerate(parent_variances):
        if var >= tau_gate:
            for ci in parent_to_children.get(pi, []):
                child_parent_active[ci] = True
    child_edge = g3d_raw_c >= np.percentile(g3d_raw_c, tau_train_pct)
    combined   = child_parent_active & child_edge
    if int(combined.sum()) < 5:
        combined = child_edge
    return combined


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(db_path, child_block_sizes, tau_gates, tau_train_pct,
                   q_steps, K=10, gamma_0=1000.0, rlgr_alpha=0.5):

    init_db(db_path)

    params  = load_experiment_config(project_root / "config/base_config.yaml")
    pc      = PointCloud.from_file(
        project_root / params.sequential_params.point_cloud_path,
        "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm        = params.sequential_params.decider_mode
    lag       = params.sequential_params.lagrange_proportional
    total_v   = pc.V.shape[0]
    K_hybrid  = K - 1
    run_prefix = f"e6opt_tt{tau_train_pct:.0f}_K{K}_rlgr{rlgr_alpha}"

    log(f"PC: {total_v} voxels | mode={dm} | lag={lag}")
    log(f"K={K} | gamma_0={gamma_0} | rlgr_alpha={rlgr_alpha}")
    log("Fixes active: payload slicing | persistent pool | @profile no-op | sg_cache | SCIPY_OPENBLAS=1")

    # ------------------------------------------------------------------
    # Parent blocks (B=32)
    # ------------------------------------------------------------------
    section("PARENT BLOCKS B=32")
    Bp = 32
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=Bp)
    Np = len(blocks_p)
    log(f"Parent blocks: {Np}")

    tangents_p = extract_block_tangent_planes(blocks_p, pc.V, pc.A)
    g3d_norm_p, g3d_raw_p, normals_p = [], [], []
    for i, b in enumerate(blocks_p):
        b.init_data(pc.V, pc.A)
        normals_p.append(tangents_p[i]["v3"])
        g, rn = fit_gradient_3d(b.Vblock, b.Ablock,
                                 tangents_p[i]["v1"], tangents_p[i]["v2"])
        g3d_norm_p.append(g); g3d_raw_p.append(rn)
        b.clear_data()
    g3d_norm_p = np.array(g3d_norm_p)
    g3d_raw_p  = np.array(g3d_raw_p)
    normals_p  = np.array(normals_p)

    nw_p = np.ones(Np)
    for i in range(1, Np):
        nw_p[i] = abs(np.dot(normals_p[i], normals_p[i - 1]))

    et_p = float(np.percentile(g3d_raw_p, tau_train_pct)) if tau_train_pct > 0 else 0.0
    save_gradient_dist(db_path, run_prefix, Bp, "parent", g3d_raw_p,
                       int((g3d_raw_p >= et_p).sum()),
                       100.0 * (g3d_raw_p >= et_p).sum() / Np)

    section("PARENT BLOCK VARIANCES")
    parent_variances = compute_parent_variances(blocks_p, pc.V, pc.A)
    log(f"Var: min={parent_variances.min():.1f} | p25={np.percentile(parent_variances,25):.1f} | "
        f"median={np.median(parent_variances):.1f} | max={parent_variances.max():.1f}")

    section(f"PHASE 1: EDGE-FOCUSED CENTROIDS (tau_train={tau_train_pct}%)")
    centroids_p, edge_mask_p, grad_threshold = edge_focused_centroids(
        g3d_norm_p, g3d_raw_p, K_hybrid, tau_train_pct)
    n_training = int(edge_mask_p.sum())
    log(f"Centroids trained on {n_training} parent blocks")
    save_centroids(db_path, run_prefix, tau_train_pct, K,
                   centroids_p, n_training, grad_threshold)

    # ------------------------------------------------------------------
    # Child block sweep
    # ------------------------------------------------------------------
    for Bc in child_block_sizes:
        section(f"CHILD BLOCK SIZE B={Bc}")
        _, blocks_c = MortonBlockPartition().partition(pc, bsize=Bc)
        Nc = len(blocks_c)
        log(f"Child blocks: {Nc} (~{Nc/Np:.1f} per parent)")

        tangents_c = extract_block_tangent_planes(blocks_c, pc.V, pc.A)
        g3d_norm_c, g3d_raw_c = [], []
        for i, b in enumerate(blocks_c):
            b.init_data(pc.V, pc.A)
            g, rn = fit_gradient_3d(b.Vblock, b.Ablock,
                                     tangents_c[i]["v1"], tangents_c[i]["v2"])
            g3d_norm_c.append(g); g3d_raw_c.append(rn)
            b.clear_data()
        g3d_norm_c = np.array(g3d_norm_c)
        g3d_raw_c  = np.array(g3d_raw_c)

        et_c = float(np.percentile(g3d_raw_c, tau_train_pct)) if tau_train_pct > 0 else 0.0
        save_gradient_dist(db_path, run_prefix, Bc, "child", g3d_raw_c,
                           int((g3d_raw_c >= et_c).sum()),
                           100.0 * (g3d_raw_c >= et_c).sum() / Nc)

        gft_comp = GFTStrategyWraper()
        struct_map_c = {}
        for ci, cb in enumerate(blocks_c):
            cb.init_data(pc.V, pc.A)
            sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
            _, coeffs = gft_comp(cb, sg)
            struct_map_c[ci] = coeffs
            cb.clear_data()

        parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc.V, Bp)

        log("Computing baseline rates (RLGR proxy)...")
        base_res = {}
        for q in q_steps:
            dec_b = RLGRDecider(dm, lag, rlgr_alpha); dec_b._set_vars(q)
            tot_rate, tot_dist = 0.0, 0.0
            for ci in range(Nc):
                _, r, d = dec_b._RDcost(struct_map_c[ci])
                tot_rate += r; tot_dist += d
            bpv  = tot_rate / total_v
            mse  = tot_dist / total_v
            psnr = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
            base_res[q] = (bpv, psnr, mse)
            log(f"  Baseline B={Bc} Q={q}: {bpv:.5f} BPV | {psnr:.2f} dB")

        # ------------------------------------------------------------------
        # FIX 2: Single persistent worker pool for the entire (Q, tau_gate) sweep
        # ------------------------------------------------------------------
        print(f"\n{'Q':>4} | {'tau_g':>7} | {'gamma_q':>8} | {'TrainN':>6} | "
              f"{'Gate%':>6} | {'OH_bpv':>9} | {'Base_bpv':>9} | "
              f"{'Net_bpv':>9} | {'Delta_bpv':>10} | {'PSNR':>7} | Status")
        print("-" * 108)

        # ── FIX 2: wrap sweep in parallel_backend so loky pool is kept alive ──
        with parallel_backend("loky", n_jobs=-1):
            for q in q_steps:
                gq  = gamma_q(gamma_0, q)
                dec = RLGRDecider(dm, lag, rlgr_alpha)
                dec._set_vars(q)
                b_bpv, b_psnr, b_mse = base_res[q]

                for tau_gate in tau_gates:
                    run_id = (f"e6opt_B{Bc}_q{q}_tg{tau_gate:.1f}_"
                              f"tt{tau_train_pct:.0f}_K{K}_rlgr{rlgr_alpha}")

                    # Improvement 3: active-block slope training mask
                    child_train_mask = build_child_active_training_mask(
                        parent_variances, parent_to_children,
                        g3d_raw_c, tau_gate, tau_train_pct, Nc)
                    n_train_blocks = int(child_train_mask.sum())

                    # Improvement 1: Q-dependent grid search
                    slp_opt, slw_opt = grid_search(
                        blocks_c, tangents_c, centroids_p,
                        g3d_raw_c, g3d_norm_c,
                        pc.V, pc.A, dm, lag, q, K, child_train_mask)

                    save_slopes(db_path, run_id, Bc, q, tau_gate, K,
                                slp_opt, slw_opt, n_train_blocks)

                    # FIX 1 + FIX 4: use optimized compute function
                    p_coeffs = compute_all_raw_coeffs_hierarchical_opt(
                        blocks_p, blocks_c, parent_to_children,
                        tangents_p, tangents_c, struct_map_c,
                        centroids_p, slp_opt, slw_opt,
                        g3d_norm_p, g3d_raw_p, pc.V, pc.A, K)

                    cost_mat, rate_mat, dist_mat = evaluate_hierarchical_decider_on_coeffs(
                        p_coeffs, K, dec)

                    gated_cost, active_mask = apply_implicit_gate(
                        cost_mat, parent_variances, tau_gate, Np, K)

                    # Improvement 4: Q-scaled Viterbi penalty
                    vl = run_viterbi(gated_cost, K, gq, nw_p)

                    # ---- Rate accounting (identical to exp_e6_rdmap.py) ----
                    n_active = int(active_mask.sum())
                    n_gated  = Np - n_active
                    gate_pct = 100.0 * n_gated / Np
                    tr_rate  = sum(rate_mat[i, vl[i]] for i in range(Np))
                    tr_bpv   = tr_rate / total_v

                    if n_active > 0:
                        act_labels = np.array([vl[i] for i in range(Np) if active_mask[i]])
                        H_act      = compute_entropy(act_labels, K)
                        oh_bpv     = H_act * n_active / total_v
                    else:
                        H_act, oh_bpv = 0.0, 0.0

                    net_bpv   = tr_bpv + oh_bpv
                    delta_bpv = net_bpv - b_bpv
                    delta_pct = delta_bpv / (b_bpv + 1e-12) * 100.0

                    tot_dist = sum(dist_mat[i, vl[i]] for i in range(Np))
                    mse      = tot_dist / total_v
                    psnr     = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
                    psnr_d   = psnr - b_psnr

                    if n_active > 0:
                        sr_act = sum(rate_mat[i, 0]     for i in range(Np) if active_mask[i])
                        cr_act = sum(rate_mat[i, vl[i]] for i in range(Np) if active_mask[i])
                        ec_pct = (sr_act - cr_act) / (sr_act + 1e-12) * 100.0
                    else:
                        ec_pct = 0.0

                    vl_arr   = np.array(vl)
                    n_sw     = int((vl_arr[1:] != vl_arr[:-1]).sum())
                    mode_cnt = np.bincount(vl_arr, minlength=K)
                    dom_k    = int(np.argmax(mode_cnt))
                    dom_pct  = float(mode_cnt[dom_k] / Np * 100.0)
                    status   = "GAIN" if delta_bpv < 0 else "loss"

                    save_gating_result(db_path, {
                        "run_id": run_id, "child_b": Bc, "q_step": q,
                        "tau_gate": tau_gate, "tau_train_pct": tau_train_pct,
                        "K": K, "gamma_0": gamma_0, "gamma_q": gq,
                        "rlgr_alpha": rlgr_alpha,
                        "n_parent_blocks": Np, "n_child_blocks": Nc,
                        "n_active_parents": n_active, "n_gated_parents": n_gated,
                        "gating_ratio_pct": gate_pct,
                        "effective_oh_bpv": oh_bpv,
                        "entropy_active_bits": float(H_act),
                        "active_edge_compaction_pct": ec_pct,
                        "baseline_bpv": b_bpv, "baseline_psnr_db": b_psnr,
                        "hier_transform_bpv": tr_bpv,
                        "net_bpv": net_bpv, "delta_bpv": delta_bpv,
                        "delta_pct": delta_pct,
                        "psnr_db": psnr, "psnr_delta_db": psnr_d,
                        "mse": mse,
                        "n_viterbi_switches": n_sw,
                        "dominant_mode_k": dom_k, "dominant_mode_pct": dom_pct,
                        "n_training_blocks_used": n_train_blocks,
                        "timestamp": datetime.now().isoformat()
                    })

                    telem = []
                    for pi in range(Np):
                        n_ch = len(parent_to_children.get(pi, []))
                        sr   = float(rate_mat[pi, 0])       if not np.isinf(rate_mat[pi, 0])       else None
                        cr   = float(rate_mat[pi, vl[pi]])  if not np.isinf(rate_mat[pi, vl[pi]])  else None
                        rd   = (sr - cr) if (sr is not None and cr is not None) else None
                        rs   = (rd / (sr + 1e-12) * 100.0) if rd is not None else None
                        dist = float(dist_mat[pi, vl[pi]]) if not np.isinf(dist_mat[pi, vl[pi]]) else None
                        telem.append((run_id, pi,
                                      float(parent_variances[pi]),
                                      float(g3d_raw_p[pi]),
                                      int(active_mask[pi]),
                                      int(vl[pi]),
                                      sr, cr, rd, rs, dist, n_ch))
                    save_parent_telemetry(db_path, run_id, telem)
                    save_mode_stats(db_path, run_id, vl, rate_mat, dist_mat, K, Np)

                    print(f"{q:>4} | {tau_gate:>7.1f} | {gq:>8.1f} | {n_train_blocks:>6} | "
                          f"{gate_pct:>5.1f}% | {oh_bpv:>9.6f} | {b_bpv:>9.5f} | "
                          f"{net_bpv:>9.5f} | {delta_bpv:>+10.5f} | {psnr:>7.2f} | {status}")
        # ── end parallel_backend context (pool released here) ────────────────

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("BEST CONFIG PER (SCALE, Q STEP)")
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    print(f"\n{'B':>4} | {'Q':>4} | {'tau_g':>7} | {'gamma_q':>8} | "
          f"{'TrainN':>6} | {'Gate%':>6} | {'Net BPV':>9} | "
          f"{'Delta':>10} | {'PSNR':>7} | Status")
    print("-" * 90)
    for Bc in child_block_sizes:
        for q in q_steps:
            cur.execute("""
            SELECT tau_gate, gamma_q, n_training_blocks_used,
                   gating_ratio_pct, net_bpv, delta_bpv, psnr_db
            FROM gating_results WHERE child_b=? AND q_step=?
            ORDER BY net_bpv ASC LIMIT 1""", (Bc, q))
            row = cur.fetchone()
            if row:
                tg, gq_, ntr, gr, nb, db_, ps = row
                s = "GAIN" if db_ < 0 else "loss"
                print(f"{Bc:>4} | {q:>4} | {tg:>7.1f} | {gq_:>8.1f} | "
                      f"{ntr:>6} | {gr:>5.1f}% | {nb:>9.5f} | "
                      f"{db_:>+10.5f} | {ps:>7.2f} | {s}")
    conn.close()
    log(f"All results saved → {db_path}")


# ---------------------------------------------------------------------------
# CLI (identical flags to exp_e6_rdmap.py)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp E6 Optimized: RA-GFT Roadmap (parallelization-fixed, same algorithm)")
    p.add_argument("--child-block-sizes", type=str, default="32,16,8,4")
    p.add_argument("--tau-gates",         type=str,
                   default="0.0,50.0,150.0,300.0,500.0,800.0,1200.0,1600.0")
    p.add_argument("--tau-train-pct",     type=float, default=75.0)
    p.add_argument("--q-steps",           type=str, default="24,32,48,64")
    p.add_argument("--ks",                type=str, default="4,8,10,12,16")
    p.add_argument("--gamma-0",           type=float, default=1000.0)
    p.add_argument("--rlgr-alpha",        type=float, default=0.5)
    p.add_argument("--db-path",           type=str,
                   default="results/exp_e6_master_suite.db")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    k_list = [int(k) for k in args.ks.split(",")]
    child_b_list = [int(x) for x in args.child_block_sizes.split(",")]
    tau_gates_list = [float(x) for x in args.tau_gates.split(",")]
    q_steps_list = [int(x) for x in args.q_steps.split(",")]

    for k_val in k_list:
        log(f"==========================================================")
        log(f"STARTING SWEEP FOR K = {k_val}")
        log(f"==========================================================")
        run_experiment(
            db_path           = args.db_path,
            child_block_sizes = child_b_list,
            tau_gates         = tau_gates_list,
            tau_train_pct     = args.tau_train_pct,
            q_steps           = q_steps_list,
            K                 = k_val,
            gamma_0           = args.gamma_0,
            rlgr_alpha        = args.rlgr_alpha,
        )

