#!/usr/bin/env python3
"""
=============================================================================
PCADC EXPERIMENT E10: Multi-Sequence Offline Codebook Verification Benchmark
=============================================================================

Protocol:
  - Train codebook (spherical K-means centroids + mode adapters) on frame 0
    of each sequence.
  - Encode frames 1..T using the frozen trained codebook (offline verification).
  - Compute aggregate PSNR_Y and Rate (bpv) using the paper formula:
      PSNR_Y = -10*log10( (1/T) * sum_t( ||Y_t - Y_hat_t||^2 / (255^2 * N_t) ) )
      R      = sum_t(B_t) / sum_t(N_t)   [bpv]
    where B_t = RLGR_bits_t + CABAC_mode_bits_t  (per-frame overhead)

  - Codebook sidestream (centroids) counted separately / amortized.

Storage Strategy:
  - PLY frames streamed in-memory from zip (no disk extraction).
  - SQLite DB: sequences, codebook tables, frame_results, aggregate_results,
    cluster_distribution, kmeans_iterations, adapter_opt_steps.
  - Parquet files: per-block results at B=32 level (all K, Q). Snappy compressed.
  - Bitstreams: saved for ALL Q of the BEST (B_c, K) per sequence after all
    frames have been encoded (identified by best aggregate BD-Rate).

Resumability:
  - Each (seq_id, frame_idx, child_b, K, q_step) row is UNIQUE-constrained.
    Interrupted runs skip already-completed rows automatically.

Sequences: 14 sequences (andrew9/10, david9/10, phil9/10, ricardo9/10,
           sarah9/10, longdress, loot, redandblack, soldier)
=============================================================================
"""

import os
os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]            = "1"
os.environ["OMP_NUM_THREADS"]            = "1"

import sys
import time
import math
import zlib
import bz2
import lzma
import json
import sqlite3
import logging
import zipfile
import datetime
import re
import io
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from pcadc.pointcloud import PointCloud, PointCloudMetadata, MortonBlockPartition
from pcadc.blocks import Block, BlockMetadata
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper, ConnectedGFTProcessor, DisconnectedGFTProcessor
from pcadc.color import Colourist, Approximator
from pcadc.parameters import load_experiment_config

# Re-use all stable primitives from exp_e9
from exp_e9_multi_entropy_suite import (
    RLGRDecider,
    fast_lstsq_2d,
    extract_block_tangent,
    build_parent_child_map,
    spherical_kmeans,
    project_slope,
    _eval_single_pair,
    run_viterbi,
    decode_block_attributes,
    compress_coefficients_pyrlgr,
    Order1ArithmeticCoder,
    evaluate_all_overhead_encoders,
)

try:
    import rlgr as _rlgr_mod
    PYRLGR_AVAILABLE = True
except ImportError:
    _rlgr_mod = None
    PYRLGR_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sequence Registry
# ---------------------------------------------------------------------------

SEQUENCES = [
    # 8iVFB sequences first (depth 10, 1024^3 bounding box), then dynamic sequences
    # (name,         zip_filename,         depth)
    ("longdress",    "longdress.zip",      10),
    ("loot",         "loot.zip",           10),
    ("redandblack",  "redandblack.zip",    10),
    ("soldier",      "soldier.zip",        10),
    ("andrew9",      "andrew9.zip",        9),
    ("andrew10",     "andrew10.zip",       10),
    ("david9",       "david9.zip",         9),
    ("david10",      "david10.zip",        10),
    ("phil9",        "phil9.zip",          9),
    ("phil10",       "phil10.zip",         10),
    ("ricardo9",     "ricardo9.zip",       9),
    ("ricardo10",    "ricardo10.zip",      10),
    ("sarah9",       "sarah9.zip",         9),
    ("sarah10",      "sarah10.zip",        10),
]

# Depth-adaptive parent block size:
#   depth 9  → B_p=16  (16/512 = 1/32 of bounding box — same relative scale as B_p=32 in depth 10)
#   depth 10 → B_p=32  (32/1024 = 1/32 of bounding box)
def get_bp(depth: int) -> int:
    return 16 if depth == 9 else 32

# Depth-adaptive child experiment grid.
# For depth 9, B_c=32 is excluded — those blocks average ~1000+ pts causing O(N^3) eigensolver to
# dominate (>40 min/config/frame). B_c=16 is the maximum at depth 9 and geometrically equivalent.
def get_experiments(depth: int, include_b32: bool = True):
    if depth == 9 or not include_b32:
        return [
            (16, 30), (16, 24), (16, 16), (16, 8), (16, 4),
            (8,  30), (8,  24), (8,  16), (8,  8), (8,  4),
            (4,  30), (4,  24), (4,  16), (4,  8), (4,  4),
        ]
    else:  # depth 10 with B_c=32
        return [
            (32, 30), (32, 24), (32, 16), (32, 8), (32, 4),
            (16, 30), (16, 24), (16, 16), (16, 8), (16, 4),
            (8,  30), (8,  24), (8,  16), (8,  8), (8,  4),
            (4,  30), (4,  24), (4,  16), (4,  8), (4,  4),
        ]

Q_STEPS        = [24, 32, 48, 64]
GAMMA_VAL      = 0.001
ALPHA_GATE     = 100.0
ALPHA_RLGR     = 0.5
MAX_SAMPLE_BLK = 30   # max blocks per cluster for adapter training


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def section(title, w=95):
    log("=" * w)
    log(f"  {title}")
    log("=" * w)

def subsection(title, w=70):
    log(f"  {'─' * w}")
    log(f"    {title}")
    log(f"  {'─' * w}")


# ---------------------------------------------------------------------------
# PLY streaming from ZIP  (no disk extraction)
# ---------------------------------------------------------------------------

def list_ply_files_in_zip(zip_path: Path):
    """Return sorted list of .ply member names inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
    return sorted(n for n in names if n.lower().endswith(".ply"))


def load_ply_from_zip(zip_path: Path, member: str):
    """
    Stream one PLY file from the zip into RAM.
    Returns (V: float64 [N,3], A: float64 [N,3]  RGB in [0,255]).
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        raw = z.read(member)

    # Normalize line endings
    raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = raw.split(b"\n")

    # Parse ASCII header
    n_verts = None
    header_end = 0
    for i, line in enumerate(lines):
        m = re.search(rb"element vertex (\d+)", line)
        if m:
            n_verts = int(m.group(1))
        if line.strip() == b"end_header":
            header_end = i + 1
            break
    assert n_verts is not None, f"No vertex count in {member}"

    data_lines = [l for l in lines[header_end:] if l.strip()]
    assert len(data_lines) >= n_verts, f"Truncated PLY {member}"

    mat = np.array(
        [list(map(float, l.split())) for l in data_lines[:n_verts]],
        dtype=np.float64
    )
    return mat[:, :3], mat[:, 3:6]   # V, A_rgb


def make_pc(V, A_rgb, name, depth, frame_idx):
    """Wrap arrays into a PointCloud object (RGB → YUV applied externally)."""
    meta = PointCloudMetadata(
        dataset="8i" if depth == 10 else "8i_v9",
        sequence=name, depth=depth, frame=frame_idx
    )
    pc = PointCloud(meta)
    pc.V = V.copy()
    pc.A = A_rgb.copy()
    return pc


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

SCHEMA = {
    "sequences": """
        CREATE TABLE IF NOT EXISTS sequences (
            seq_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT UNIQUE,
            depth           INTEGER,
            zip_path        TEXT,
            n_frames        INTEGER,
            n_train_voxels  INTEGER,
            train_ply_path  TEXT,
            created_at      TEXT
        )""",

    "cluster_centroids": """
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            seq_id          INTEGER,
            K               INTEGER,
            mode_idx        INTEGER,
            cx REAL, cy REAL, cz REAL,
            theta_spherical REAL,
            PRIMARY KEY (seq_id, K, mode_idx)
        )""",

    "mode_adapters": """
        CREATE TABLE IF NOT EXISTS mode_adapters (
            seq_id           INTEGER,
            K                INTEGER,
            Q                INTEGER,
            mode_idx         INTEGER,
            s_lp             REAL,
            s_lw             REAL,
            n_sample_blocks  INTEGER,
            mean_train_cost  REAL,
            training_seconds REAL,
            PRIMARY KEY (seq_id, K, Q, mode_idx)
        )""",

    "kmeans_iterations": """
        CREATE TABLE IF NOT EXISTS kmeans_iterations (
            seq_id                 INTEGER,
            K                      INTEGER,
            iteration              INTEGER,
            inertia                REAL,
            n_reassigned           INTEGER,
            max_centroid_shift_deg REAL,
            converged              INTEGER,
            iteration_seconds      REAL,
            PRIMARY KEY (seq_id, K, iteration)
        )""",

    "adapter_opt_steps": """
        CREATE TABLE IF NOT EXISTS adapter_opt_steps (
            step_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            seq_id             INTEGER,
            K                  INTEGER,
            Q                  INTEGER,
            mode_idx           INTEGER,
            step_type          TEXT,
            step_idx           INTEGER,
            s_lp               REAL,
            s_lw               REAL,
            rd_cost            REAL,
            n_blocks_evaluated INTEGER,
            step_seconds       REAL
        )""",

    "frame_results": """
        CREATE TABLE IF NOT EXISTS frame_results (
            result_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            seq_id                 INTEGER,
            frame_idx              INTEGER,
            frame_ply              TEXT,
            child_b                INTEGER,
            K                      INTEGER,
            q_step                 INTEGER,

            n_voxels               INTEGER,
            n_parent_blocks        INTEGER,
            n_child_blocks         INTEGER,
            n_active_parents       INTEGER,
            n_gated_parents        INTEGER,
            gated_pct              REAL,
            tau_gate               REAL,

            n_trellis_switches     INTEGER,
            n_unique_modes         INTEGER,
            mode_entropy_bits      REAL,

            rlgr_bits              INTEGER,
            cabac_overhead_bits    INTEGER,
            deflate_overhead_bits  INTEGER,
            lzma_overhead_bits     INTEGER,
            raw_fixed_bits         INTEGER,
            baseline_rlgr_bits     INTEGER,

            frame_bpv              REAL,
            baseline_bpv           REAL,
            delta_bpv              REAL,
            delta_pct              REAL,

            decoded_se_Y           REAL,
            decoded_se_U           REAL,
            decoded_se_V           REAL,
            n_decoded_pts          INTEGER,
            frame_psnr_Y           REAL,
            baseline_psnr_Y        REAL,

            bitstream_rlgr_path    TEXT,
            bitstream_modes_path   TEXT,
            bitstream_meta_path    TEXT,

            frame_runtime_seconds  REAL,
            timestamp              TEXT,

            UNIQUE(seq_id, frame_idx, child_b, K, q_step)
        )""",

    "cluster_distribution": """
        CREATE TABLE IF NOT EXISTS cluster_distribution (
            seq_id             INTEGER,
            K                  INTEGER,
            frame_idx          INTEGER,
            mode_idx           INTEGER,
            n_blocks_assigned  INTEGER,
            mean_cosine_sim    REAL,
            std_cosine_sim     REAL,
            mean_g_raw         REAL,
            n_gated            INTEGER,
            pct_gated          REAL,
            PRIMARY KEY (seq_id, K, frame_idx, mode_idx)
        )""",

    "aggregate_results": """
        CREATE TABLE IF NOT EXISTS aggregate_results (
            agg_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            seq_id           INTEGER,
            child_b          INTEGER,
            K                INTEGER,
            q_step           INTEGER,

            T_frames         INTEGER,
            sum_total_bits   INTEGER,
            sum_voxels       INTEGER,
            sum_se_Y         REAL,
            sum_pts          INTEGER,
            sum_baseline_bits INTEGER,

            aggregate_bpv    REAL,
            baseline_bpv     REAL,
            aggregate_psnr_Y REAL,
            baseline_psnr_Y  REAL,
            delta_bpv        REAL,
            delta_pct        REAL,
            bd_rate_pct      REAL,

            completed_at     TEXT,
            UNIQUE(seq_id, child_b, K, q_step)
        )""",
}


def init_db(db_path, overwrite=False):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if overwrite:
        for tbl in SCHEMA:
            c.execute(f"DROP TABLE IF EXISTS {tbl}")
        log(f"[DB] Dropped all tables (overwrite=True)")
    for ddl in SCHEMA.values():
        c.execute(ddl)
    conn.commit()
    conn.close()
    log(f"[DB] Ready: {db_path}")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn(db_path):
    return sqlite3.connect(db_path, timeout=30)

def get_or_create_seq_id(db_path, name, depth, zip_path):
    with _conn(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT seq_id FROM sequences WHERE name=?", (name,))
        row = c.fetchone()
        if row:
            return row[0]
        c.execute(
            "INSERT INTO sequences (name, depth, zip_path, created_at) VALUES (?,?,?,?)",
            (name, depth, str(zip_path), datetime.datetime.now().isoformat())
        )
        return c.lastrowid

def has_codebook(db_path, seq_id, K):
    with _conn(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM cluster_centroids WHERE seq_id=? AND K=?", (seq_id, K)
        ).fetchone()[0]
    return n > 0

def has_adapters(db_path, seq_id, K, Q):
    with _conn(db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM mode_adapters WHERE seq_id=? AND K=? AND Q=?", (seq_id, K, Q)
        ).fetchone()[0]
    return n > 0

def load_codebook(db_path, seq_id, K):
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT cx, cy, cz FROM cluster_centroids WHERE seq_id=? AND K=? ORDER BY mode_idx",
            (seq_id, K)
        ).fetchall()
    return np.array([[r[0], r[1], r[2]] for r in rows])

def load_adapters(db_path, seq_id, K, Q):
    slp = np.concatenate([[0.0], np.full(K - 1, 0.25)])
    slw = np.concatenate([[1.0], np.full(K - 1, 0.35)])
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT mode_idx, s_lp, s_lw FROM mode_adapters WHERE seq_id=? AND K=? AND Q=? ORDER BY mode_idx",
            (seq_id, K, Q)
        ).fetchall()
    for mode_idx, p, w in rows:
        slp[mode_idx] = p
        slw[mode_idx] = w
    return slp, slw

def get_completed_frame_keys(db_path, seq_id):
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT frame_idx, child_b, K, q_step FROM frame_results WHERE seq_id=?", (seq_id,)
        ).fetchall()
    return set(tuple(r) for r in rows)


# ---------------------------------------------------------------------------
# Instrumented Spherical K-Means  (saves convergence trace to DB)
# ---------------------------------------------------------------------------

def spherical_kmeans_instrumented(directions, K, db_path, seq_id, max_iter=50, seed=42):
    np.random.seed(seed)
    N = len(directions)
    if N < K:
        pad = np.random.randn(K - N, 3)
        pad /= np.linalg.norm(pad, axis=1, keepdims=True)
        directions = np.vstack([directions, pad])

    idx = np.random.choice(len(directions), K, replace=False)
    centroids = directions[idx].copy()
    labels = np.zeros(len(directions), dtype=int)
    iter_rows = []

    for it in range(max_iter):
        t0 = time.time()
        prev_labels    = labels.copy()
        prev_centroids = centroids.copy()

        sims   = np.abs(directions @ centroids.T)
        labels = np.argmax(sims, axis=1)

        for k in range(K):
            m = directions[labels == k]
            if len(m) == 0:
                continue
            sg = np.zeros(3)
            mu = centroids[k]
            for v in m:
                sg += np.sign(np.dot(v, mu)) * v
            n = np.linalg.norm(sg)
            if n > 1e-8:
                centroids[k] = sg / n

        converged = bool(np.all(labels == prev_labels))
        best_sims = np.max(np.abs(directions @ centroids.T), axis=1)
        inertia   = float(np.mean(1.0 - best_sims))
        n_reass   = int(np.sum(labels != prev_labels))
        dots      = np.clip(np.abs(np.einsum("ki,ki->k", centroids, prev_centroids)), 0.0, 1.0)
        max_shift = float(np.max(np.arccos(dots) * 180.0 / np.pi))
        iter_rows.append((seq_id, K, it, inertia, n_reass, max_shift, int(converged), time.time()-t0))

        if converged:
            break

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO kmeans_iterations
            (seq_id, K, iteration, inertia, n_reassigned, max_centroid_shift_deg, converged, iteration_seconds)
            VALUES (?,?,?,?,?,?,?,?)
        """, iter_rows)

    return centroids, labels


# ---------------------------------------------------------------------------
# Instrumented adapter optimization  (saves evaluation trace to DB)
# ---------------------------------------------------------------------------

def optimize_adapter_instrumented(sample_blocks, centroid_k, k, dm, lag, q, db_path, seq_id, K):
    """Coarse grid + 1D golden-section on w.  Every eval saved to DB."""
    if not sample_blocks:
        return 0.25, 0.35, 0.0

    inv_phi = (math.sqrt(5) - 1) / 2
    step_rows = []
    idx = [0]

    def eval_and_record(p, w, stype):
        t0 = time.time()
        cost = _eval_single_pair(p, w, sample_blocks, centroid_k, k, dm, lag, q)
        step_rows.append((seq_id, K, q, k, stype, idx[0], p, w, cost, len(sample_blocks), time.time()-t0))
        idx[0] += 1
        return cost

    # Coarse grid
    coarse_p = [0.05, 0.25, 0.60]
    coarse_w = [0.15, 0.35, 0.75]
    best_p, best_w, best_cost = 0.25, 0.35, 1e18
    for p in coarse_p:
        for w in coarse_w:
            c = eval_and_record(p, w, "coarse")
            if c < best_cost:
                best_cost, best_p, best_w = c, p, w

    # Golden-section on w
    w_lo = max(0.05, best_w - 0.15)
    w_hi = min(1.00, best_w + 0.15)
    w1 = w_hi - inv_phi * (w_hi - w_lo)
    w2 = w_lo + inv_phi * (w_hi - w_lo)
    c1 = eval_and_record(best_p, w1, "golden_w")
    c2 = eval_and_record(best_p, w2, "golden_w")
    for _ in range(4):
        if c1 < c2:
            w_hi = w2; w2 = w1; c2 = c1
            w1 = w_hi - inv_phi * (w_hi - w_lo)
            c1 = eval_and_record(best_p, w1, "golden_w")
        else:
            w_lo = w1; w1 = w2; c1 = c2
            w2 = w_lo + inv_phi * (w_hi - w_lo)
            c2 = eval_and_record(best_p, w2, "golden_w")

    opt_w = float(np.clip((w_lo + w_hi) / 2.0, 0.05, 1.0))
    opt_p = float(np.clip(best_p, 0.01, 1.0))

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT INTO adapter_opt_steps
            (seq_id, K, Q, mode_idx, step_type, step_idx, s_lp, s_lw, rd_cost, n_blocks_evaluated, step_seconds)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, step_rows)

    return opt_p, opt_w, best_cost


# ---------------------------------------------------------------------------
# Parallel GFT worker  (same logic as exp_e9 _worker_block_coeffs)
# ---------------------------------------------------------------------------

def _worker_gft(ci, Vb, Ab, meta, v1, v2, unc, centroids, slp, slw, K):
    logging.disable(logging.CRITICAL)
    gft = GFTStrategyWraper()
    app = Approximator()
    cmap = {}
    b_tmp = Block(meta)
    b_tmp.Vblock = Vb
    b_tmp.Ablock = Ab

    sg0 = StructuralGraph(meta)
    sg0.set_data(Vb)
    try:
        _, c0 = gft(b_tmp, sg0)
        cmap[0] = c0
    except Exception:
        cmap[0] = None

    for k in range(1, K):
        s3d = project_slope(centroids[k-1], v1, v2, unc)
        sgp = StructuralGraph(meta)
        sgp.set_data(Vb)
        ag = AttributeGraph(sgp, s3d, k, slp[k], slw[k])
        Vr = app._spatial_norm(Vb)
        Aa = Ab.copy(); Aa[:, 0] = Vr @ s3d.T
        ag.set_data(Vb, Aa)
        try:
            _, ck = gft(b_tmp, ag)
            cmap[k] = ck
        except Exception:
            cmap[k] = None

    return ci, cmap


def _worker_decode_block(ci, meta, Vb, Ab, q_c, Q, sk, c_k, p_k, w_k, t_c):
    logging.disable(logging.CRITICAL)
    if q_c is None:
        return ci, 0.0, 0.0, 0.0, 0
    app_dec = Approximator()
    if sk == 0:
        sg = StructuralGraph(meta)
        sg.set_data(Vb)
        rec_Y = decode_block_attributes(meta, Vb, q_c, Q, sg)
    else:
        s3d = project_slope(c_k, t_c["v1"], t_c["v2"], t_c["g_raw"])
        sgp = StructuralGraph(meta)
        sgp.set_data(Vb)
        ag = AttributeGraph(sgp, s3d, sk, p_k, w_k)
        Vr = app_dec._spatial_norm(Vb)
        Aa = Ab.copy()
        Aa[:, 0] = Vr @ s3d.T
        ag.set_data(Vb, Aa)
        rec_Y = decode_block_attributes(meta, Vb, q_c, Q, ag)
    se_y = float(np.sum((Ab[:, 0:1] - rec_Y) ** 2))
    se_u = float(np.sum((Ab[:, 1:2] - rec_Y) ** 2)) if Ab.shape[1] > 1 else 0.0
    se_v = float(np.sum((Ab[:, 2:3] - rec_Y) ** 2)) if Ab.shape[1] > 2 else 0.0
    return ci, se_y, se_u, se_v, len(Vb)


def _worker_decode_baseline(ci, meta, Vb, Ab, qc, Q):
    logging.disable(logging.CRITICAL)
    if qc is None:
        return ci, 0.0, 0
    sg = StructuralGraph(meta)
    sg.set_data(Vb)
    rec = decode_block_attributes(meta, Vb, qc, Q, sg)
    bse = float(np.sum((Ab[:, 0:1] - rec) ** 2))
    return ci, bse, len(Vb)


# ---------------------------------------------------------------------------
# RLGR compression returning raw bytes for optional saving
# ---------------------------------------------------------------------------

def compress_to_bytes(quant_list, scratch_dir):
    """Returns (n_bytes: int, raw: bytes)."""
    if not PYRLGR_AVAILABLE:
        flat = np.concatenate([c.flatten() for c in quant_list if c is not None])
        raw = zlib.compress(flat.astype(np.int32).tobytes())
        return len(raw), raw

    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, f"e10_{os.getpid()}_{time.time_ns()}.bin")
    int_stream = [v for c in quant_list if c is not None for v in c.flatten().astype(int).tolist()]
    enc = _rlgr_mod.file(path, 1)
    enc.rlgrWrite(int_stream, 1)
    enc.close()
    raw = Path(path).read_bytes()
    os.remove(path)
    return len(raw), raw


# ---------------------------------------------------------------------------
# Bitstream writer
# ---------------------------------------------------------------------------

def write_bitstreams(bitstream_dir, seq_name, frame_idx, B_c, K, Q,
                     rlgr_raw, modes_raw, meta_dict):
    d = Path(bitstream_dir) / seq_name / f"frame_{frame_idx:04d}"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"B{B_c}_K{K:02d}_Q{Q:02d}"
    rp = d / f"{tag}_rlgr.bin";  rp.write_bytes(rlgr_raw)
    mp = d / f"{tag}_modes.bin"; mp.write_bytes(modes_raw)
    jp = d / f"{tag}_meta.json"; jp.write_text(json.dumps(meta_dict, indent=2))
    rel = lambda p: str(Path(p).relative_to(REPO_ROOT))
    return rel(rp), rel(mp), rel(jp)


# ---------------------------------------------------------------------------
# BD-Rate computation  (log-rate cubic polynomial, 4-point)
# ---------------------------------------------------------------------------

def bdrate(bpv_anchor, psnr_anchor, bpv_test, psnr_test):
    try:
        psnr_min = max(min(psnr_anchor), min(psnr_test))
        psnr_max = min(max(psnr_anchor), max(psnr_test))
        if psnr_min >= psnr_max:
            return float("nan")
        pa = np.polyfit(psnr_anchor, np.log(np.array(bpv_anchor) + 1e-12), 3)
        pt = np.polyfit(psnr_test,   np.log(np.array(bpv_test)   + 1e-12), 3)
        xs = np.linspace(psnr_min, psnr_max, 100)
        return float((np.exp(np.mean(np.polyval(pt, xs) - np.polyval(pa, xs))) - 1.0) * 100.0)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# TRAINING PHASE
# ---------------------------------------------------------------------------

def train_sequence(seq_id, seq_name, depth, zip_path, db_path, scratch_dir, dm, lag, B_p, experiments):
    """
    Train codebooks and adapters on frame 0 of the sequence.
    B_p: parent block size (16 for depth 9, 32 for depth 10).
    experiments: depth-specific (B_c, K) grid — used to derive K_vals.
    Returns (codebooks, all_adapters, ply_files, blocks_p, nw_p, tangent_p_train).
    """
    subsection(f"TRAIN  {seq_name}  (frame 0)  [B_p={B_p}]")

    ply_files = list_ply_files_in_zip(zip_path)
    train_ply = ply_files[0]
    n_frames  = len(ply_files)

    log(f"    Loading frame 0: {train_ply}")
    t0 = time.time()
    V_raw, A_raw = load_ply_from_zip(zip_path, train_ply)
    log(f"    {len(V_raw):,} voxels loaded in {time.time()-t0:.2f}s")

    pc = make_pc(V_raw, A_raw, seq_name, depth, 0)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    pc_V = pc.V.copy()
    pc_A = pc.A.copy()

    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE sequences SET n_frames=?, n_train_voxels=?, train_ply_path=? WHERE seq_id=?",
            (n_frames, len(pc_V), train_ply, seq_id)
        )

    # Fixed parent partition — depth-adaptive B_p
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=B_p)
    N_p = len(blocks_p)
    log(f"    Parent partition: {N_p} blocks (B_p={B_p})")

    tangent_p = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_p]
    g_raw_p  = np.array([t["g_raw"]  for t in tangent_p])
    g_norm_p = np.array([t["g_norm"] for t in tangent_p])

    # Neighbour normal weights for Viterbi (used at inference time)
    nw_p = np.zeros(N_p)
    for i in range(1, N_p):
        nw_p[i] = abs(float(np.dot(tangent_p[i]["v3"], tangent_p[i-1]["v3"])))

    # Training directions: top-50% gradient energy blocks
    thresh50    = float(np.percentile(g_raw_p, 50))
    train_dirs  = g_norm_p[g_raw_p >= thresh50]
    log(f"    Training directions: {len(train_dirs)} / {N_p} blocks (g_raw >= {thresh50:.3f})")

    codebooks    = {}   # K -> centroids  (K-1 × 3)
    all_adapters = {}   # K -> {Q -> (slp, slw)}

    K_vals = sorted(set(K for _, K in experiments))

    for K_val in K_vals:
        subsection(f"TRAIN  {seq_name}  K={K_val}")

        # --- Codebook ---
        if has_codebook(db_path, seq_id, K_val):
            log(f"      Codebook K={K_val} already in DB — loading.")
            centroids = load_codebook(db_path, seq_id, K_val)
        else:
            log(f"      K-means (K={K_val-1} directions) ...")
            t_km = time.time()
            centroids, _ = spherical_kmeans_instrumented(
                train_dirs, K_val - 1, db_path, seq_id, max_iter=50, seed=42
            )
            log(f"      K-means done in {time.time()-t_km:.2f}s")

            rows = []
            for m_idx, c3d in enumerate(centroids):
                theta = float(np.arccos(np.clip(abs(c3d[1]), 0.0, 1.0)) * 180.0 / np.pi)
                rows.append((seq_id, K_val, m_idx+1, float(c3d[0]), float(c3d[1]), float(c3d[2]), theta))
            with _conn(db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO cluster_centroids
                    (seq_id, K, mode_idx, cx, cy, cz, theta_spherical) VALUES (?,?,?,?,?,?,?)
                """, rows)

        codebooks[K_val] = centroids
        all_adapters[K_val] = {}

        # --- Adapters ---
        sims         = np.abs(g_norm_p @ centroids.T)
        voronoi_p    = np.argmax(sims, axis=1)
        active_p     = g_raw_p >= thresh50

        # Check which Q steps are already done (for resumability)
        needed_Q = [Q for Q in Q_STEPS if not has_adapters(db_path, seq_id, K_val, Q)]
        for Q in Q_STEPS:
            if Q not in needed_Q:
                log(f"      Adapters K={K_val} Q={Q:02d} already in DB — loading.")
                slp, slw = load_adapters(db_path, seq_id, K_val, Q)
                all_adapters[K_val][Q] = (slp, slw)

        if needed_Q:
            log(f"      Adapters K={K_val} optimizing for Q in {needed_Q} (batched)...")
            t_ad = time.time()

            # Pre-build sample blocks per mode k (shared across all Q)
            mode_sample_blocks = {}
            for k in range(1, K_val):
                c_idx = k - 1
                k_idxs = np.where((voronoi_p == c_idx) & active_p)[0]
                if len(k_idxs) < 3:
                    k_idxs = np.argsort(sims[:, c_idx])[::-1][:min(MAX_SAMPLE_BLK, N_p)]

                sample_blocks = []
                for pi in k_idxs[:MAX_SAMPLE_BLK]:
                    b = blocks_p[pi]
                    b.init_data(pc_V, pc_A)
                    if len(b.Vblock) >= 4:
                        sample_blocks.append((
                            b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                            tangent_p[pi]["v1"], tangent_p[pi]["v2"],
                            tangent_p[pi]["g_raw"]
                        ))
                    b.clear_data()
                mode_sample_blocks[k] = sample_blocks

            # Flatten tasks across all (Q, k) combinations
            tasks = [
                (k, mode_sample_blocks[k], centroids[k - 1], dm, lag, Q)
                for Q in needed_Q
                for k in range(1, K_val)
            ]

            def _run_one(k, sb, cd, _dm, _lag, _Q):
                p, w, cost = optimize_adapter_instrumented(sb, cd, k, _dm, _lag, _Q, db_path, seq_id, K_val)
                return k, _Q, p, w, cost, len(sb)

            results = Parallel(n_jobs=-1, backend="loky", batch_size=1, pre_dispatch="all")(
                delayed(_run_one)(*t) for t in tasks
            )

            # Group results by Q and commit to DB
            for Q in needed_Q:
                q_res = [r for r in results if r[1] == Q]
                slp = np.concatenate([[0.0], np.full(K_val - 1, 0.25)])
                slw = np.concatenate([[1.0], np.full(K_val - 1, 0.35)])
                adapter_rows = []
                for k, _, p_opt, w_opt, best_cost, n_samp in q_res:
                    slp[k] = p_opt; slw[k] = w_opt
                    adapter_rows.append((seq_id, K_val, Q, k, p_opt, w_opt, n_samp, best_cost, time.time() - t_ad))

                with _conn(db_path) as conn:
                    conn.executemany("""
                        INSERT OR REPLACE INTO mode_adapters
                        (seq_id, K, Q, mode_idx, s_lp, s_lw, n_sample_blocks, mean_train_cost, training_seconds)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, adapter_rows)

                all_adapters[K_val][Q] = (slp.copy(), slw.copy())
                log(f"      Adapters K={K_val} Q={Q:02d} done in {time.time()-t_ad:.1f}s")

    log(f"    [TRAIN DONE] {seq_name} — all codebooks & adapters ready.")
    return codebooks, all_adapters, ply_files, blocks_p, nw_p, tangent_p


# ---------------------------------------------------------------------------
# ENCODING: single frame × all configs
# ---------------------------------------------------------------------------

def encode_frame(
    seq_id, seq_name, frame_idx, frame_ply, zip_path, depth,
    codebooks, all_adapters, blocks_p, nw_p,
    db_path, scratch_dir, completed_runs,
    dm, lag, B_p, experiments,
    ablation_no_selfloop=False,
):
    """
    Encode one frame across all (B_c, K, Q) operating points for this depth.
    B_p: parent block size used during training (16 for depth 9, 32 for depth 10).
    experiments: depth-specific (B_c, K) grid.
    ablation_no_selfloop: E13 ablation -- force s_lp=0 for all directional modes (k>=1),
        keeping the trained codebook direction/assignment and the trellis/gating mechanism
        untouched. With s_lp=0, AttributeGraph._self_loops_selection() selects zero nodes
        (index = int(0*N) = 0), so the resulting Laplacian is identical to the plain
        structural graph (C=0) regardless of which direction k was assigned -- this isolates
        whether the self-loop mechanism itself (not just codebook-based mode classification)
        is what drives the measured gain.
    Returns list of result dicts.
    """
    # Load and colour-convert frame
    V_raw, A_raw = load_ply_from_zip(zip_path, frame_ply)
    pc = make_pc(V_raw, A_raw, seq_name, depth, frame_idx)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    pc_V = pc.V.copy()
    pc_A = pc.A.copy()
    N_t  = len(pc_V)

    dec_master = RLGRDecider(dm, lag, alpha_rlgr=ALPHA_RLGR)

    # Per-frame parent partition (each frame has different voxel counts / Morton ordering)
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=B_p)
    N_p  = len(blocks_p)

    # Per-frame parent tangents (geometry can vary slightly per frame)
    frame_tang_p = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_p]
    g_raw_p  = np.array([t["g_raw"]  for t in frame_tang_p])
    g_norm_p = np.array([t["g_norm"] for t in frame_tang_p])

    # Neighbour weights for Viterbi
    nw_frame = np.zeros(N_p)
    for i in range(1, N_p):
        nw_frame[i] = abs(float(np.dot(frame_tang_p[i]["v3"], frame_tang_p[i-1]["v3"])))

    # Gating variance proxy (per parent block)
    parent_var = np.array([
        float(np.var(pc_A[b.metadata.start:b.metadata.end, 0]))
        if b.metadata.end > b.metadata.start else 0.0
        for b in blocks_p
    ])

    results = []
    prev_B_c = None
    blocks_c = child_tang = child_data = None
    p2c = c2p = N_c = None

    for B_c, K_locked in experiments:
        # Re-partition only when block size changes
        if B_c != prev_B_c:
            _, blocks_c = MortonBlockPartition().partition(pc, bsize=B_c)
            N_c = len(blocks_c)
            p2c = build_parent_child_map(blocks_p, blocks_c, pc_V, B_p)
            c2p = {ci: pi for pi, cl in p2c.items() for ci in cl}
            child_tang = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_c]
            child_data = []
            for b in blocks_c:
                b.init_data(pc_V, pc_A)
                child_data.append((b.Vblock.copy(), b.Ablock.copy()))
                b.clear_data()
            prev_B_c = B_c

        centroids_3d = codebooks[K_locked]

        for Q in Q_STEPS:
            if (frame_idx, B_c, K_locked, Q) in completed_runs:
                continue

            t_run = time.time()
            dec_master._set_vars(Q)
            slp, slw = all_adapters[K_locked][Q]
            if ablation_no_selfloop:
                slp = np.zeros_like(slp)

            # Sort child indices largest-first (LPT scheduling) so the slowest tasks
            # are dispatched first, minimising tail idle time across workers.
            lpt_order = sorted(range(N_c), key=lambda ci: len(child_data[ci][0]), reverse=True)

            # Parallel GFT across all child blocks (safe batching, no pre_dispatch='all' to avoid pipe deadlock)
            cr = Parallel(n_jobs=-1, backend="loky", batch_size=1)(
                delayed(_worker_gft)(
                    ci, child_data[ci][0], child_data[ci][1],
                    blocks_c[ci].metadata,
                    child_tang[ci]["v1"], child_tang[ci]["v2"],
                    child_tang[ci]["g_raw"],
                    centroids_3d, slp, slw, K_locked
                )
                for ci in lpt_order
            )
            ccd = {ci: cmap for ci, cmap in cr}

            # Aggregate costs to parent level
            pcm = np.zeros((N_p, K_locked))
            for pi in range(N_p):
                for k in range(K_locked):
                    for ci in p2c.get(pi, []):
                        cv, _, _ = dec_master._RDcost(ccd[ci].get(k))
                        pcm[pi, k] += cv

            # -----------------------------------------------------------------------
            # Encoder RDO Gating Barrier (Asymmetric Optimization):
            # Flat parent blocks (parent_var < tau_gate) have negligible gradient energy.
            # We inject an infinite cost barrier (1e18) on modes k >= 1 for flat blocks,
            # compelling Viterbi to select structural mode 0. This clusters homogeneous
            # spatial areas into long sequences of 0, minimizing CABAC signaling overhead.
            #
            # Invertibility note: The decoder is completely deterministic and never evaluates
            # tau_gate or variances. It parses the finalized integer mode sequence directly from CABAC.
            # -----------------------------------------------------------------------
            tau_gate   = ALPHA_GATE * ((Q / 32.0) ** 2)
            active_msk = parent_var >= tau_gate
            n_active   = int(active_msk.sum())
            n_gated    = N_p - n_active
            gated_cost = pcm.copy()
            for pi in range(N_p):
                if not active_msk[pi]:
                    gated_cost[pi, 1:] = 1e18

            # Viterbi Trellis Optimization
            gamma_q   = GAMMA_VAL * ((Q / 24.0) ** 2)
            mode_seq  = run_viterbi(gated_cost, K_locked, gamma_q, nw_frame)
            n_sw      = int(np.sum(mode_seq[1:] != mode_seq[:-1]))
            n_uniq    = len(np.unique(mode_seq))

            # Mode entropy
            mc = np.bincount(mode_seq, minlength=K_locked).astype(float)
            mp = mc / mc.sum()
            h_bits = float(-np.sum([p * np.log2(p + 1e-12) for p in mp if p > 0]) * len(mode_seq))

            # Select quantized coefficients
            sel_q = []
            bas_q = []
            for ci in range(N_c):
                pi  = c2p.get(ci, 0)
                sk  = mode_seq[pi]
                cs  = ccd[ci].get(sk)
                sel_q.append(np.round(cs[:, 0:1] / Q) if cs is not None else None)
                c0  = ccd[ci].get(0)
                bas_q.append(np.round(c0[:, 0:1] / Q) if c0 is not None else None)

            # RLGR compression
            rlgr_n, rlgr_raw = compress_to_bytes(sel_q, scratch_dir)
            base_n, _        = compress_to_bytes(bas_q, scratch_dir)

            # CABAC overhead on active mode labels
            if n_active > 0:
                act_lbl = np.array([mode_seq[pi] for pi in range(N_p) if active_msk[pi]])
                cabac_b, defl_b, lzma_b, bzip2_b, raw_b = evaluate_all_overhead_encoders(act_lbl, K_locked)
                modes_raw = Order1ArithmeticCoder.encode(act_lbl.tolist(), K_locked)
            else:
                cabac_b = defl_b = lzma_b = raw_b = 0
                modes_raw = b""

            # Rate
            rlgr_bits  = rlgr_n * 8
            cabac_bits = cabac_b * 8
            total_bits = rlgr_bits + cabac_bits
            base_bits  = base_n  * 8
            fbpv       = total_bits / N_t
            bbpv       = base_bits  / N_t
            delta_bpv  = fbpv - bbpv
            delta_pct  = (delta_bpv / (bbpv + 1e-12)) * 100.0

            # Parallel Decoder reconstruction → SE_Y, SE_U, SE_V
            decode_tasks = []
            for ci in range(N_c):
                pi = c2p.get(ci, 0)
                sk = mode_seq[pi]
                Vb, Ab = child_data[ci]
                qc = sel_q[ci]
                c_k = centroids_3d[sk - 1] if sk > 0 else None
                p_k = slp[sk] if sk > 0 else 0.0
                w_k = slw[sk] if sk > 0 else 1.0
                t_c = child_tang[ci]
                decode_tasks.append((ci, blocks_c[ci].metadata, Vb, Ab, qc, Q, sk, c_k, p_k, w_k, t_c))
            decode_tasks.sort(key=lambda dt: len(dt[2]), reverse=True)

            dec_results = Parallel(n_jobs=-1, backend="loky", batch_size=1)(
                delayed(_worker_decode_block)(*dt) for dt in decode_tasks
            )
            se_Y = sum(r[1] for r in dec_results)
            se_U = sum(r[2] for r in dec_results)
            se_V = sum(r[3] for r in dec_results)
            n_pts = sum(r[4] for r in dec_results)

            mse_Y  = se_Y / max(1, n_pts)
            psnr_Y = 10.0 * math.log10(255.0**2 / (mse_Y + 1e-12))

            # Parallel Baseline (mode-0) PSNR
            base_tasks = sorted(
                [
                    (ci, blocks_c[ci].metadata, child_data[ci][0], child_data[ci][1], bas_q[ci], Q)
                    for ci in range(N_c)
                ],
                key=lambda bt: len(bt[2]), reverse=True
            )
            base_dec_results = Parallel(n_jobs=-1, backend="loky", batch_size=1)(
                delayed(_worker_decode_baseline)(*bt) for bt in base_tasks
            )
            bse_Y = sum(r[1] for r in base_dec_results)
            bn_pts = sum(r[2] for r in base_dec_results)
            base_psnr_Y = 10.0 * math.log10(255.0**2 / (max(bse_Y / max(1, bn_pts), 1e-12)))

            run_s = time.time() - t_run

            # Cluster distribution rows for this frame × K
            dist_rows = []
            for mode_idx in range(K_locked):
                assigned = np.where(mode_seq == mode_idx)[0]
                if len(assigned) == 0:
                    dist_rows.append((seq_id, K_locked, frame_idx, mode_idx,
                                      0, 0.0, 0.0, 0.0, 0, 0.0))
                    continue
                cos_s = np.max(np.abs(g_norm_p[assigned] @ centroids_3d.T), axis=1) \
                    if mode_idx > 0 and len(centroids_3d) > 0 else np.ones(len(assigned))
                ng = int(np.sum(~active_msk[assigned]))
                dist_rows.append((
                    seq_id, K_locked, frame_idx, mode_idx,
                    len(assigned), float(np.mean(cos_s)), float(np.std(cos_s)),
                    float(np.mean(g_raw_p[assigned])), ng,
                    100.0 * ng / max(1, len(assigned))
                ))

            # Block-level data (only for B_c=32; used for parquet export)
            blk_rows = None
            if B_c == 32:
                blk_rows = []
                for ci in range(N_c):
                    pi  = c2p.get(ci, 0)
                    sk  = mode_seq[pi]
                    Vb, Ab = child_data[ci]
                    tang = child_tang[ci]
                    gn   = tang["g_norm"]
                    ctr  = Vb.mean(axis=0) if len(Vb) > 0 else np.zeros(3)
                    theta = float(np.arccos(np.clip(abs(gn[1]), 0.0, 1.0)) * 180.0 / np.pi)
                    qc = sel_q[ci]
                    bbits = int(qc.size) if qc is not None else 0
                    blk_rows.append({
                        "seq_id": seq_id, "frame_idx": frame_idx,
                        "K": K_locked, "q_step": Q,
                        "block_idx": ci, "parent_block_idx": pi,
                        "center_x": float(ctr[0]),
                        "center_y": float(ctr[1]),
                        "center_z": float(ctr[2]),
                        "n_voxels": len(Vb),
                        "g_raw": float(tang["g_raw"]),
                        "g_norm_x": float(gn[0]), "g_norm_y": float(gn[1]), "g_norm_z": float(gn[2]),
                        "theta_spherical": theta,
                        "assigned_mode": int(sk),
                        "is_gated": int(not active_msk[pi]),
                        "n_quant_coeffs": bbits,
                    })

            r = {
                "seq_id": seq_id, "frame_idx": frame_idx, "frame_ply": frame_ply,
                "child_b": B_c, "K": K_locked, "q_step": Q,
                "n_voxels": N_t, "n_parent_blocks": N_p, "n_child_blocks": N_c,
                "n_active_parents": n_active, "n_gated_parents": n_gated,
                "gated_pct": 100.0 * n_gated / N_p, "tau_gate": tau_gate,
                "n_trellis_switches": n_sw, "n_unique_modes": n_uniq,
                "mode_entropy_bits": h_bits,
                "rlgr_bits": rlgr_bits, "cabac_overhead_bits": cabac_bits,
                "deflate_overhead_bits": defl_b * 8,
                "lzma_overhead_bits": lzma_b * 8,
                "raw_fixed_bits": raw_b * 8,
                "baseline_rlgr_bits": base_bits,
                "frame_bpv": fbpv, "baseline_bpv": bbpv,
                "delta_bpv": delta_bpv, "delta_pct": delta_pct,
                "decoded_se_Y": se_Y, "decoded_se_U": se_U, "decoded_se_V": se_V,
                "n_decoded_pts": n_pts,
                "frame_psnr_Y": psnr_Y, "baseline_psnr_Y": base_psnr_Y,
                "bitstream_rlgr_path": None, "bitstream_modes_path": None, "bitstream_meta_path": None,
                "frame_runtime_seconds": run_s,
                "timestamp": datetime.datetime.now().isoformat(),
                # internal — not persisted to DB directly
                "_dist_rows": dist_rows,
                "_blk_rows":  blk_rows,
                "_rlgr_raw":  rlgr_raw,
                "_modes_raw": modes_raw,
                "_mode_seq":  mode_seq,
                "_active_msk": active_msk,
            }
            results.append(r)

            # Persist this single config immediately to SQLite so progress is never lost on interruption
            save_frame_results(db_path, [r], str(REPO_ROOT / "results" / "exp_e10" / "bitstreams"), seq_name, {(B_p, 24, Q)})

            log(
                f"    [{seq_name}] fr{frame_idx:04d} B={B_c:>2} K={K_locked:>2} Q={Q:>2} | "
                f"BPV {fbpv:.4f} (Δ{delta_pct:+.2f}%) | "
                f"PSNR {psnr_Y:.2f}dB | "
                f"OH {cabac_b}B | "
                f"{run_s:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Persist frame results
# ---------------------------------------------------------------------------

def save_frame_results(db_path, results, bitstream_dir, seq_name, save_bitstream_configs):
    """Write frame_results + cluster_distribution to DB; optionally write bitstreams."""
    fr_rows  = []
    dist_rows_all = []

    for r in results:
        config = (r["child_b"], r["K"], r["q_step"])
        rlgr_path = modes_path = meta_path = None

        if config in save_bitstream_configs:
            rlgr_path, modes_path, meta_path = write_bitstreams(
                bitstream_dir, seq_name, r["frame_idx"],
                r["child_b"], r["K"], r["q_step"],
                r["_rlgr_raw"], r["_modes_raw"],
                {
                    "seq": seq_name, "frame_idx": r["frame_idx"],
                    "B_c": r["child_b"], "K": r["K"], "Q": r["q_step"],
                    "n_voxels": r["n_voxels"],
                    "mode_sequence": r["_mode_seq"].tolist(),
                    "active_mask": r["_active_msk"].tolist(),
                    "s_lp": [],  # written from adapters table
                    "s_lw": [],
                }
            )

        fr_rows.append((
            r["seq_id"], r["frame_idx"], r["frame_ply"],
            r["child_b"], r["K"], r["q_step"],
            r["n_voxels"], r["n_parent_blocks"], r["n_child_blocks"],
            r["n_active_parents"], r["n_gated_parents"], r["gated_pct"], r["tau_gate"],
            r["n_trellis_switches"], r["n_unique_modes"], r["mode_entropy_bits"],
            r["rlgr_bits"], r["cabac_overhead_bits"],
            r["deflate_overhead_bits"], r["lzma_overhead_bits"], r["raw_fixed_bits"],
            r["baseline_rlgr_bits"],
            r["frame_bpv"], r["baseline_bpv"], r["delta_bpv"], r["delta_pct"],
            r["decoded_se_Y"], r["decoded_se_U"], r["decoded_se_V"],
            r["n_decoded_pts"], r["frame_psnr_Y"], r["baseline_psnr_Y"],
            rlgr_path, modes_path, meta_path,
            r["frame_runtime_seconds"], r["timestamp"]
        ))
        dist_rows_all.extend(r.get("_dist_rows", []))

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO frame_results (
                seq_id, frame_idx, frame_ply, child_b, K, q_step,
                n_voxels, n_parent_blocks, n_child_blocks,
                n_active_parents, n_gated_parents, gated_pct, tau_gate,
                n_trellis_switches, n_unique_modes, mode_entropy_bits,
                rlgr_bits, cabac_overhead_bits, deflate_overhead_bits,
                lzma_overhead_bits, raw_fixed_bits, baseline_rlgr_bits,
                frame_bpv, baseline_bpv, delta_bpv, delta_pct,
                decoded_se_Y, decoded_se_U, decoded_se_V,
                n_decoded_pts, frame_psnr_Y, baseline_psnr_Y,
                bitstream_rlgr_path, bitstream_modes_path, bitstream_meta_path,
                frame_runtime_seconds, timestamp
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, fr_rows)
        conn.executemany("""
            INSERT OR REPLACE INTO cluster_distribution
            (seq_id, K, frame_idx, mode_idx, n_blocks_assigned,
             mean_cosine_sim, std_cosine_sim, mean_g_raw, n_gated, pct_gated)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, dist_rows_all)


# ---------------------------------------------------------------------------
# Parquet block export
# ---------------------------------------------------------------------------

def append_parquet(parquet_dir, seq_name, K, Q, blk_rows):
    if not PANDAS_AVAILABLE or not blk_rows:
        return
    path = Path(parquet_dir) / f"blocks_{seq_name}_K{K:02d}_Q{Q:02d}.parquet"
    df_new = pd.DataFrame(blk_rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, compression="snappy", index=False)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_sequence(db_path, seq_id, seq_name):
    subsection(f"AGGREGATE  {seq_name}")
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT child_b, K, q_step,
                   COUNT(*) as T,
                   SUM(rlgr_bits + cabac_overhead_bits) as sb,
                   SUM(n_voxels) as sv,
                   SUM(decoded_se_Y) as sse,
                   SUM(n_decoded_pts) as sp,
                   SUM(baseline_rlgr_bits) as sbb,
                   AVG(baseline_psnr_Y) as avg_bpsnr
            FROM frame_results WHERE seq_id=?
            GROUP BY child_b, K, q_step
        """, (seq_id,)).fetchall()

    by_bk = defaultdict(list)
    for B_c, K, Q, T, sb, sv, sse, sp, sbb, avg_bpsnr in rows:
        if not sv or not sp:
            continue
        agg_bpv   = sb  / sv
        base_bpv  = sbb / sv
        agg_psnr  = 10.0 * math.log10(255.0**2 * sp / (sse + 1e-12))
        delta_bpv = agg_bpv - base_bpv
        delta_pct = (delta_bpv / (base_bpv + 1e-12)) * 100.0
        by_bk[(B_c, K)].append((Q, T, sb, sv, sse, sp, sbb,
                                  agg_bpv, base_bpv, agg_psnr, avg_bpsnr,
                                  delta_bpv, delta_pct))

    agg_rows = []
    for (B_c, K), qdata in by_bk.items():
        qdata.sort(key=lambda x: x[0])
        bpv_t  = [d[7]  for d in qdata]
        psnr_t = [d[9]  for d in qdata]
        bpv_a  = [d[8]  for d in qdata]
        psnr_a = [d[10] for d in qdata]
        bd     = bdrate(bpv_a, psnr_a, bpv_t, psnr_t) if len(qdata) == 4 else float("nan")

        log(
            f"  {seq_name} B={B_c:>2} K={K:>2} | "
            f"BPV=[{' '.join(f'{d[7]:.4f}' for d in qdata)}] | "
            f"PSNR=[{' '.join(f'{d[9]:.1f}' for d in qdata)}] | "
            f"BD-Rate={bd:+.3f}%"
        )

        for d in qdata:
            Q, T, sb, sv, sse, sp, sbb, agg_bpv, base_bpv, agg_psnr, avg_bpsnr, delta_bpv, delta_pct = d
            agg_rows.append((
                seq_id, B_c, K, Q, T, sb, sv, sse, sp, sbb,
                agg_bpv, base_bpv, agg_psnr, avg_bpsnr,
                delta_bpv, delta_pct, bd,
                datetime.datetime.now().isoformat()
            ))

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO aggregate_results
            (seq_id, child_b, K, q_step, T_frames,
             sum_total_bits, sum_voxels, sum_se_Y, sum_pts, sum_baseline_bits,
             aggregate_bpv, baseline_bpv, aggregate_psnr_Y, baseline_psnr_Y,
             delta_bpv, delta_pct, bd_rate_pct, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, agg_rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_e10_benchmark(
    db_path: str = None,
    overwrite: bool = False,
    sequences_filter: list = None,
    stride: int = 1,
    include_b32: bool = True,
    child_sizes: set = None,
    k_values: set = None,
    ablation_no_selfloop: bool = False,
):
    if db_path is None:
        db_path = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")

    frames_dir    = REPO_ROOT / "res" / "frames"
    scratch_dir   = str(REPO_ROOT / "tmp" / "e10_scratch")
    bitstream_dir = str(REPO_ROOT / "results" / "exp_e10" / "bitstreams")
    parquet_dir   = str(REPO_ROOT / "results" / "exp_e10" / "block_parquet")
    for d in [scratch_dir, bitstream_dir, parquet_dir]:
        os.makedirs(d, exist_ok=True)

    section("PCADC E10 — Multi-Sequence Offline Codebook Verification")
    log(f"  DB          : {db_path}")
    log(f"  Bitstreams  : {bitstream_dir}")
    log(f"  Parquet     : {parquet_dir}")
    log(f"  PyRLGR      : {'YES' if PYRLGR_AVAILABLE else 'NO (zlib fallback)'}")
    log(f"  Pandas      : {'YES' if PANDAS_AVAILABLE else 'NO (parquet skipped)'}")
    log(f"  Sequences   : {len(SEQUENCES)}")
    log(f"  Stride      : {stride} ({'all frames' if stride == 1 else f'every {stride}th frame — DB is append-only, omitted frames can be filled in later without --stride'})")
    log(f"  B_c=32      : {'Enabled for depth-10' if include_b32 else 'Disabled (running only B_c in {16, 8, 4})'}")
    if child_sizes:
        log(f"  Child sizes : restricted to {sorted(child_sizes)} (grid-reduction override)")
    if k_values:
        log(f"  K values    : restricted to {sorted(k_values)} (grid-reduction override)")

    init_db(db_path, overwrite=overwrite)

    params = load_experiment_config(REPO_ROOT / "config" / "base_config.yaml")
    dm  = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    t_total = time.time()

    for seq_num, (seq_name, zip_fn, depth) in enumerate(SEQUENCES):
        if sequences_filter and seq_name not in sequences_filter:
            continue

        zip_path = frames_dir / zip_fn
        if not zip_path.exists():
            log(f"[SKIP] {zip_path} not found")
            continue

        # Depth-adaptive configuration for this sequence
        B_p         = get_bp(depth)
        experiments = get_experiments(depth, include_b32=include_b32)
        if child_sizes:
            experiments = [(b, k) for b, k in experiments if b in child_sizes]
        if k_values:
            experiments = [(b, k) for b, k in experiments if k in k_values]
        n_configs   = len(experiments) * len(Q_STEPS)

        section(f"SEQUENCE {seq_num+1}/{len(SEQUENCES)}: {seq_name}  (depth={depth}, B_p={B_p}, {n_configs} configs/frame)")

        seq_id = get_or_create_seq_id(db_path, seq_name, depth, zip_path)
        log(f"  seq_id={seq_id}")

        # ── TRAIN ────────────────────────────────────────────────────────
        t_train = time.time()
        codebooks, all_adapters, ply_files, blocks_p, nw_p, _ = train_sequence(
            seq_id, seq_name, depth, zip_path, db_path, scratch_dir, dm, lag,
            B_p=B_p, experiments=experiments
        )
        log(f"  Training total: {time.time()-t_train:.1f}s | {len(ply_files)} frames")

        # Default bitstream config: largest B_c with K=24, all Q (written during encoding)
        default_bs_configs = {(B_p, 24, Q) for Q in Q_STEPS}

        # Frame list: all encode frames (skip frame 0, which was training)
        all_encode_frames = list(enumerate(ply_files[1:], start=1))
        # Apply stride — select every Nth frame by original index, preserving frame_idx
        strided_frames = all_encode_frames[::stride]
        T_enc = len(strided_frames)

        completed = get_completed_frame_keys(db_path, seq_id)
        log(f"  Encoding {T_enc}/{len(all_encode_frames)} frames (stride={stride}) | {len(completed)} rows already in DB")

        t_enc = time.time()
        done_frames = 0

        for frame_idx, frame_ply in strided_frames:
            # Skip if all configs for this frame already done
            frame_done = sum(1 for k in completed if k[0] == frame_idx)
            if frame_done == n_configs:
                done_frames += 1
                if done_frames % 50 == 0:
                    log(f"  [SKIP] {done_frames}/{T_enc} frames already complete")
                continue

            results = encode_frame(
                seq_id, seq_name, frame_idx, frame_ply, zip_path, depth,
                codebooks, all_adapters, blocks_p, nw_p,
                db_path, scratch_dir, completed, dm, lag,
                B_p=B_p, experiments=experiments,
                ablation_no_selfloop=ablation_no_selfloop,
            )

            # Persist frame results + optional bitstreams for default configs
            save_frame_results(db_path, results, bitstream_dir, seq_name, default_bs_configs)

            # Block-level parquet (largest B_c only — B_p)
            if PANDAS_AVAILABLE:
                by_kq = defaultdict(list)
                for r in results:
                    if r["child_b"] == B_p and r.get("_blk_rows"):
                        by_kq[(r["K"], r["q_step"])].extend(r["_blk_rows"])
                for (K, Q), rows in by_kq.items():
                    append_parquet(parquet_dir, seq_name, K, Q, rows)

            for r in results:
                completed.add((r["frame_idx"], r["child_b"], r["K"], r["q_step"]))

            done_frames += 1
            elapsed = time.time() - t_enc
            eta_s   = (elapsed / done_frames) * (T_enc - done_frames)
            log(
                f"  [PROGRESS] {done_frames}/{T_enc} frames (stride={stride}) | "
                f"elapsed {elapsed/60:.1f}min | ETA {eta_s/60:.1f}min"
            )

        log(f"  [ENCODE DONE] {seq_name} | {done_frames} frames | {(time.time()-t_enc)/60:.1f}min")

        # ── AGGREGATE ───────────────────────────────────────────────────
        aggregate_sequence(db_path, seq_id, seq_name)

        # ── BITSTREAMS for best config (all Q) ──────────────────────────
        # Skipped entirely for the E13 ablation: this block only re-encodes to write example
        # bitstream files to disk, doubling runtime for no benefit to a diagnostic run that only
        # needs the aggregate_results/frame_results numbers already written above.
        best = None
        if not ablation_no_selfloop:
            with _conn(db_path) as conn:
                best = conn.execute("""
                    SELECT child_b, K, AVG(delta_pct) as avg_delta
                    FROM aggregate_results WHERE seq_id=?
                    GROUP BY child_b, K ORDER BY avg_delta ASC LIMIT 1
                """, (seq_id,)).fetchone()

        if best:
            best_B, best_K, best_d = best
            if (best_B, best_K) != (B_p, 24):   # already written above for default (B_p, 24)
                log(f"  Best config: B={best_B} K={best_K} (avg Δ={best_d:.2f}%) — saving bitstreams")
                extra_bs = {(best_B, best_K, Q) for Q in Q_STEPS}
                # Re-encode only the strided frames that were actually encoded
                for frame_idx, frame_ply in strided_frames:
                    results = encode_frame(
                        seq_id, seq_name, frame_idx, frame_ply, zip_path, depth,
                        codebooks, all_adapters, blocks_p, nw_p,
                        db_path, scratch_dir,
                        # Pass all as "completed" except the best config so
                        # only best_B/best_K/Q rows are re-computed
                        completed - {(frame_idx, best_B, best_K, Q) for Q in Q_STEPS},
                        dm, lag, B_p=B_p, experiments=experiments,
                        ablation_no_selfloop=ablation_no_selfloop,
                    )
                    # Write only the best config bitstreams
                    for r in results:
                        config = (r["child_b"], r["K"], r["q_step"])
                        if config not in extra_bs:
                            continue
                        rp, mp, jp = write_bitstreams(
                            bitstream_dir, seq_name, r["frame_idx"],
                            r["child_b"], r["K"], r["q_step"],
                            r["_rlgr_raw"], r["_modes_raw"],
                            {"seq": seq_name, "frame_idx": r["frame_idx"],
                             "B_c": r["child_b"], "K": r["K"], "Q": r["q_step"],
                             "n_voxels": r["n_voxels"],
                             "mode_sequence": r["_mode_seq"].tolist(),
                             "active_mask": r["_active_msk"].tolist()}
                        )
                        with _conn(db_path) as conn:
                            conn.execute("""
                                UPDATE frame_results
                                SET bitstream_rlgr_path=?, bitstream_modes_path=?, bitstream_meta_path=?
                                WHERE seq_id=? AND frame_idx=? AND child_b=? AND K=? AND q_step=?
                            """, (rp, mp, jp, seq_id, frame_idx, r["child_b"], r["K"], r["q_step"]))

        log(f"  [SEQUENCE COMPLETE] {seq_name} in {(time.time()-t_train)/60:.1f}min total")

    section(f"E10 COMPLETE — Total: {(time.time()-t_total)/3600:.2f}h | DB: {db_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCADC E10: Multi-Sequence Offline Codebook Verification Benchmark"
    )
    parser.add_argument("--db-path",   type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sequences", type=str, default=None,
        help="Comma-separated sequence names to run, e.g. longdress,soldier"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help=(
            "Encode every Nth frame (default: 1 = all frames). "
            "The DB is append-only: strided and non-strided runs share the same DB safely."
        )
    )
    parser.add_argument(
        "--no-b32", action="store_true",
        help="Skip B_c=32 (runs only B_c in {16, 8, 4}). Dramatically accelerates execution."
    )
    parser.add_argument(
        "--child-sizes", type=str, default=None,
        help=(
            "Comma-separated child block sizes to run, e.g. '4,8' (subset of {4,8,16,32}). "
            "Grid-reduction override for time-boxed multi-frame runs — omit to use the full "
            "depth-adaptive grid (see --no-b32 for dropping just B_c=32)."
        )
    )
    parser.add_argument(
        "--k-values", type=str, default=None,
        help=(
            "Comma-separated codebook sizes K to run, e.g. '24' or '16,24' (subset of "
            "{4,8,16,24,30}). Grid-reduction override — omit to use the full K grid."
        )
    )
    parser.add_argument(
        "--q-values", type=str, default=None,
        help=(
            "Comma-separated quantization steps to run, e.g. '16,20,28,40' -- overrides the "
            "default Q_STEPS=[24,32,48,64] for this invocation only (purely a configuration "
            "override, applied before any encode/decode logic runs; does not change how a "
            "given Q is processed). Omit to use the default grid."
        )
    )
    parser.add_argument(
        "--ablation-no-selfloop", action="store_true",
        help=(
            "E13 ablation: force s_lp=0 for every directional mode (k>=1), keeping the "
            "trained codebook/trellis/gating untouched. Isolates whether the self-loop "
            "mechanism itself (not just codebook-based mode classification) drives the "
            "measured gain. Requires --db-path pointing at a DB pre-seeded with the "
            "sequences/cluster_centroids/mode_adapters tables from the main run (so the "
            "codebook is reused, not retrained) — see scripts/seed_ablation_db.py."
        )
    )
    args = parser.parse_args()

    if args.q_values:
        # Rebinds the module-level Q_STEPS global (referenced, not copied, by every function
        # that uses it) -- a pure startup-time configuration override, no encode/decode logic
        # touched. Must happen before run_e10_benchmark() is called.
        Q_STEPS = sorted(int(x) for x in args.q_values.split(","))
        log(f"  Q values    : overridden to {Q_STEPS} (--q-values)")

    seq_filter = [s.strip() for s in args.sequences.split(",")] if args.sequences else None
    child_sizes = {int(x) for x in args.child_sizes.split(",")} if args.child_sizes else None
    k_values = {int(x) for x in args.k_values.split(",")} if args.k_values else None
    run_e10_benchmark(
        db_path=args.db_path,
        overwrite=args.overwrite,
        sequences_filter=seq_filter,
        stride=args.stride,
        include_b32=not args.no_b32,
        child_sizes=child_sizes,
        k_values=k_values,
        ablation_no_selfloop=args.ablation_no_selfloop,
    )

