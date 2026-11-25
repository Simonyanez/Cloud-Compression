#!/usr/bin/env python3
"""
=============================================================================
RAHT baseline benchmark: real bpv/PSNR_Y numbers on a small sanity subset,
comparable to PCADC's own E10 benchmark on identical frames/Q/B_c.
=============================================================================

Purpose: the thesis explicitly deferred any comparison against an external reference codec
(RAHT/G-PCC). This is that comparison, implemented from scratch (src/pcadc/raht.py) and
scoped as a feasibility/sanity test -- NOT the full 14-sequence sweep E10 ran for PCADC.

RAHT is a single, deterministic, non-adaptive transform (see src/pcadc/raht.py module
docstring for the full disclosed-approximation list: this is BLOCK-LOCAL RAHT, not full-cloud
RAHT as in the real G-PCC/TMC13 reference software). It has no training phase, no codebook,
no K sweep, no Viterbi/RDO mode search -- unlike E10's loop, the only axis swept here is Q.

Reuses E9/E10 primitives directly rather than re-deriving them: SEQUENCES, load_ply_from_zip,
make_pc, Q_STEPS, bdrate() from exp_e10_multisequence_benchmark.py; compress_to_bytes (the
same PyRLGR zigzag entropy coder PCADC itself uses, for a fair bpv comparison) also from
exp_e10_multisequence_benchmark.py.

Writes to a NEW, separate results DB (results/raht_baseline.db) -- never opens
results/exp_e10_multisequence.db for writing.
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
import sqlite3
import datetime
import argparse
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from pcadc.raht import RAHTProcessor
from pcadc.color import Colourist

from exp_e10_multisequence_benchmark import (
    SEQUENCES, load_ply_from_zip, make_pc, list_ply_files_in_zip,
    compress_to_bytes, bdrate, Q_STEPS,
)

DEFAULT_SEQUENCES = ("longdress", "phil9", "phil10")
DEFAULT_B_C = 16
DEFAULT_STRIDE = 50
DEFAULT_DB = REPO_ROOT / "results" / "raht_baseline.db"
SCRATCH_DIR = REPO_ROOT / "scratch" / "raht_baseline"


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = {
    "sequences_raht": """
        CREATE TABLE IF NOT EXISTS sequences_raht (
            seq_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT UNIQUE,
            depth    INTEGER,
            zip_path TEXT,
            created_at TEXT
        )""",
    "frame_results_raht": """
        CREATE TABLE IF NOT EXISTS frame_results_raht (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            seq_id    INTEGER,
            frame_idx INTEGER,
            frame_ply TEXT,
            child_b   INTEGER,
            q_step    INTEGER,
            n_voxels  INTEGER,
            n_blocks  INTEGER,
            rlgr_bits INTEGER,
            frame_bpv REAL,
            decoded_se_Y   REAL,
            n_decoded_pts  INTEGER,
            frame_psnr_Y   REAL,
            frame_runtime_seconds REAL,
            timestamp TEXT,
            UNIQUE(seq_id, frame_idx, child_b, q_step)
        )""",
    "aggregate_results_raht": """
        CREATE TABLE IF NOT EXISTS aggregate_results_raht (
            agg_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            seq_id   INTEGER,
            child_b  INTEGER,
            q_step   INTEGER,
            T_frames INTEGER,
            sum_rlgr_bits INTEGER,
            sum_voxels    INTEGER,
            sum_se_Y REAL,
            sum_pts  INTEGER,
            aggregate_bpv    REAL,
            aggregate_psnr_Y REAL,
            completed_at TEXT,
            UNIQUE(seq_id, child_b, q_step)
        )""",
}


def init_db(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for ddl in SCHEMA.values():
        c.execute(ddl)
    conn.commit()
    conn.close()
    log(f"[DB] Ready: {db_path}")


def _conn(db_path):
    return sqlite3.connect(db_path, timeout=30)


def get_or_create_seq_id(db_path, name, depth, zip_path):
    with _conn(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT seq_id FROM sequences_raht WHERE name=?", (name,))
        row = c.fetchone()
        if row:
            return row[0]
        c.execute(
            "INSERT INTO sequences_raht (name, depth, zip_path, created_at) VALUES (?,?,?,?)",
            (name, depth, str(zip_path), datetime.datetime.now().isoformat())
        )
        return c.lastrowid


def get_completed_frame_keys(db_path, seq_id):
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT frame_idx, child_b, q_step FROM frame_results_raht WHERE seq_id=?",
            (seq_id,)
        ).fetchall()
    return set(rows)


# ---------------------------------------------------------------------------
# Per-frame encode
# ---------------------------------------------------------------------------

def encode_frame_raht(seq_id, seq_name, frame_idx, frame_ply, zip_path, depth,
                       db_path, scratch_dir, completed_keys, B_c=DEFAULT_B_C,
                       q_steps=None):
    """Loads one frame, partitions into B_c blocks, RAHT-transforms + RLGR-codes the Y
    channel at each Q in q_steps. Returns list of result dicts (one per Q), already
    persisted (crash-safe, mirrors E10's resumability pattern)."""
    q_steps = q_steps or Q_STEPS
    pending_qs = [q for q in q_steps if (frame_idx, B_c, q) not in completed_keys]
    if not pending_qs:
        return []

    V, A_rgb = load_ply_from_zip(zip_path, frame_ply)
    pc = make_pc(V, A_rgb, seq_name, depth, frame_idx)
    pc.transform_attributes(Colourist()._RGBtoYUV)

    from pcadc.pointcloud import MortonBlockPartition
    part = MortonBlockPartition()
    _, blocks = part.partition(pc, bsize=B_c)

    # Build the RAHT tree once per block (geometry-only, reused across all Q -- exactly
    # analogous to how PCADC's own decoder reconstructs its GFT basis from geometry alone).
    proc = RAHTProcessor(bsize=B_c)
    block_data = []  # (Vb, Yb, tree) per block
    n_voxels = 0
    for blk in blocks:
        blk.init_data(pc.V, pc.A)
        Vb, Ab = blk.get_data()
        Yb = Ab[:, 0]
        _, _, tree = proc.compute(Vb, Yb)
        block_data.append((Vb, Yb, tree))
        n_voxels += Vb.shape[0]

    results = []
    for Q in pending_qs:
        t0 = time.time()
        quant_list = []
        se_y_total = 0.0
        n_pts_total = 0
        for Vb, Yb, tree in block_data:
            _, coeffs, _ = proc.compute(Vb, Yb)
            quant = np.round(coeffs / Q)
            quant_list.append(quant)

            dequant = quant * Q
            rec_per_leaf = proc.inverse(tree, dequant)
            rec_per_point = rec_per_leaf[tree.dup_inverse_idx]
            se_y_total += float(np.sum((Yb - rec_per_point) ** 2))
            n_pts_total += Yb.shape[0]

        n_bytes, _ = compress_to_bytes(quant_list, str(scratch_dir))
        rlgr_bits = n_bytes * 8
        frame_bpv = rlgr_bits / max(1, n_voxels)
        psnr_y = 10.0 * math.log10(255.0 ** 2 / max(se_y_total / max(1, n_pts_total), 1e-12))
        runtime_s = time.time() - t0

        row = dict(
            seq_id=seq_id, frame_idx=frame_idx, frame_ply=frame_ply,
            child_b=B_c, q_step=Q,
            n_voxels=n_voxels, n_blocks=len(blocks),
            rlgr_bits=rlgr_bits, frame_bpv=frame_bpv,
            decoded_se_Y=se_y_total, n_decoded_pts=n_pts_total, frame_psnr_Y=psnr_y,
            frame_runtime_seconds=runtime_s,
            timestamp=datetime.datetime.now().isoformat(),
        )
        results.append(row)
        log(f"  {seq_name} frame={frame_idx} B_c={B_c} Q={Q:>2d} | "
            f"bpv={frame_bpv:.4f} PSNR_Y={psnr_y:.2f}dB | {runtime_s:.2f}s")

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO frame_results_raht
            (seq_id, frame_idx, frame_ply, child_b, q_step, n_voxels, n_blocks,
             rlgr_bits, frame_bpv, decoded_se_Y, n_decoded_pts, frame_psnr_Y,
             frame_runtime_seconds, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [(r["seq_id"], r["frame_idx"], r["frame_ply"], r["child_b"], r["q_step"],
               r["n_voxels"], r["n_blocks"], r["rlgr_bits"], r["frame_bpv"],
               r["decoded_se_Y"], r["n_decoded_pts"], r["frame_psnr_Y"],
               r["frame_runtime_seconds"], r["timestamp"]) for r in results])

    return results


def aggregate_sequence_raht(db_path, seq_id, seq_name, B_c=DEFAULT_B_C, q_steps=None):
    q_steps = q_steps or Q_STEPS
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT q_step, COUNT(*), SUM(rlgr_bits), SUM(n_voxels), SUM(decoded_se_Y), SUM(n_decoded_pts)
            FROM frame_results_raht WHERE seq_id=? AND child_b=?
            GROUP BY q_step
        """, (seq_id, B_c)).fetchall()

    agg_rows = []
    for Q, T, sb, sv, sse, sp in rows:
        if not sv or not sp:
            continue
        agg_bpv = sb / sv
        agg_psnr = 10.0 * math.log10(255.0 ** 2 * sp / (sse + 1e-12))
        agg_rows.append((seq_id, B_c, Q, T, sb, sv, sse, sp, agg_bpv, agg_psnr,
                          datetime.datetime.now().isoformat()))
        log(f"  AGG {seq_name} B_c={B_c} Q={Q:>2d} | bpv={agg_bpv:.4f} PSNR_Y={agg_psnr:.2f}dB | T={T} frames")

    with _conn(db_path) as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO aggregate_results_raht
            (seq_id, child_b, q_step, T_frames, sum_rlgr_bits, sum_voxels, sum_se_Y, sum_pts,
             aggregate_bpv, aggregate_psnr_Y, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, agg_rows)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_raht_benchmark(db_path=None, sequences_filter=DEFAULT_SEQUENCES,
                        B_c=DEFAULT_B_C, q_steps=None, stride=DEFAULT_STRIDE):
    db_path = str(db_path or DEFAULT_DB)
    q_steps = q_steps or Q_STEPS
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    init_db(db_path)

    frames_dir = REPO_ROOT / "res" / "frames"
    seqs = [s for s in SEQUENCES if s[0] in set(sequences_filter)]
    log(f"RAHT baseline benchmark: sequences={[s[0] for s in seqs]} B_c={B_c} Q={q_steps} stride={stride}")

    for name, zip_fn, depth in seqs:
        zip_path = frames_dir / zip_fn
        if not zip_path.exists():
            log(f"  [SKIP] {name}: {zip_path} not found")
            continue

        seq_id = get_or_create_seq_id(db_path, name, depth, zip_path)
        ply_files = list_ply_files_in_zip(zip_path)
        # Same frame selection E10 used for these sequences: stride over ply_files[1:]
        # (frame 0 is E10's training frame; not used here since RAHT has no training phase,
        # but excluded anyway to land on the identical physical evaluation frames E10 scored).
        strided = list(enumerate(ply_files[1:], start=1))[::stride]

        completed_keys = get_completed_frame_keys(db_path, seq_id)
        log(f"[{name}] {len(strided)} frames to evaluate: {[i for i, _ in strided]}")

        for frame_idx, frame_ply in strided:
            encode_frame_raht(seq_id, name, frame_idx, frame_ply, zip_path, depth,
                               db_path, SCRATCH_DIR, completed_keys, B_c=B_c, q_steps=q_steps)

        aggregate_sequence_raht(db_path, seq_id, name, B_c=B_c, q_steps=q_steps)

    log("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--b-c", type=int, default=DEFAULT_B_C)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = ap.parse_args()
    run_raht_benchmark(db_path=args.db_path, sequences_filter=args.sequences,
                        B_c=args.b_c, stride=args.stride)
