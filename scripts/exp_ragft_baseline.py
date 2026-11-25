#!/usr/bin/env python3
"""
=============================================================================
RA-GFT baseline benchmark: real bpv/PSNR_Y numbers on the same 3-sequence sanity subset used
for the RAHT baseline, comparable to PCADC's own E10 benchmark on identical frames/Q/B_c.
=============================================================================

Unlike the RAHT comparison (which turned out not to be the meaningful test -- RAHT trades
rate-distortion performance for near-zero complexity by design), RA-GFT is a genuine
multiresolution graph transform, already a stronger baseline by construction, and PCADC's own
graph/color code is a direct descendant of the RA-GFT reference implementation (see
docs/thesis/ra_gft_compatibility_note.md). This is the real, meaningful external comparison.

Applies RA-GFT WITHIN each PCADC B_c=16 block (same spatial partition, frames, Q grid, entropy
coder as every other benchmark in this repo), via a genuine 4-level cascade (bsize_per_level =
[2,2,2,2], reaching the block's full 16-voxel extent) -- i.e. "block-local RA-GFT", the same
kind of disclosed simplification already used for the RAHT baseline: this gives RA-GFT its real
multiresolution structure WITHIN each block, but not the full-cloud cross-block redundancy
capture the original paper's own results demonstrate (which cascades across the WHOLE point
cloud, not per fixed-size block). Expect this to understate full-cloud RA-GFT's true ceiling,
same direction of caveat as the RAHT baseline.

Reuses E9/E10 primitives directly: SEQUENCES, load_ply_from_zip, make_pc, Q_STEPS,
compress_to_bytes, bdrate() from exp_e10_multisequence_benchmark.py -- same PyRLGR entropy
coder PCADC itself uses, for a fair bpv comparison.

Writes to a NEW, separate results DB (results/ragft_baseline.db) -- never opens
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

from pcadc.ragft import region_adaptive_gft, inverse_region_adaptive_gft
from pcadc.color import Colourist

from exp_e10_multisequence_benchmark import (
    SEQUENCES, load_ply_from_zip, make_pc, list_ply_files_in_zip,
    compress_to_bytes, bdrate, Q_STEPS,
)

DEFAULT_SEQUENCES = ("longdress", "phil9", "phil10")
DEFAULT_B_C = 16
DEFAULT_BSIZE_PER_LEVEL = (2, 2, 2, 2)  # 2^4 = 16, matches DEFAULT_B_C
DEFAULT_STRIDE = 50
DEFAULT_DB = REPO_ROOT / "results" / "ragft_baseline.db"
SCRATCH_DIR = REPO_ROOT / "scratch" / "ragft_baseline"


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Database (same shape as raht_baseline's, "_ragft" suffix)
# ---------------------------------------------------------------------------

SCHEMA = {
    "sequences_ragft": """
        CREATE TABLE IF NOT EXISTS sequences_ragft (
            seq_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT UNIQUE,
            depth    INTEGER,
            zip_path TEXT,
            created_at TEXT
        )""",
    "frame_results_ragft": """
        CREATE TABLE IF NOT EXISTS frame_results_ragft (
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
    "aggregate_results_ragft": """
        CREATE TABLE IF NOT EXISTS aggregate_results_ragft (
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
        c.execute("SELECT seq_id FROM sequences_ragft WHERE name=?", (name,))
        row = c.fetchone()
        if row:
            return row[0]
        c.execute(
            "INSERT INTO sequences_ragft (name, depth, zip_path, created_at) VALUES (?,?,?,?)",
            (name, depth, str(zip_path), datetime.datetime.now().isoformat())
        )
        return c.lastrowid


def get_completed_frame_keys(db_path, seq_id):
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT frame_idx, child_b, q_step FROM frame_results_ragft WHERE seq_id=?",
            (seq_id,)
        ).fetchall()
    return set(rows)


# ---------------------------------------------------------------------------
# Per-frame encode
# ---------------------------------------------------------------------------

def encode_frame_ragft(seq_id, seq_name, frame_idx, frame_ply, zip_path, depth,
                        db_path, scratch_dir, completed_keys, B_c=DEFAULT_B_C,
                        bsize_per_level=DEFAULT_BSIZE_PER_LEVEL, q_steps=None):
    """Loads one frame, partitions into B_c blocks, applies block-local RA-GFT (a genuine
    multi-level cascade within each block) + RLGR-codes the Y channel at each Q. Returns list
    of result dicts (one per Q), already persisted (crash-safe, mirrors E10/RAHT's
    resumability pattern)."""
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

    block_data = []  # (Vb, Yb) per block
    n_voxels = 0
    for blk in blocks:
        blk.init_data(pc.V, pc.A)
        Vb, Ab = blk.get_data()
        Yb = Ab[:, 0:1]  # keep 2D (n,1) -- region_adaptive_gft expects (N,C)
        block_data.append((Vb, Yb))
        n_voxels += Vb.shape[0]

    results = []
    for Q in pending_qs:
        t0 = time.time()
        quant_list = []
        se_y_total = 0.0
        n_pts_total = 0
        for Vb, Yb in block_data:
            coeffs, _ = region_adaptive_gft(Vb, Yb, list(bsize_per_level))
            quant = np.round(coeffs / Q)
            quant_list.append(quant)

            dequant = quant * Q
            rec = inverse_region_adaptive_gft(Vb, dequant, list(bsize_per_level))
            se_y_total += float(np.sum((Yb - rec) ** 2))
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
            INSERT OR REPLACE INTO frame_results_ragft
            (seq_id, frame_idx, frame_ply, child_b, q_step, n_voxels, n_blocks,
             rlgr_bits, frame_bpv, decoded_se_Y, n_decoded_pts, frame_psnr_Y,
             frame_runtime_seconds, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [(r["seq_id"], r["frame_idx"], r["frame_ply"], r["child_b"], r["q_step"],
               r["n_voxels"], r["n_blocks"], r["rlgr_bits"], r["frame_bpv"],
               r["decoded_se_Y"], r["n_decoded_pts"], r["frame_psnr_Y"],
               r["frame_runtime_seconds"], r["timestamp"]) for r in results])

    return results


def aggregate_sequence_ragft(db_path, seq_id, seq_name, B_c=DEFAULT_B_C, q_steps=None):
    q_steps = q_steps or Q_STEPS
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT q_step, COUNT(*), SUM(rlgr_bits), SUM(n_voxels), SUM(decoded_se_Y), SUM(n_decoded_pts)
            FROM frame_results_ragft WHERE seq_id=? AND child_b=?
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
            INSERT OR REPLACE INTO aggregate_results_ragft
            (seq_id, child_b, q_step, T_frames, sum_rlgr_bits, sum_voxels, sum_se_Y, sum_pts,
             aggregate_bpv, aggregate_psnr_Y, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, agg_rows)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_ragft_benchmark(db_path=None, sequences_filter=DEFAULT_SEQUENCES,
                         B_c=DEFAULT_B_C, bsize_per_level=DEFAULT_BSIZE_PER_LEVEL,
                         q_steps=None, stride=DEFAULT_STRIDE):
    db_path = str(db_path or DEFAULT_DB)
    q_steps = q_steps or Q_STEPS
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    init_db(db_path)

    frames_dir = REPO_ROOT / "res" / "frames"
    seqs = [s for s in SEQUENCES if s[0] in set(sequences_filter)]
    log(f"RA-GFT baseline benchmark: sequences={[s[0] for s in seqs]} B_c={B_c} "
        f"bsize_per_level={bsize_per_level} Q={q_steps} stride={stride}")

    for name, zip_fn, depth in seqs:
        zip_path = frames_dir / zip_fn
        if not zip_path.exists():
            log(f"  [SKIP] {name}: {zip_path} not found")
            continue

        seq_id = get_or_create_seq_id(db_path, name, depth, zip_path)
        ply_files = list_ply_files_in_zip(zip_path)
        # Same frame selection as E10/RAHT for these sequences.
        strided = list(enumerate(ply_files[1:], start=1))[::stride]

        completed_keys = get_completed_frame_keys(db_path, seq_id)
        log(f"[{name}] {len(strided)} frames to evaluate: {[i for i, _ in strided]}")

        for frame_idx, frame_ply in strided:
            encode_frame_ragft(seq_id, name, frame_idx, frame_ply, zip_path, depth,
                                db_path, SCRATCH_DIR, completed_keys, B_c=B_c,
                                bsize_per_level=bsize_per_level, q_steps=q_steps)

        aggregate_sequence_ragft(db_path, seq_id, name, B_c=B_c, q_steps=q_steps)

    log("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default=None)
    ap.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    ap.add_argument("--b-c", type=int, default=DEFAULT_B_C)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = ap.parse_args()
    run_ragft_benchmark(db_path=args.db_path, sequences_filter=args.sequences,
                         B_c=args.b_c, stride=args.stride)
