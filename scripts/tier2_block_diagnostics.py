#!/usr/bin/env python3
"""
Tier 2: decode-only per-CHILD-block diagnostic pass, formalized from a session scratchpad script
(see git history / thesis session notes) and extended for the block-level dependence figures of
resultados.tex, Sección~\\ref{sec:e10_block_dependence} (item 6 of the E10 results revision).

Reuses the production pipeline's own functions (no reimplementation of the codec) but skips the
expensive K-way mode SEARCH: the mode each parent block uses is already known (from the
already-computed "best config" bitstream meta.json -- real Viterbi output, not re-derived here).
For each child block this only computes the forward GFT for the two modes that matter for a
gain/loss comparison -- the block's already-assigned mode, and mode 0 (baseline) -- then decodes
both and keeps:
  - the real per-block SE_Y (normally computed transiently and discarded) -> real PSNR gain,
  - real spectral energy compaction (fraction of Y-channel GFT coefficient energy in the first
    min(4,N) coefficients -- same definition as scripts/oracle_real_rlgr_and_compaction.py's
    `_quant_metrics`) for BOTH modes -> real Delta-compaction vs. baseline,
  - point_count (n_pts), the block-level analogue of the sequence-level log10(N_vox) driver
    already used in e10_driver_correlations.png.

Both are aggregated to PARENT-block level (energy-weighted for compaction, SE/n_pts-weighted for
PSNR, matching how the production benchmark computes frame-level PSNR from per-block SE).

Scope: 6 sequences (2 per family: longdress/loot 8iVFB, phil9/ricardo9 depth-9,
phil10/ricardo10 depth-10), looped over every frame_idx that has a K=24,Q=24 bitstream meta.json
available (multi-frame, not just frame_idx=1) -- typically frame_idx in {1,51,101,151,201} or a
subset thereof depending on how far that sequence's follow-up sweep got.

Usage: `python3 scripts/tier2_block_diagnostics.py [seq ...]` (default: the 6-sequence scope).
Output: one consolidated `scripts/_tier2_block_records.json` (list of per-parent-block dicts,
across all sequences x frames processed) plus a `scripts/_tier2_summary.json` run manifest.
"""
import os, sys, json, math, zipfile
os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sqlite3
from pathlib import Path
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcadc.pointcloud import MortonBlockPartition
from pcadc.blocks import Block
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.color import Colourist, Approximator

from exp_e9_multi_entropy_suite import extract_block_tangent, build_parent_child_map, project_slope
from exp_e10_multisequence_benchmark import (
    load_ply_from_zip, make_pc, get_bp, SEQUENCES,
    _worker_decode_block, _worker_decode_baseline,
)

DB = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
BS_DIR = REPO_ROOT / "results" / "exp_e10" / "bitstreams"
FRAMES_DIR = REPO_ROOT / "res" / "frames"
OUT_RECORDS = REPO_ROOT / "scripts" / "_tier2_block_records.json"
OUT_SUMMARY = REPO_ROOT / "scripts" / "_tier2_summary.json"

N_JOBS = 4  # capped: coexist with other work on the same machine

# 2 sequences per family: 8iVFB, MVUB depth-9, MVUB depth-10 (AskUserQuestion-confirmed scope).
TARGET_SEQUENCES = ["longdress", "loot", "phil9", "ricardo9", "phil10", "ricardo10"]


def discover_frames(seq_name, K=24, Q=24):
    """Every frame_idx under this sequence's bitstream dir with a K,Q meta.json available
    (any B_c -- find_best_config_meta below picks the canonical one for each frame)."""
    seq_dir = BS_DIR / seq_name
    out = []
    for fd in sorted(seq_dir.glob("frame_*")):
        if sorted(fd.glob(f"B*_K{K}_Q{Q}_meta.json")):
            out.append(int(fd.name.split("_")[1]))
    return sorted(out)


def find_best_config_meta(seq_name, frame_idx, K=24, Q=24):
    d = BS_DIR / seq_name / f"frame_{frame_idx:04d}"
    matches = sorted(d.glob(f"B*_K{K}_Q{Q}_meta.json"))
    return matches[0] if matches else None


def load_codebook_and_adapters(seq_id, K, Q):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c = con.cursor()
    rows = c.execute(
        "SELECT cx,cy,cz FROM cluster_centroids WHERE seq_id=? AND K=? ORDER BY mode_idx", (seq_id, K)
    ).fetchall()
    centroids = np.array([[r[0], r[1], r[2]] for r in rows])
    slp = np.concatenate([[0.0], np.full(K - 1, 0.25)])
    slw = np.concatenate([[1.0], np.full(K - 1, 0.35)])
    for mode_idx, p, w in c.execute(
        "SELECT mode_idx,s_lp,s_lw FROM mode_adapters WHERE seq_id=? AND K=? AND Q=? ORDER BY mode_idx", (seq_id, K, Q)
    ).fetchall():
        slp[mode_idx] = p
        slw[mode_idx] = w
    con.close()
    return centroids, slp, slw


def _gft_two_modes(Vb, Ab, meta, v1, v2, g_raw, sk, centroid_sk, p_k, w_k):
    """Forward GFT for mode 0 and mode sk ONLY (not the full 1..K-1 search)."""
    gft = GFTStrategyWraper()
    app = Approximator()
    b_tmp = Block(meta)
    b_tmp.Vblock = Vb
    b_tmp.Ablock = Ab

    sg0 = StructuralGraph(meta)
    sg0.set_data(Vb)
    try:
        _, c0 = gft(b_tmp, sg0)
    except Exception:
        c0 = None

    if sk == 0:
        return c0, c0

    s3d = project_slope(centroid_sk, v1, v2, g_raw)
    sgp = StructuralGraph(meta)
    sgp.set_data(Vb)
    ag = AttributeGraph(sgp, s3d, sk, p_k, w_k)
    Vr = app._spatial_norm(Vb)
    Aa = Ab.copy()
    Aa[:, 0] = Vr @ s3d.T
    ag.set_data(Vb, Aa)
    try:
        _, ck = gft(b_tmp, ag)
    except Exception:
        ck = None
    return c0, ck


def _compaction(y, n=4):
    """Fraction of coefficient energy in the first min(n,N) coefficients -- same definition as
    scripts/oracle_real_rlgr_and_compaction.py's `_quant_metrics`."""
    y = np.asarray(y).ravel()
    energy_total = float(np.sum(y ** 2))
    if energy_total <= 1e-12:
        return 1.0, 0.0
    nn = min(n, len(y))
    return float(np.sum(y[:nn] ** 2) / energy_total), energy_total


def _process_child(ci, Vb, Ab, meta, v1, v2, g_raw, sk, centroids, slp, slw, Q):
    c0, ck = _gft_two_modes(Vb, Ab, meta, v1, v2, g_raw, sk, centroids[sk - 1] if sk > 0 else None, slp[sk], slw[sk])
    if c0 is None or ck is None:
        return None
    bas_q = np.round(c0[:, 0:1] / Q)
    sel_q = np.round(ck[:, 0:1] / Q)
    c_k = centroids[sk - 1] if sk > 0 else None
    p_k = slp[sk] if sk > 0 else 0.0
    w_k = slw[sk] if sk > 0 else 1.0
    t_c = {"v1": v1, "v2": v2, "g_raw": g_raw}
    _, codec_se_y, _, _, n_pts = _worker_decode_block(ci, meta, Vb, Ab, sel_q, Q, sk, c_k, p_k, w_k, t_c)
    _, base_se_y, n_pts2 = _worker_decode_baseline(ci, meta, Vb, Ab, bas_q, Q)
    codec_nnz = int(np.count_nonzero(sel_q))
    base_nnz = int(np.count_nonzero(bas_q))
    # Real spectral compaction, both modes, on the Y-channel forward-GFT coefficients (column 0).
    codec_compaction, codec_energy = _compaction(ck[:, 0])
    base_compaction, base_energy = _compaction(c0[:, 0])
    return (ci, codec_se_y, base_se_y, n_pts, codec_nnz, base_nnz,
            codec_compaction, codec_energy, base_compaction, base_energy)


def process_frame(seq_name, zip_fn, depth, frame_idx, K=24, Q=24, ply_names_cache=None):
    meta_path = find_best_config_meta(seq_name, frame_idx, K=K, Q=Q)
    if meta_path is None:
        print(f"  [SKIP] {seq_name} frame_idx={frame_idx}: no meta.json for K={K} Q={Q}", file=sys.stderr)
        return None
    meta_j = json.load(open(meta_path))
    B_c = meta_j["B_c"]
    B_p = get_bp(depth)
    mode_sequence = meta_j["mode_sequence"]
    active_mask = meta_j["active_mask"]
    assert meta_j["frame_idx"] == frame_idx

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    seq_id = con.execute("SELECT seq_id FROM sequences WHERE name=?", (seq_name,)).fetchone()[0]
    con.close()
    centroids, slp, slw = load_codebook_and_adapters(seq_id, K, Q)

    zip_path = FRAMES_DIR / zip_fn
    if ply_names_cache is not None and zip_fn in ply_names_cache:
        names = ply_names_cache[zip_fn]
    else:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".ply"))
        if ply_names_cache is not None:
            ply_names_cache[zip_fn] = names
    V_raw, A_raw = load_ply_from_zip(zip_path, names[frame_idx])
    pc = make_pc(V_raw, A_raw, seq_name, depth, frame_idx)
    pc.transform_attributes(Colourist()._RGBtoYUV)

    _, blocks_p = MortonBlockPartition().partition(pc, bsize=B_p)
    N_p = len(blocks_p)
    if N_p != len(mode_sequence):
        print(f"  [SKIP] {seq_name} frame_idx={frame_idx}: N_p={N_p} vs meta {len(mode_sequence)} "
              f"(partition/meta mismatch)", file=sys.stderr)
        return None

    parent_info = []
    for i, b in enumerate(blocks_p):
        t = extract_block_tangent(b, pc.V, pc.A)
        b.init_data(pc.V, pc.A)
        center = b.Vblock.mean(axis=0).tolist() if len(b.Vblock) > 0 else [0, 0, 0]
        b.clear_data()
        parent_info.append({
            "center": center, "g_raw": float(t["g_raw"]),
            "theta": float(np.degrees(np.arccos(np.clip(abs(t["g_norm"][1]), 0.0, 1.0)))),
            "mode": mode_sequence[i], "active": bool(active_mask[i]),
        })

    _, blocks_c = MortonBlockPartition().partition(pc, bsize=B_c)
    N_c = len(blocks_c)
    p2c = build_parent_child_map(blocks_p, blocks_c, pc.V, B_p)
    c2p = {ci: pi for pi, cl in p2c.items() for ci in cl}
    child_tang = [extract_block_tangent(b, pc.V, pc.A) for b in blocks_c]

    tasks = []
    for ci in range(N_c):
        b = blocks_c[ci]
        b.init_data(pc.V, pc.A)
        Vb, Ab = b.Vblock.copy(), b.Ablock.copy()
        b.clear_data()
        pi = c2p.get(ci, 0)
        sk = mode_sequence[pi]
        t = child_tang[ci]
        tasks.append((ci, Vb, Ab, blocks_c[ci].metadata, t["v1"], t["v2"], t["g_raw"], sk, centroids, slp, slw, Q))

    results = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=4)(
        delayed(_process_child)(*t) for t in tasks
    )

    # Aggregate child results up to parent: SE_Y/n_pts (-> PSNR, as in the production benchmark),
    # and compaction energy sums (-> energy-weighted parent-level compaction, both modes).
    parent_agg = {i: {"codec_se_y": 0.0, "base_se_y": 0.0, "n_pts": 0, "codec_nnz": 0, "base_nnz": 0,
                       "codec_energy_first4": 0.0, "codec_energy_total": 0.0,
                       "base_energy_first4": 0.0, "base_energy_total": 0.0} for i in range(N_p)}
    for r in results:
        if r is None:
            continue
        (ci, codec_se_y, base_se_y, n_pts, codec_nnz, base_nnz,
         codec_compaction, codec_energy, base_compaction, base_energy) = r
        pi = c2p.get(ci, 0)
        agg = parent_agg[pi]
        agg["codec_se_y"] += codec_se_y
        agg["base_se_y"] += base_se_y
        agg["n_pts"] += n_pts
        agg["codec_nnz"] += codec_nnz
        agg["base_nnz"] += base_nnz
        agg["codec_energy_first4"] += codec_compaction * codec_energy
        agg["codec_energy_total"] += codec_energy
        agg["base_energy_first4"] += base_compaction * base_energy
        agg["base_energy_total"] += base_energy

    rows = []
    total_codec_se = total_base_se = total_pts = 0
    for i in range(N_p):
        agg = parent_agg[i]
        info = parent_info[i]
        codec_compaction_p = (agg["codec_energy_first4"] / agg["codec_energy_total"]
                               if agg["codec_energy_total"] > 1e-12 else float("nan"))
        base_compaction_p = (agg["base_energy_first4"] / agg["base_energy_total"]
                              if agg["base_energy_total"] > 1e-12 else float("nan"))
        codec_psnr_i = (10 * math.log10(255.0 ** 2 / max(agg["codec_se_y"] / max(1, agg["n_pts"]), 1e-12))
                        if agg["n_pts"] > 0 else float("nan"))
        base_psnr_i = (10 * math.log10(255.0 ** 2 / max(agg["base_se_y"] / max(1, agg["n_pts"]), 1e-12))
                       if agg["n_pts"] > 0 else float("nan"))
        rows.append({
            "seq": seq_name, "frame_idx": frame_idx, "parent_idx": i,
            "g_raw": info["g_raw"], "theta": info["theta"], "mode": info["mode"], "active": info["active"],
            "point_count": agg["n_pts"],
            "codec_psnr_y": codec_psnr_i, "base_psnr_y": base_psnr_i,
            "psnr_gain_db": (codec_psnr_i - base_psnr_i) if agg["n_pts"] > 0 else float("nan"),
            "codec_compaction": codec_compaction_p, "base_compaction": base_compaction_p,
            "delta_compaction": (codec_compaction_p - base_compaction_p
                                  if not math.isnan(codec_compaction_p) and not math.isnan(base_compaction_p)
                                  else float("nan")),
        })
        total_codec_se += agg["codec_se_y"]; total_base_se += agg["base_se_y"]; total_pts += agg["n_pts"]

    codec_psnr = 10 * math.log10(255.0**2 / max(total_codec_se / max(1, total_pts), 1e-12))
    base_psnr = 10 * math.log10(255.0**2 / max(total_base_se / max(1, total_pts), 1e-12))
    print(f"  [OK] {seq_name} frame_idx={frame_idx}: N_p={N_p} N_c={N_c} B_c={B_c} "
          f"codec_psnr={codec_psnr:.2f} base_psnr={base_psnr:.2f}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else TARGET_SEQUENCES
    depth_by_name = {n: d for n, _, d in SEQUENCES}
    zip_by_name = {n: z for n, z, _ in SEQUENCES}

    all_rows = []
    manifest = {}
    ply_names_cache = {}
    for seq_name in only:
        if seq_name not in depth_by_name:
            print(f"  [SKIP] {seq_name}: not in SEQUENCES", file=sys.stderr)
            continue
        frames = discover_frames(seq_name)
        if not frames:
            print(f"  [SKIP] {seq_name}: no K=24,Q=24 meta.json found in any frame dir", file=sys.stderr)
            continue
        print(f"== {seq_name}: {len(frames)} frame(s) -> {frames}", file=sys.stderr)
        ok_frames = []
        for frame_idx in frames:
            try:
                rows = process_frame(seq_name, zip_by_name[seq_name], depth_by_name[seq_name],
                                      frame_idx, ply_names_cache=ply_names_cache)
            except Exception as e:
                print(f"  [FAIL] {seq_name} frame_idx={frame_idx}: {e}", file=sys.stderr)
                rows = None
            if rows:
                all_rows.extend(rows)
                ok_frames.append(frame_idx)
        manifest[seq_name] = {"frames_found": frames, "frames_ok": ok_frames,
                               "n_parent_blocks_total": sum(1 for r in all_rows if r["seq"] == seq_name)}

    with open(OUT_RECORDS, "w") as f:
        json.dump(all_rows, f)
    with open(OUT_SUMMARY, "w") as f:
        json.dump({"n_records": len(all_rows), "sequences": manifest}, f, indent=1)
    print(f"\nDone. {len(all_rows)} parent-block records across {len(manifest)} sequences.", file=sys.stderr)
    print(f"  [+] {OUT_RECORDS}", file=sys.stderr)
    print(f"  [+] {OUT_SUMMARY}", file=sys.stderr)
