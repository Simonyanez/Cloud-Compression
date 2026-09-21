#!/usr/bin/env python3
"""
Qualitative visual comparison: decode 6 example sequences (2 per family: longdress/loot 8iVFB,
phil9/ricardo9 MVUB depth-9, phil10/ricardo10 MVUB depth-10 -- the same subset used by
scripts/tier2_block_diagnostics.py) at each sequence's own best B_c, K=24, Q=32, frame_idx=1
(Q=32: at Q=24 codec vs. baseline are visually indistinguishable, while at Q=64 the PCADC/baseline
gain itself has narrowed toward zero on the RD curve -- Q=32 keeps visible compression artifacts
at an operating point where PCADC's advantage over baseline is close to its RD-curve maximum), and
render an 8-panel PNG per sequence: full body + zoomed head crop, each with (a/e) original point
cloud in true RGB, (b/f) PCADC-decoded reconstruction in true RGB, (c/g) baseline (mode-0, no
self-loop) per-point |Delta Y| error heatmap vs. ground truth, (d/h) PCADC per-point |Delta Y|
error heatmap vs. ground truth -- (c/g) and (d/h) share one color scale so they're directly
comparable.

Reuses the same "real decode" primitives as the rest of this thesis (no reimplementation):
  - decode_block_attributes (scripts/exp_e9_multi_entropy_suite.py): dequantize + inverse GFT.
  - find_best_config_meta / load_codebook_and_adapters / _gft_two_modes
    (scripts/tier2_block_diagnostics.py): forward GFT for mode 0 (baseline) and the block's
    assigned mode, both derived from the codebook + adapters actually trained for this sequence.
q_c (the quantized integer coefficients) is obtained by re-running the FORWARD GFT + quantize on
the reloaded ground-truth attributes rather than parsing the saved rlgr.bin bitstream -- there is
no working bitstream-read path anywhere in this repo (no arithmetic-coder .decode(), nothing
calls rlgr.rlgrRead), and this is exactly the methodology scripts/tier2_block_diagnostics.py
already uses and that this thesis already calls "real decode, not proxy": forward-quantize-then-
decode is mathematically identical to a true bitstream round trip, since RLGR/arithmetic coding
are lossless entropy coders around the same integers.

Only the Y (luma) channel is ever coded; U,V are lossless passthrough from the source PLY.

Usage: `python3 scripts/generate_qualitative_comparison.py [seq ...]` (default: all 6).
Output: docs/thesis/img/resultados/e10_qualitative_<seq>.png
"""
import os, sys, json, math, zipfile
os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcadc.pointcloud import MortonBlockPartition
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.color import Colourist, Approximator

from exp_e9_multi_entropy_suite import extract_block_tangent, build_parent_child_map, project_slope, decode_block_attributes
from exp_e10_multisequence_benchmark import load_ply_from_zip, make_pc, get_bp, SEQUENCES
from tier2_block_diagnostics import find_best_config_meta, load_codebook_and_adapters, _gft_two_modes

DB = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
FRAMES_DIR = REPO_ROOT / "res" / "frames"
OUT_DIR = REPO_ROOT / "docs" / "thesis" / "img" / "resultados"

TARGET_SEQUENCES = ["longdress", "loot", "phil9", "ricardo9", "phil10", "ricardo10"]
# Q=32: at Q=24 codec vs. baseline are visually indistinguishable (PSNR gap ~0.1-0.2dB on top of
# 28-44dB absolute); at Q=64 the self-loop/codebook gain itself shrinks toward the baseline (the
# RD curves elsewhere in this chapter show the PCADC/baseline gap narrowing at coarse Q), so a
# side-by-side there would show two similarly-bad reconstructions rather than an actual PCADC
# advantage. Q=32 keeps real, visible compression artifacts while sitting where the gain over
# baseline is closest to its max on the RD curve -- an operating point where PCADC is actually
# doing something, not just "some quantization, whoever wins".
K, Q, FRAME_IDX = 24, 32, 1


# -----------------------------------------------------------------------------
# Local decode-to-ARRAY forks of exp_e10_multisequence_benchmark's
# _worker_decode_block / _worker_decode_baseline -- same bodies, but returning the actual
# reconstructed Y array instead of a squared-error scalar (the production functions discard
# it). Not editing the production file: these are read-only diagnostics, not a benchmark path.
# -----------------------------------------------------------------------------
def _decode_block_rec(meta, Vb, Ab, q_c, Q, sk, c_k, p_k, w_k, t_c):
    if q_c is None:
        return np.zeros((len(Vb), 1))
    app_dec = Approximator()
    if sk == 0:
        sg = StructuralGraph(meta)
        sg.set_data(Vb)
        return decode_block_attributes(meta, Vb, q_c, Q, sg)
    s3d = project_slope(c_k, t_c["v1"], t_c["v2"], t_c["g_raw"])
    sgp = StructuralGraph(meta)
    sgp.set_data(Vb)
    ag = AttributeGraph(sgp, s3d, sk, p_k, w_k)
    Vr = app_dec._spatial_norm(Vb)
    Aa = Ab.copy()
    Aa[:, 0] = Vr @ s3d.T
    ag.set_data(Vb, Aa)
    return decode_block_attributes(meta, Vb, q_c, Q, ag)


def _decode_block_baseline_rec(meta, Vb, q_c, Q):
    if q_c is None:
        return np.zeros((len(Vb), 1))
    sg = StructuralGraph(meta)
    sg.set_data(Vb)
    return decode_block_attributes(meta, Vb, q_c, Q, sg)


def process_sequence(seq_name, zip_fn, depth):
    meta_path = find_best_config_meta(seq_name, FRAME_IDX, K=K, Q=Q)
    if meta_path is None:
        raise RuntimeError(f"{seq_name}: no meta.json for K={K} Q={Q} frame_idx={FRAME_IDX}")
    meta_j = json.load(open(meta_path))
    B_c = meta_j["B_c"]
    B_p = get_bp(depth)
    mode_sequence = meta_j["mode_sequence"]
    assert meta_j["frame_idx"] == FRAME_IDX

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    seq_id = con.execute("SELECT seq_id FROM sequences WHERE name=?", (seq_name,)).fetchone()[0]
    con.close()
    centroids, slp, slw = load_codebook_and_adapters(seq_id, K, Q)

    zip_path = FRAMES_DIR / zip_fn
    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".ply"))
    V_raw, A_rgb_raw = load_ply_from_zip(zip_path, names[FRAME_IDX])  # true RGB, [0,255]

    pc = make_pc(V_raw, A_rgb_raw, seq_name, depth, FRAME_IDX)
    pc.transform_attributes(Colourist()._RGBtoYUV)  # pc.A is now [Y,U,V] ground truth, [0,255]

    _, blocks_p = MortonBlockPartition().partition(pc, bsize=B_p)
    N_p = len(blocks_p)
    if N_p != len(mode_sequence):
        raise RuntimeError(f"{seq_name}: N_p={N_p} vs meta {len(mode_sequence)} (partition/meta mismatch)")

    _, blocks_c = MortonBlockPartition().partition(pc, bsize=B_c)
    N_c = len(blocks_c)
    p2c = build_parent_child_map(blocks_p, blocks_c, pc.V, B_p)
    c2p = {ci: pi for pi, cl in p2c.items() for ci in cl}
    child_tang = [extract_block_tangent(b, pc.V, pc.A) for b in blocks_c]

    N = len(pc.V)
    Y_hat_base = np.zeros((N, 1))
    Y_hat_codec = np.zeros((N, 1))

    for ci in range(N_c):
        b = blocks_c[ci]
        b.init_data(pc.V, pc.A)
        Vb, Ab = b.Vblock.copy(), b.Ablock.copy()
        b.clear_data()
        idxs = b.as_index()  # absolute row indices into pc.V/pc.A -- Morton partition is
                              # contiguous & un-permuted, no index inversion needed

        pi = c2p.get(ci, 0)
        sk = mode_sequence[pi]
        t = child_tang[ci]

        centroid_sk = centroids[sk - 1] if sk > 0 else None
        p_k = slp[sk] if sk > 0 else 0.0
        w_k = slw[sk] if sk > 0 else 1.0
        t_c = {"v1": t["v1"], "v2": t["v2"], "g_raw": t["g_raw"]}

        c0, ck = _gft_two_modes(Vb, Ab, b.metadata, t["v1"], t["v2"], t["g_raw"],
                                 sk, centroid_sk, p_k, w_k)
        if c0 is None or ck is None:
            continue
        bas_q = np.round(c0[:, 0:1] / Q)
        sel_q = np.round(ck[:, 0:1] / Q)

        rec_base = _decode_block_baseline_rec(b.metadata, Vb, bas_q, Q)
        rec_codec = _decode_block_rec(b.metadata, Vb, Ab, sel_q, Q, sk, centroid_sk, p_k, w_k, t_c)

        Y_hat_base[idxs] = rec_base
        Y_hat_codec[idxs] = rec_codec

    Y_gt = pc.A[:, 0:1]

    A_yuv_decoded = pc.A.copy()
    A_yuv_decoded[:, 0:1] = Y_hat_codec
    A_rgb_decoded = Colourist()._YUVtoRGB(A_yuv_decoded)

    err_base = np.abs(Y_gt - Y_hat_base).ravel()
    err_codec = np.abs(Y_gt - Y_hat_codec).ravel()

    return dict(
        seq=seq_name, depth=depth, B_c=B_c, B_p=B_p, N=N,
        V=pc.V, A_rgb_original=A_rgb_raw, A_rgb_decoded=A_rgb_decoded,
        err_base=err_base, err_codec=err_codec,
    )


# Per-sequence camera config, found by an explicit azim/elev (and, for MVUB, screen-axis-order)
# sweep against the real decoded point clouds -- the two datasets do NOT share a coordinate
# convention: 8iVFB has V[:,1] as the vertical (up) axis, while MVUB has V[:,2] as vertical and
# V[:,1] as depth (front-back) -- using the 8iVFB convention on MVUB clouds put the camera
# looking along the body instead of at it. `order` is the (screen_x, screen_y, screen_z) column
# index into V, with screen_z always the "up" axis passed to ax.view_init's elev; `up_col` is
# that same up-axis column, reused for the zoom crop below. 5 of 6 sequences resolve to a clear
# frontal face at these settings; "longdress" does not (her head is bowed down looking at the
# floor in the source capture -- no camera angle reveals her face) and keeps the 8iVFB default
# for consistency with "loot" rather than chasing an unreachable frontal face for her alone.
# zoom_frac (top fraction of the up-axis range kept for the zoom row) is tuned separately per
# family: 0.22 lands on the face for the 8iVFB standing poses (confirmed on "loot"; "longdress"'s
# face is unreachable at any crop -- see above), but the MVUB busts need a deeper crop (0.34) to
# reach past the forehead/scalp down to the eyes/nose/mouth, since their "up" range is dominated
# by hair sticking up above the actual face.
VIEW_CONFIG = {
    "longdress": dict(order=(0, 2, 1), up_col=1, elev=15, azim=60, zoom_frac=0.22),
    "loot":      dict(order=(0, 2, 1), up_col=1, elev=15, azim=60, zoom_frac=0.22),
    "phil9":     dict(order=(0, 1, 2), up_col=2, elev=15, azim=-90, zoom_frac=0.34),
    "ricardo9":  dict(order=(0, 1, 2), up_col=2, elev=15, azim=-90, zoom_frac=0.34),
    "phil10":    dict(order=(0, 1, 2), up_col=2, elev=15, azim=-90, zoom_frac=0.34),
    "ricardo10": dict(order=(0, 1, 2), up_col=2, elev=15, azim=-90, zoom_frac=0.34),
}


def render_figure(res, out_png, sample_n=45000, zoom_n=70000, seed=42):
    cfg = VIEW_CONFIG[res["seq"]]
    ox, oy, oz = cfg["order"]
    zoom_frac = cfg["zoom_frac"]

    rng = np.random.default_rng(seed)
    N = res["N"]
    idx = rng.choice(N, size=min(sample_n, N), replace=False)

    V = res["V"][idx]
    rgb_orig = res["A_rgb_original"][idx] / 255.0
    rgb_dec = res["A_rgb_decoded"][idx] / 255.0
    err_base = res["err_base"][idx]
    err_codec = res["err_codec"][idx]

    # Zoom crop: top `zoom_frac` of this sequence's vertical (up_col) range -- the head/face
    # region for these standing/bust human scans -- at higher point density (less/no
    # subsampling) so compression artifacts, only visible up close, actually show up (the
    # full-body view above is too sparse after subsampling to reveal them).
    up_all = res["V"][:, cfg["up_col"]]
    up_thresh = up_all.max() - zoom_frac * (up_all.max() - up_all.min())
    crop_idx = np.where(up_all >= up_thresh)[0]
    if len(crop_idx) > zoom_n:
        zidx = rng.choice(crop_idx, size=zoom_n, replace=False)
    else:
        zidx = crop_idx
    Vz = res["V"][zidx]
    rgb_orig_z = res["A_rgb_original"][zidx] / 255.0
    rgb_dec_z = res["A_rgb_decoded"][zidx] / 255.0
    err_base_z = res["err_base"][zidx]
    err_codec_z = res["err_codec"][zidx]

    vmax = float(np.percentile(np.concatenate([res["err_base"], res["err_codec"]]), 99))
    vmax = max(vmax, 1.0)

    psnr_codec = 10 * math.log10(255.0 ** 2 / max(np.mean(res["err_codec"] ** 2), 1e-12))
    psnr_base = 10 * math.log10(255.0 ** 2 / max(np.mean(res["err_base"] ** 2), 1e-12))

    fig = plt.figure(figsize=(20, 9.4), dpi=300)

    panels = [
        (V, rgb_orig, None, "(a) Original (RGB verdadero)", False),
        (V, rgb_dec, None, f"(b) PCADC decodificado (PSNR$_Y$={psnr_codec:.2f} dB)", False),
        (V, err_base, "err", f"(c) Error baseline $|\\Delta Y|$ (PSNR={psnr_base:.2f} dB)", True),
        (V, err_codec, "err", f"(d) Error PCADC $|\\Delta Y|$ (PSNR={psnr_codec:.2f} dB)", True),
        (Vz, rgb_orig_z, None, "(e) Original, zoom cabeza (RGB verdadero)", False),
        (Vz, rgb_dec_z, None, "(f) PCADC decodificado, zoom cabeza", False),
        (Vz, err_base_z, "err", "(g) Error baseline $|\\Delta Y|$, zoom cabeza", True),
        (Vz, err_codec_z, "err", "(h) Error PCADC $|\\Delta Y|$, zoom cabeza", True),
    ]
    axis_names = ["$X$", "$Y$", "$Z$"]
    axes, err_axes, last_err_scatter = [], [], None
    for i, (Vp, color, kind, title, is_err) in enumerate(panels):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        s = 1.2 if i < 4 else 3.0
        if is_err:
            sc = ax.scatter(Vp[:, ox], Vp[:, oy], Vp[:, oz], c=color, cmap="inferno",
                             vmin=0, vmax=vmax, s=s, alpha=0.8)
            err_axes.append(ax)
            last_err_scatter = sc
        else:
            ax.scatter(Vp[:, ox], Vp[:, oy], Vp[:, oz], c=color, s=s, alpha=0.8)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=-6)
        ax.set_xlabel(axis_names[ox]); ax.set_ylabel(axis_names[oy]); ax.set_zlabel(axis_names[oz])
        ax.view_init(elev=cfg["elev"], azim=cfg["azim"])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        axes.append(ax)

    fig.subplots_adjust(left=0.01, right=0.92, top=0.94, bottom=0.02, wspace=0.05, hspace=0.15)

    cb = fig.colorbar(last_err_scatter, ax=err_axes, shrink=0.5, pad=0.02, location="right")
    cb.set_label(f"$|\\Delta Y|$ (escala compartida, percentil 99 = {vmax:.1f})", fontsize=9)

    n_str = f"{res['N']:,}".replace(",", "\\,")
    fig.suptitle(f"{res['seq']}  ($B_c={res['B_c']}$, $K={K}$, $Q={Q}$, "
                 f"frame_idx={FRAME_IDX}, $N={n_str}$)", fontsize=13, fontweight="bold",
                 family="monospace", y=0.99)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return psnr_codec, psnr_base


CACHE_DIR = REPO_ROOT / "scripts" / "_qualitative_cache"


def load_or_process(seq_name, zip_fn, depth, use_cache=True):
    """Decode is the expensive step; cache its output per sequence so the render step (camera
    angle / zoom crop / styling) can be iterated on quickly without re-decoding."""
    cache_path = CACHE_DIR / f"{seq_name}_Q{Q}.npz"
    if use_cache and cache_path.exists():
        d = np.load(cache_path)
        return dict(seq=seq_name, depth=int(d["depth"]), B_c=int(d["B_c"]), B_p=int(d["B_p"]),
                    N=int(d["N"]), V=d["V"], A_rgb_original=d["A_rgb_original"],
                    A_rgb_decoded=d["A_rgb_decoded"], err_base=d["err_base"], err_codec=d["err_codec"])
    res = process_sequence(seq_name, zip_fn, depth)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **res)
    return res


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    no_cache = "--no-cache" in args
    args = [a for a in args if a != "--no-cache"]
    only = args if args else TARGET_SEQUENCES
    depth_by_name = {n: d for n, _, d in SEQUENCES}
    zip_by_name = {n: z for n, z, _ in SEQUENCES}

    for seq_name in only:
        if seq_name not in depth_by_name:
            print(f"  [SKIP] {seq_name}: not in SEQUENCES", file=sys.stderr)
            continue
        print(f"== {seq_name} ==", file=sys.stderr)
        res = load_or_process(seq_name, zip_by_name[seq_name], depth_by_name[seq_name], use_cache=not no_cache)
        out_png = OUT_DIR / f"e10_qualitative_{seq_name}.png"
        psnr_codec, psnr_base = render_figure(res, out_png)
        print(f"  N={res['N']:,} B_c={res['B_c']} codec_psnr={psnr_codec:.2f} base_psnr={psnr_base:.2f}",
              file=sys.stderr)
        print(f"  [+] {out_png}", file=sys.stderr)
    print("\nDone.", file=sys.stderr)
