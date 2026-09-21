#!/usr/bin/env python3
"""
Generates the figures (PNG, 300dpi) and prints the table values needed to populate the E10
results section of the thesis (docs/thesis/resultados.tex).

Follows the same pattern as scripts/generate_annex_figures.py / docs/report/generate_report_figures.py:
read-only sqlite connection, matplotlib Agg backend, no analysis text produced — this script only
produces pictures/numbers, the thesis prose stays hand-written.

IMPORTANT (fairness): all figures/tables for the "Frame Único" E10 characterization (the original
scaffolded subsections) are computed from frame_idx=1 ONLY. The live follow-up run has pooled
extra frames into aggregate_results for some (child_b,K) cells but not others, per sequence — using
that table directly would silently mix single-frame and multi-frame cells. The one exception is the
explicitly-new "temporal stability (preliminary)" figure, which is scoped to the multi-frame data on
purpose and says so.
"""
import os, sys, json, math, sqlite3
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
DB = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
OUT_DIR = REPO_ROOT / "docs" / "thesis" / "img" / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = ["longdress", "loot", "redandblack", "soldier",
             "andrew10", "david10", "phil10", "ricardo10", "sarah10",
             "andrew9", "david9", "phil9", "ricardo9", "sarah9"]
DEPTH10_8I = ["longdress", "loot", "redandblack", "soldier"]
MVUB_ORDER = ["andrew9", "andrew10", "david9", "david10", "phil9", "phil10",
              "ricardo9", "ricardo10", "sarah9", "sarah10"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300,
})

def conn():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# Bjontegaard metrics (mirrors scripts/exp_e10_multisequence_benchmark.py bdrate())
# ---------------------------------------------------------------------------
def bdrate(bpv_a, psnr_a, bpv_t, psnr_t):
    try:
        pmin = max(min(psnr_a), min(psnr_t)); pmax = min(max(psnr_a), max(psnr_t))
        if pmin >= pmax:
            return float("nan")
        pa = np.polyfit(psnr_a, np.log(np.array(bpv_a) + 1e-12), 3)
        pt = np.polyfit(psnr_t, np.log(np.array(bpv_t) + 1e-12), 3)
        xs = np.linspace(pmin, pmax, 100)
        return float((np.exp(np.mean(np.polyval(pt, xs) - np.polyval(pa, xs))) - 1.0) * 100.0)
    except Exception:
        return float("nan")

def bdpsnr(bpv_a, psnr_a, bpv_t, psnr_t):
    """Symmetric companion metric: mean PSNR delta at matched (log-)rate."""
    try:
        ra = np.log(np.array(bpv_a) + 1e-12); rt = np.log(np.array(bpv_t) + 1e-12)
        rmin = max(min(ra), min(rt)); rmax = min(max(ra), max(rt))
        if rmin >= rmax:
            return float("nan")
        pa = np.polyfit(ra, psnr_a, 3)
        pt = np.polyfit(rt, psnr_t, 3)
        xs = np.linspace(rmin, rmax, 100)
        return float(np.mean(np.polyval(pt, xs) - np.polyval(pa, xs)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Pull frame_idx=1-only per-(seq,child_b,K,Q) rows
# ---------------------------------------------------------------------------
def load_f1_rows(c):
    rows = c.execute("""
        SELECT se.name, fr.child_b, fr.K, fr.q_step, fr.frame_bpv, fr.frame_psnr_Y,
               fr.baseline_bpv, fr.baseline_psnr_Y, fr.frame_runtime_seconds
        FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
        WHERE fr.frame_idx = 1 AND NOT (se.name='longdress' AND fr.child_b=32)
        ORDER BY se.name, fr.child_b, fr.K, fr.q_step
    """).fetchall()
    data = defaultdict(list)
    for name, cb, K, Q, bpv, psnr, bbpv, bpsnr, rt in rows:
        data[(name, cb, K)].append({"Q": Q, "bpv": bpv, "psnr": psnr, "bbpv": bbpv, "bpsnr": bpsnr, "rt": rt})
    return data


def main():
    c = conn().cursor()
    f1 = load_f1_rows(c)
    # Shared family grouping (8iVFB / MVUB depth-9 / MVUB depth-10), used by several sections
    # below (items 12, 13) to disaggregate what used to be pooled across all 14 sequences.
    FAMILIES = [("8iVFB", DEPTH10_8I), ("MVUB depth-9", [n for n in SEQUENCES if n.endswith("9")]),
                ("MVUB depth-10", [n for n in SEQUENCES if n.endswith("10") and n not in DEPTH10_8I])]

    # -----------------------------------------------------------------
    # Table 1: BD-Rate / BD-PSNR / runtime per sequence, best config
    # -----------------------------------------------------------------
    print("=" * 100)
    print("TABLE: tab:e10_bd_summary (frame_idx=1 only)")
    print("=" * 100)
    best_rows = []
    for name in SEQUENCES:
        cfgs = {(cb, K): v for (n, cb, K), v in f1.items() if n == name}
        results = []
        for (cb, K), qs in cfgs.items():
            qs_sorted = sorted(qs, key=lambda r: r["Q"])
            if len(qs_sorted) < 4:
                continue
            bpv = [r["bpv"] for r in qs_sorted]; psnr = [r["psnr"] for r in qs_sorted]
            bbpv = [r["bbpv"] for r in qs_sorted]; bpsnr = [r["bpsnr"] for r in qs_sorted]
            bd_r = bdrate(bbpv, bpsnr, bpv, psnr)
            bd_p = bdpsnr(bbpv, bpsnr, bpv, psnr)
            rt = float(np.mean([r["rt"] for r in qs_sorted]))
            results.append((cb, K, bd_r, bd_p, rt))
        results = [r for r in results if not math.isnan(r[2])]
        results.sort(key=lambda r: r[2])
        best = results[0] if results else (None, None, float("nan"), float("nan"), float("nan"))
        depth = 9 if name.endswith("9") else 10
        best_rows.append((name, depth, *best))
        print(f"{name:12s} depth={depth} B_c*={best[0]} K*={best[1]} "
              f"BD-Rate={best[2]:+.3f}% BD-PSNR={best[3]:+.3f}dB runtime={best[4]:.1f}s")

    valid = [r for r in best_rows if not math.isnan(r[4])]
    avg_bdr = np.mean([r[4] for r in valid])
    avg_bdp = np.mean([r[5] for r in valid])
    avg_rt = np.mean([r[6] for r in valid])
    print(f"{'PROMEDIO':12s} BD-Rate={avg_bdr:+.3f}% BD-PSNR={avg_bdp:+.3f}dB runtime={avg_rt:.1f}s")

    with open(REPO_ROOT / "scripts" / "_e10_thesis_table_data.json", "w") as f:
        json.dump({"best_rows": best_rows, "avg": [avg_bdr, avg_bdp, avg_rt]}, f, indent=1, default=str)

    # -----------------------------------------------------------------
    # Fig: RD curves, ONE standalone plot per (sequence, B_c) -- item 1's follow-up: generate
    # all of them individually and let LaTeX place each one at full readable width instead of
    # pre-compositing into subfigure grids. All available K overlaid per plot (K in
    # {4,8,16,24,30}), restricted to the base Q in {24,32,48,64} grid -- the only Q set common
    # to every (B_c,K) cell across all 14 sequences (the Q-densified 8-point grid exists at only
    # one (B_c,K) per sequence, and which cell differs between andrew9/10 and everyone else, so
    # there is no cell all 14 share beyond these 4 points). The "(+n Q)" annotation from the old
    # 2-panel design is dropped entirely -- moot once standardized on this common grid. Each plot
    # gets a zoomed inset over the low-bitrate/high-Q end, where curve separation is documented
    # as largest (see discussion paragraph below).
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIG: e10_rd_<seq>_bc<N>.png (one standalone plot per sequence x B_c, all K overlaid)")
    print("=" * 100)
    BASE_QS = {24, 32, 48, 64}
    RD_KS = [4, 8, 16, 24, 30]

    def rd_curve_data(name, cb, K):
        """(bpv,psnr,bbpv,bpsnr) rows for (name,cb,K), filtered to the base Q grid, sorted by Q.
        None if fewer than all 4 base-grid points are present."""
        qs = f1.get((name, cb, K))
        if not qs:
            return None
        qs = sorted([r for r in qs if r["Q"] in BASE_QS], key=lambda r: r["Q"])
        return qs if len(qs) >= 4 else None

    rd_files = {}  # name -> {cb: fname}
    for name in SEQUENCES:
        cbs_avail = sorted(set(cb for (n, cb, K) in f1 if n == name))
        if name == "longdress":
            cbs_avail.append(32)  # excluded from f1/load_f1_rows -- handled directly below
        for cb in cbs_avail:
            if name == "longdress" and cb == 32:
                # Only K in {16,24} have coverage at this (sequence,B_c); query directly.
                curves = {}
                for K in (16, 24):
                    r32 = c.execute("""
                        SELECT fr.q_step, fr.frame_bpv, fr.frame_psnr_Y, fr.baseline_bpv, fr.baseline_psnr_Y
                        FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
                        WHERE se.name='longdress' AND fr.child_b=32 AND fr.K=? AND fr.frame_idx=1
                        ORDER BY fr.q_step
                    """, (K,)).fetchall()
                    qs = sorted([{"Q": r[0], "bpv": r[1], "psnr": r[2], "bbpv": r[3], "bpsnr": r[4]}
                                 for r in r32 if r[0] in BASE_QS], key=lambda r: r["Q"])
                    if len(qs) >= 4:
                        curves[K] = qs
            else:
                curves = {K: qs for K in RD_KS if (qs := rd_curve_data(name, cb, K)) is not None}
            if not curves:
                continue

            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            Ks_here = sorted(curves)
            colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(Ks_here), 1)))
            for K, color in zip(Ks_here, colors):
                qs = curves[K]
                ax.plot([r["bpv"] for r in qs], [r["psnr"] for r in qs], "o-",
                        color=color, label=f"$K={K}$", linewidth=1.3, markersize=3.5)
            qs0 = curves[Ks_here[0]]
            ax.plot([r["bbpv"] for r in qs0], [r["bpsnr"] for r in qs0], "k--",
                    linewidth=1.1, label="baseline", alpha=0.75)
            ax.set_xlabel("bpv"); ax.set_ylabel("PSNR$_Y$ (dB)")
            ax.set_title(f"{name}, $B_c={cb}$", fontsize=10)
            # upper-left: RD curves run bottom-left (low bpv/low PSNR) to top-right, so this
            # corner stays clear of both the data and the low-bitrate zoomed inset (bottom-right).
            ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)

            # Zoomed inset tightly around the Q=32 point specifically (not just "the low-bitrate
            # end") -- separation between the PCADC curves and the baseline at any single fixed
            # Q is often subtle at full-plot scale, so this crops in close on that one operating
            # point (still plotting the actual lines through it, not isolated markers) to make
            # the gap visible.
            zoom_bpv = [r["bpv"] for qs in curves.values() for r in qs if r["Q"] == 32] \
                + [r["bbpv"] for r in qs0 if r["Q"] == 32]
            zoom_psnr = [r["psnr"] for qs in curves.values() for r in qs if r["Q"] == 32] \
                + [r["bpsnr"] for r in qs0 if r["Q"] == 32]
            if len(zoom_bpv) >= 2 and max(zoom_bpv) - min(zoom_bpv) > 1e-9:
                axins = ax.inset_axes([0.52, 0.08, 0.44, 0.38])
                for K, color in zip(Ks_here, colors):
                    qs = curves[K]
                    axins.plot([r["bpv"] for r in qs], [r["psnr"] for r in qs], "o-",
                               color=color, linewidth=1.3, markersize=4)
                axins.plot([r["bbpv"] for r in qs0], [r["bpsnr"] for r in qs0], "k--",
                          linewidth=1.1, alpha=0.75)
                # tight: 12% padding around just the Q=32 cluster of points (vs. 15% over a
                # wider 2-point span before) so the crop centers closely on that operating point.
                pad_x = (max(zoom_bpv) - min(zoom_bpv)) * 0.12 + 1e-6
                pad_y = (max(zoom_psnr) - min(zoom_psnr)) * 0.12 + 1e-6
                axins.set_xlim(min(zoom_bpv) - pad_x, max(zoom_bpv) + pad_x)
                axins.set_ylim(min(zoom_psnr) - pad_y, max(zoom_psnr) + pad_y)
                axins.tick_params(labelsize=6)
                axins.grid(alpha=0.25)
                ax.indicate_inset_zoom(axins, edgecolor="gray")

            fig.tight_layout()
            fname = f"e10_rd_{name}_bc{cb}.png"
            fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            rd_files.setdefault(name, {})[cb] = fname

    n_total = sum(len(v) for v in rd_files.values())
    print(f"  [+] e10_rd_<seq>_bc<N>.png x{n_total} ({len(rd_files)} sequences)")
    with open(REPO_ROOT / "scripts" / "_e10_rd_files.json", "w") as f:
        json.dump(rd_files, f, indent=1)

    # -----------------------------------------------------------------
    # Fig 3: BD-Rate vs K, split by depth (FAIRNESS FIX — see module docstring's sibling note
    # below). A single "B_c=16" line averaged across all 14 sequences would silently mix two
    # different parent:child ratios: depth-9's B_p=16 means child_b=16 IS the parent (ratio 1,
    # no subdivision), while depth-10's B_p=32 means child_b=16 is a real 2x subdivision (ratio
    # 2). Same problem at child_b=8 (ratio 2 vs 4) and child_b=4 (ratio 4 vs 8). Splitting by
    # depth keeps each panel's child_b values at one consistent ratio internally.
    # -----------------------------------------------------------------
    def bd_mean_by_k(names, cb):
        Ks_avail = sorted(set(K for (n, ccb, K) in f1 if ccb == cb and n in names))
        out = {}
        for K in Ks_avail:
            vals = []
            for name in names:
                qs = f1.get((name, cb, K))
                if not qs or len(qs) < 4:
                    continue
                qs_sorted = sorted(qs, key=lambda r: r["Q"])
                bpv = [r["bpv"] for r in qs_sorted]; psnr = [r["psnr"] for r in qs_sorted]
                bbpv = [r["bbpv"] for r in qs_sorted]; bpsnr = [r["bpsnr"] for r in qs_sorted]
                bd = bdrate(bbpv, bpsnr, bpv, psnr)
                if not math.isnan(bd):
                    vals.append(bd)
            out[K] = np.mean(vals) if vals else np.nan
        return out

    DEPTH9 = [n for n in SEQUENCES if n.endswith("9")]
    DEPTH10 = [n for n in SEQUENCES if n.endswith("10")]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for cb, color, ratio9, ratio10 in [(4, "#4C78A8", 4, 8), (8, "#F58518", 2, 4), (16, "#54A24B", 1, 2)]:
        d9 = bd_mean_by_k(DEPTH9, cb)
        d10 = bd_mean_by_k(DEPTH10, cb)
        Ks9 = sorted(d9); Ks10 = sorted(d10)
        axL.plot(Ks9, [d9[k] for k in Ks9], "o-", color=color,
                 label=f"$B_c={cb}$ (razón {ratio9})" + (" = padre" if ratio9 == 1 else ""), linewidth=1.5)
        axR.plot(Ks10, [d10[k] for k in Ks10], "o-", color=color,
                 label=f"$B_c={cb}$ (razón {ratio10})", linewidth=1.5)
    axL.set_title("Depth-9 ($B_p=16$)"); axR.set_title("Depth-10 ($B_p=32$)")
    for ax in (axL, axR):
        ax.set_xlabel("$K$ (tamaño del libro de códigos)")
        ax.set_xticks([4, 8, 16, 24]); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axL.set_ylabel("BD-Rate promedio (\\%)".replace("\\%", "%"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_k_sweep.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_k_sweep.png (depth-split)")

    # -----------------------------------------------------------------
    # Fig: overhead breakdown, all 14 sequences (replaces longdress-only scaffold intent)
    # -----------------------------------------------------------------
    rows = c.execute("""
        SELECT se.name, fr.rlgr_bits, fr.cabac_overhead_bits, fr.deflate_overhead_bits,
               fr.lzma_overhead_bits, fr.raw_fixed_bits
        FROM frame_results fr JOIN sequences se ON se.seq_id=fr.seq_id
        WHERE fr.frame_idx=1 AND NOT (se.name='longdress' AND fr.child_b=32)
    """).fetchall()
    by_seq = defaultdict(list)
    for name, rlgr, cabac, defl, lzma, raw in rows:
        total = rlgr + cabac
        by_seq[name].append(cabac / total * 100.0 if total > 0 else 0.0)
    # item 17: add std-dev (as error bars on the mean). Regenerated from the CURRENT db -- the
    # previous "max approaching 1(00%)" reading was a stale render from before the Q-densification
    # run finished writing; live max right now is ~1.1%, confirmed by direct query.
    # item 5: ordered by mean descending (same pattern as ADJ_SEQS below for the adjacency
    # z-score figure), instead of the fixed SEQUENCES list order.
    OVH_SEQS = sorted(SEQUENCES, key=lambda n: -np.mean(by_seq[n]))
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [np.mean(by_seq[n]) for n in OVH_SEQS]
    stds = [np.std(by_seq[n]) for n in OVH_SEQS]
    maxs = [np.max(by_seq[n]) for n in OVH_SEQS]
    x = np.arange(len(OVH_SEQS))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4C78A8", label="promedio ± std")
    ax.scatter(x, maxs, color="#B23A48", marker="_", s=200, linewidths=2, label="máximo observado", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(OVH_SEQS, rotation=45, ha="right")
    ax.set_ylabel("Overhead CABAC / bits totales (\\%)".replace("\\%", "%"))
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_overhead_breakdown.png", dpi=300)
    plt.close(fig)
    print(f"  [overhead] live max ratio across all configs: {max(maxs):.3f}%")
    print("  [+] e10_overhead_breakdown.png")

    # -----------------------------------------------------------------
    # Fig: codebook direction atlas (theta strip plot, all 14, K=24)
    # Data source: cluster_centroids, pulled fresh (cheap, no new computation logic).
    # -----------------------------------------------------------------
    theta_rows = c.execute("""
        SELECT se.name, cc.mode_idx, cc.theta_spherical FROM cluster_centroids cc
        JOIN sequences se ON se.seq_id = cc.seq_id WHERE cc.K = 24 ORDER BY se.seq_id, cc.mode_idx
    """).fetchall()
    theta_by_seq = defaultdict(list)
    for name, mode_idx, theta in theta_rows:
        theta_by_seq[name].append((mode_idx, theta))
    # Frequency weight per (seq, mode): how many parent blocks actually use that direction
    # (cluster_distribution.n_blocks_assigned, frame_idx=1, K=24) -- item 18: does angular
    # concentration correspond to heavily-used modes, or just visually-close rarely-used ones?
    freq_by_seq_mode = {}
    for row in c.execute("""
        SELECT se.name, cd.mode_idx, cd.n_blocks_assigned FROM cluster_distribution cd
        JOIN sequences se ON se.seq_id = cd.seq_id WHERE cd.K = 24 AND cd.frame_idx = 1
    """).fetchall():
        freq_by_seq_mode[(row[0], row[1])] = row[2]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1]})
    for i, name in enumerate(SEQUENCES):
        pairs = theta_by_seq[name]
        vals = [t for _, t in pairs]
        freqs = np.array([freq_by_seq_mode.get((name, m), 0) for m, _ in pairs], dtype=float)
        y = np.full(len(vals), len(SEQUENCES) - 1 - i)
        # Left panel: unweighted (as before), for reference.
        ax.scatter(vals, y, s=14, color="#4C78A8", alpha=0.6)
        ax.scatter(np.mean(vals), len(SEQUENCES) - 1 - i, s=45, facecolors="none", edgecolors="black", linewidths=1.2)
        # Right panel: marker area proportional to how many parent blocks actually use that
        # direction -- a mode with a "close" angle but near-zero frequency shouldn't visually
        # compete with one that's both close AND heavily used.
        sizes = 8 + 60 * (freqs / max(freqs.max(), 1))
        ax2.scatter(vals, y, s=sizes, color="#F58518", alpha=0.55, edgecolors="none")
        if freqs.sum() > 0:
            w_mean = float(np.average(vals, weights=freqs + 1e-6))
            ax2.scatter(w_mean, len(SEQUENCES) - 1 - i, s=45, facecolors="none", edgecolors="black", linewidths=1.2, marker="D")
    for a, title in [(ax, "sin ponderar"), (ax2, "ponderado por frecuencia de uso")]:
        a.set_yticks(range(len(SEQUENCES)))
        a.set_yticklabels(SEQUENCES[::-1] if a is ax else [])
        a.set_xlabel(r"$\theta = \arccos(|c_y|)$ (grados)")
        a.set_xlim(-2, 92)
        a.set_title(title, fontsize=9)
        a.grid(axis="x", alpha=0.3)
    fig.suptitle(r"$0°$=vertical, $90°$=horizontal — círculo/rombo blanco: media (sin/con ponderar)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_codebook_directions.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_codebook_directions.png (unweighted + frequency-weighted panels)")

    print("\nTABLE: nesting K4->K24 (cosine similarity, per sequence)")
    for name in SEQUENCES:
        k4 = c.execute("""SELECT cx,cy,cz FROM cluster_centroids cc JOIN sequences se ON se.seq_id=cc.seq_id
                           WHERE se.name=? AND cc.K=4 ORDER BY cc.mode_idx""", (name,)).fetchall()
        k24 = c.execute("""SELECT cx,cy,cz FROM cluster_centroids cc JOIN sequences se ON se.seq_id=cc.seq_id
                            WHERE se.name=? AND cc.K=24 ORDER BY cc.mode_idx""", (name,)).fetchall()
        k4v = np.array(k4); k24v = np.array(k24)
        sims = [float(np.max(np.abs(k24v @ v))) for v in k4v]
        print(f"  {name:12s} worst={min(sims):.4f} best={max(sims):.4f}")

    # -----------------------------------------------------------------
    # Fig: adapter (self-loop) search trace -- ONE representative example, showing HOW the
    # search proceeds (not just the summary numbers, which now live in the convergence table
    # above per item 20). The former left panel (k-means inertia for 2 sequences) is dropped:
    # centroid-shift in e10_convergence_frequency.png is the direct convergence measure the
    # thesis text already argues is better than inertia, so keeping both was redundant.
    # -----------------------------------------------------------------
    scratch = Path("/tmp/claude-1001/-home-simao-Documents-Repositories-Cloud-Compression/30593cdb-b42a-4f31-a0d4-bebd0010efc4/scratchpad")
    refresh = json.load(open(scratch / "refresh_results.json"))
    adapter_trace = refresh["adapter_trace_longdress_K24_Q24_mode1"]

    fig, ax2 = plt.subplots(figsize=(5.5, 4))
    coarse = [(i, r[4]) for i, r in enumerate(adapter_trace) if r[0] == "coarse"]
    golden = [(i, r[4]) for i, r in enumerate(adapter_trace) if r[0] == "golden_w"]
    ax2.scatter([p[0] for p in coarse], [p[1] for p in coarse], color="#B23A48", label="grilla gruesa", s=35)
    ax2.scatter([p[0] for p in golden], [p[1] for p in golden], color="#4C78A8", label="sección áurea", s=35)
    best_idx = int(np.argmin([r[4] for r in adapter_trace]))
    ax2.scatter(best_idx, adapter_trace[best_idx][4], facecolors="none", edgecolors="black", s=140, linewidths=1.5, label="óptimo")
    ax2.set_xlabel("evaluación #"); ax2.set_ylabel("costo RD"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Búsqueda de adaptador (longdress, $K=24$, modo 1)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_training_diagnostics.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_training_diagnostics.png (adapter search trace only)")

    # -----------------------------------------------------------------
    # Fig: Morton adjacency z-score (naive vs active-only), all 14 sequences
    # -----------------------------------------------------------------
    adjacency_z = {
        "longdress": (53.6, 49.96), "loot": (12.6, 0.62), "redandblack": (8.2, 0.72),
        "soldier": (2.4, 2.52), "andrew9": (24.9, 14.98), "andrew10": (15.5, 2.94),
        "david9": (19.3, 8.04), "david10": (15.3, 5.38), "phil9": (26.5, 18.90),
        "phil10": (13.1, 3.24), "ricardo9": (14.4, 4.28), "ricardo10": (12.3, -0.71),
        "sarah9": (20.1, 8.62), "sarah10": (14.7, 3.01),
    }
    # Sorted by |z| of the relevant series ("solo bloques activos", per the thesis prose --
    # the "todos los bloques" series is the mechanical gating artifact, not the interesting one).
    ADJ_SEQS = sorted(SEQUENCES, key=lambda n: -abs(adjacency_z[n][1]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ADJ_SEQS)); w = 0.38
    all_z = [adjacency_z[n][0] for n in ADJ_SEQS]
    act_z = [adjacency_z[n][1] for n in ADJ_SEQS]
    ax.bar(x - w/2, all_z, width=w, color="#B0B8C1", label="todos los bloques")
    ax.bar(x + w/2, act_z, width=w, color="#4C78A8", label="solo bloques activos")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ADJ_SEQS, rotation=45, ha="right")
    ax.set_ylabel("puntaje $z$ (adyacencia de mismo modo)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_adjacency_zscore.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_adjacency_zscore.png")

    # -----------------------------------------------------------------
    # Fig: block-level dependence plots (item 6) -- SHAP-dependence-plot style pooled scatter
    # across ALL available frames of 6 target sequences (2 per family: longdress/loot 8iVFB,
    # phil9/ricardo9 depth-9, phil10/ricardo10 depth-10), real Tier-2 decode data
    # (scripts/tier2_block_diagnostics.py: 2 real decodes per parent block, baseline vs.
    # assigned mode). Two structural drivers (g_raw, log10(point_count), the block-level
    # analogue of the sequence-level log10(N_vox) driver in e10_driver_correlations.png) x two
    # outcomes (PSNR gain vs. baseline, Delta-compaction vs. baseline) = 2 figures x 2 panels
    # each. Replaces the old single-sequence, no-compaction e10_block_gradient_gain.png.
    # -----------------------------------------------------------------
    tier2_path = REPO_ROOT / "scripts" / "_tier2_block_records.json"
    if tier2_path.exists():
        tier2_rows = json.load(open(tier2_path))
        FAMILY_OF = {"longdress": "8iVFB", "loot": "8iVFB", "phil9": "MVUB depth-9",
                     "ricardo9": "MVUB depth-9", "phil10": "MVUB depth-10", "ricardo10": "MVUB depth-10"}
        fam_colors = {"8iVFB": "#4C78A8", "MVUB depth-9": "#F58518", "MVUB depth-10": "#54A24B"}
        valid = [r for r in tier2_rows if not math.isnan(r["psnr_gain_db"])
                 and not math.isnan(r["delta_compaction"]) and r["point_count"] > 0]
        g_raw_arr = np.array([r["g_raw"] for r in valid])
        log_npts = np.log10(np.array([r["point_count"] for r in valid], dtype=float))
        psnr_gain = np.array([r["psnr_gain_db"] for r in valid])
        dcompact = np.array([r["delta_compaction"] for r in valid])
        fams = [FAMILY_OF[r["seq"]] for r in valid]

        def dependence_fig(xvals, xlabel, fname, xclip=None):
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
            x_plot = np.minimum(xvals, xclip) if xclip is not None else xvals
            for ax, y, ylabel in [(axL, psnr_gain, "ganancia PSNR por bloque (dB)"),
                                   (axR, dcompact, r"$\Delta$compactación (frac.\ energía, primeros 4 coef.)")]:
                for fam, color in fam_colors.items():
                    mask = np.array([f == fam for f in fams])
                    ax.scatter(x_plot[mask], y[mask], s=6, alpha=0.25, color=color, label=fam)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
            axL.legend(fontsize=7, markerscale=2, loc="best")
            fig.tight_layout()
            fig.savefig(OUT_DIR / fname, dpi=300)
            plt.close(fig)
            n_seq = len(set(r["seq"] for r in valid))
            print(f"  [+] {fname} (n={len(xvals)} bloques padre, {n_seq} secuencias)")

        dependence_fig(g_raw_arr, "$g_{raw}$ (magnitud del gradiente de color)",
                        "e10_block_dependence_graw.png", xclip=10)
        dependence_fig(log_npts, r"$\log_{10}(\mathtt{point\_count})$ (bloque padre)",
                        "e10_block_dependence_npts.png")
    else:
        print("  [!] scripts/_tier2_block_records.json not found -- run "
              "scripts/tier2_block_diagnostics.py first; skipping block-dependence figures.")

    # -----------------------------------------------------------------
    # Fig: temporal stability, multi-sequence, LIVE query (not a hardcoded snapshot — the
    # follow-up run keeps adding frames, so this always reflects whatever's in the DB right now).
    # -----------------------------------------------------------------
    temporal_rows = c.execute("""
        SELECT se.name, fr.frame_idx, fr.delta_pct FROM frame_results fr
        JOIN sequences se ON se.seq_id = fr.seq_id
        WHERE fr.frame_idx > 1 AND fr.child_b = 16 AND fr.K = 24 AND fr.q_step = 24
        ORDER BY se.name, fr.frame_idx
    """).fetchall()
    temporal_series = defaultdict(list)
    for name, fi, dp in temporal_rows:
        temporal_series[name].append((fi, dp))
    n_complete = sum(1 for pts in temporal_series.values() if len(pts) >= 4)
    n_covered = len(temporal_series)
    print(f"\nTemporal coverage right now: {n_covered}/14 sequences have extra frames "
          f"({n_complete} with >=4 points).")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(temporal_series), 1)))
    for (name, pts), color in zip(sorted(temporal_series.items()), colors):
        pts = sorted(pts)
        style = "-" if len(pts) >= 4 else "--"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style + "o", color=color, label=name, markersize=4)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("$\\Delta$BPV (\\%), $B_c=16, K=24, Q=24$".replace("\\%", "%"))
    ax.legend(fontsize=6.5, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_temporal_stability.png", dpi=300)
    plt.close(fig)
    print(f"  [+] e10_temporal_stability.png ({n_covered}/14 sequences, {n_complete} with full 4+ points)")

    # -----------------------------------------------------------------
    # Fig: sequence-level driver/correlation matrix (Pearson, n=14)
    # -----------------------------------------------------------------
    sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from exp_e10_multisequence_benchmark import load_ply_from_zip, make_pc, SEQUENCES as SEQ_TUPLES
    from pcadc.color import Colourist
    import zipfile

    frames_dir = REPO_ROOT / "res" / "frames"
    y_std = {}
    for name, zip_fn, depth in SEQ_TUPLES:
        zip_path = frames_dir / zip_fn
        with zipfile.ZipFile(zip_path, "r") as z:
            names_in_zip = sorted(n for n in z.namelist() if n.lower().endswith(".ply"))
        V_raw, A_raw = load_ply_from_zip(zip_path, names_in_zip[1])  # frame_idx=1, matches everything else here
        pc = make_pc(V_raw, A_raw, name, depth, 1)
        pc.transform_attributes(Colourist()._RGBtoYUV)
        y_std[name] = float(np.std(pc.A[:, 0]))
    print("\n  [computed] Y_std per sequence (frame_idx=1)")

    slp_slw = {}
    for name in SEQUENCES:
        rows = c.execute("""SELECT ma.s_lp, ma.s_lw FROM mode_adapters ma JOIN sequences se ON se.seq_id=ma.seq_id
                             WHERE se.name=? AND ma.K=24 AND ma.mode_idx>0""", (name,)).fetchall()
        slp_slw[name] = (float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows])))

    g_raw_mean = {}
    for name in SEQUENCES:
        rows = c.execute("""SELECT cd.mean_g_raw, cd.n_blocks_assigned FROM cluster_distribution cd
                             JOIN sequences se ON se.seq_id=cd.seq_id WHERE se.name=? AND cd.K=24""", (name,)).fetchall()
        num = sum(r[0] * r[1] for r in rows); den = sum(r[1] for r in rows)
        g_raw_mean[name] = num / den if den > 0 else float("nan")

    gated_pct_mean = {}
    for name in SEQUENCES:
        rows = c.execute("""SELECT fr.gated_pct FROM frame_results fr JOIN sequences se ON se.seq_id=fr.seq_id
                             WHERE se.name=? AND fr.frame_idx=1""", (name,)).fetchall()
        gated_pct_mean[name] = float(np.mean([r[0] for r in rows]))

    n_train_voxels = {name: c.execute("SELECT n_train_voxels FROM sequences WHERE name=?", (name,)).fetchone()[0]
                       for name in SEQUENCES}

    bd_by_name = {r[0]: r[2:] for r in best_rows}  # (B_c*, K*, bd_rate, bd_psnr, runtime)

    driver_vars = {
        "BD-Rate":       [bd_by_name[n][2] for n in SEQUENCES],
        "BD-PSNR":       [bd_by_name[n][3] for n in SEQUENCES],
        "Gated %":       [gated_pct_mean[n] for n in SEQUENCES],
        "Overhead %":    [np.mean(by_seq[n]) for n in SEQUENCES],
        "mean s_lp":     [slp_slw[n][0] for n in SEQUENCES],
        "mean s_lw":     [slp_slw[n][1] for n in SEQUENCES],
        "g_raw mean":    [g_raw_mean[n] for n in SEQUENCES],
        "Y_std":         [y_std[n] for n in SEQUENCES],
        "log10(N_vox)":  [math.log10(n_train_voxels[n]) for n in SEQUENCES],
        "depth==10":     [1.0 if n in DEPTH10 else 0.0 for n in SEQUENCES],
    }
    labels = list(driver_vars.keys())
    M = np.array([driver_vars[k] for k in labels])  # (10, 14)
    corr = np.corrcoef(M)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson $r$ ($n=14$)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_driver_correlations.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_driver_correlations.png")

    se_r = 1.0 / math.sqrt(14 - 3)
    print(f"\nTABLE: driver correlations (n=14, SE(r)~={se_r:.3f}, only |r|>~0.5 treated as signal)")
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if j <= i:
                continue
            if abs(corr[i, j]) >= 0.5:
                print(f"  {li:14s} vs {lj:14s}: r={corr[i,j]:+.3f}")

    with open(REPO_ROOT / "scripts" / "_e10_driver_data.json", "w") as f:
        json.dump({"labels": labels, "corr": corr.tolist(), "raw": {k: v for k, v in driver_vars.items()}}, f, indent=1)

    # ===================================================================
    # DIAGNOSTIC DEEP-DIVE (added post-hoc, no new runs, no algorithm changes):
    # adapter distribution, compute time, per-frame BD-Rate +/- std.
    # ===================================================================

    # -----------------------------------------------------------------
    # Fig: s_lp/s_lw distribution across sequences (K=24, by Q)
    # Data source: mode_adapters, pulled fresh (cheap, no new computation logic).
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIG: e10_adapter_distribution.png (s_lp x s_lw heatmap, by family, K=24, Q=24)")
    print("=" * 100)
    # item 13: heatmap instead of per-sequence scatter (drops the sequence-identity axis you
    # flagged as unhelpful here), split by family instead of pooling all 14, s_lp axis reduced
    # to its 3 real values (the coarse grid -- it is never continuous, confirmed by the prior
    # round's own finding) instead of a full [0,1] range.
    coarse_p = [0.05, 0.25, 0.60]
    slw_edges = np.linspace(0, 1.0, 11)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for fi, (fam_name, fam_seqs) in enumerate(FAMILIES):
        ax = axes[fi]
        heat = np.zeros((len(coarse_p), len(slw_edges) - 1))
        placeholders = ",".join("?" * len(fam_seqs))
        rows = c.execute(f"""
            SELECT ma.s_lp, ma.s_lw FROM mode_adapters ma JOIN sequences se ON se.seq_id=ma.seq_id
            WHERE se.name IN ({placeholders}) AND ma.K=24 AND ma.Q=24 AND ma.mode_idx>0
        """, tuple(fam_seqs)).fetchall()
        for slp, slw in rows:
            pi = int(np.argmin([abs(slp - p) for p in coarse_p]))
            wi = min(np.searchsorted(slw_edges, slw, side="right") - 1, len(slw_edges) - 2)
            heat[pi, wi] += 1
        im = ax.imshow(heat, aspect="auto", origin="lower", cmap="Blues",
                        extent=(0, 1, -0.5, len(coarse_p) - 0.5))
        ax.set_yticks(range(len(coarse_p))); ax.set_yticklabels([f"{p:.2f}" for p in coarse_p])
        ax.set_title(f"{fam_name} (n={len(rows)})", fontsize=10)
        ax.set_xlabel(r"$s_{lw}$")
        if fi == 0:
            ax.set_ylabel(r"$s_{lp}$ (3 valores de la malla gruesa)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="modos")
    fig.suptitle(r"Distribución conjunta $(s_{lp}, s_{lw})$ por familia, $K=24$, $Q=24$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_adapter_distribution.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_adapter_distribution.png (heatmap, by family)")

    print("\nTABLE: s_lp/s_lw summary stats per sequence (K=24, pooled over Q and mode)")
    adapter_summary = {}
    for name in SEQUENCES:
        rows = c.execute("""
            SELECT ma.s_lp, ma.s_lw FROM mode_adapters ma JOIN sequences se ON se.seq_id=ma.seq_id
            WHERE se.name=? AND ma.K=24
        """, (name,)).fetchall()
        slp = np.array([r[0] for r in rows])
        slw = np.array([r[1] for r in rows])
        adapter_summary[name] = {
            "slp_mean": float(slp.mean()), "slp_std": float(slp.std()),
            "slw_mean": float(slw.mean()), "slw_std": float(slw.std()), "n": len(rows),
        }
        print(f"  {name:12s} s_lp: mean={slp.mean():.3f} std={slp.std():.3f}  "
              f"s_lw: mean={slw.mean():.3f} std={slw.std():.3f}  n={len(rows)}")

    with open(REPO_ROOT / "scripts" / "_e10_adapter_dist_data.json", "w") as f:
        json.dump(adapter_summary, f, indent=1)

    # -----------------------------------------------------------------
    # Table/Fig: computation time -- one-time training (kmeans + RDO) vs.
    # recurring per-frame encode cost, by (child_b, K).
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE: e10_compute_time (training: kmeans + RDO calibration; K=24)")
    print("=" * 100)
    print(f"{'sequence':12s} | {'kmeans_s':>9s} | {'rdo_s':>9s} | {'train_total_s':>13s}")
    train_summary = {}
    for name in SEQUENCES:
        seq_id = c.execute("SELECT seq_id FROM sequences WHERE name=?", (name,)).fetchone()[0]
        # kmeans_iterations.K counts DIRECTIONAL clusters only (codebook K minus the
        # structural mode 0), so codebook K=24 -> kmeans_iterations K=23.
        kmeans_s = c.execute("""
            SELECT COALESCE(SUM(iteration_seconds), 0) FROM kmeans_iterations WHERE seq_id=? AND K=23
        """, (seq_id,)).fetchone()[0]
        rdo_s = 0.0
        Qs_for_adapters = [r[0] for r in c.execute(
            "SELECT DISTINCT Q FROM mode_adapters WHERE seq_id=? AND K=24", (seq_id,)).fetchall()]
        for Q in Qs_for_adapters:
            v = c.execute("""
                SELECT MAX(training_seconds) FROM mode_adapters WHERE seq_id=? AND K=24 AND Q=?
            """, (seq_id, Q)).fetchone()[0]
            if v:
                rdo_s += v
        train_summary[name] = {"kmeans_s": kmeans_s, "rdo_s": rdo_s, "total_s": kmeans_s + rdo_s}
        print(f"{name:12s} | {kmeans_s:9.2f} | {rdo_s:9.2f} | {kmeans_s + rdo_s:13.2f}")

    # item 12: split by family instead of pooling all 14 sequences into one (child_b,K) plot --
    # not every sequence takes the same time, and pooling hid why. (FAMILIES defined at top of main().)
    print(f"\n{'family':14s} | {'child_b':>7s} | {'K':>3s} | {'n_frames':>8s} | {'mean_s':>8s} | {'std_s':>7s} | {'min_s':>7s} | {'max_s':>7s}")
    encode_summary = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for fi, (fam_name, fam_seqs) in enumerate(FAMILIES):
        ax = axes[fi]
        placeholders = ",".join("?" * len(fam_seqs))
        for cb in (4, 8, 16):
            means, stds, Ks = [], [], []
            for K in (4, 8, 16, 24, 30):
                rows = c.execute(f"""
                    SELECT fr.frame_runtime_seconds FROM frame_results fr
                    JOIN sequences se ON se.seq_id = fr.seq_id
                    WHERE se.name IN ({placeholders}) AND fr.child_b=? AND fr.K=?
                    AND fr.frame_runtime_seconds IS NOT NULL
                """, (*fam_seqs, cb, K)).fetchall()
                if not rows:
                    continue
                vals = np.array([r[0] for r in rows])
                print(f"{fam_name:14s} | {cb:7d} | {K:3d} | {len(vals):8d} | {vals.mean():8.2f} | {vals.std():7.2f} | "
                      f"{vals.min():7.2f} | {vals.max():7.2f}")
                encode_summary.append({"family": fam_name, "child_b": cb, "K": K, "n": len(vals),
                                        "mean_s": float(vals.mean()), "std_s": float(vals.std()),
                                        "min_s": float(vals.min()), "max_s": float(vals.max())})
                means.append(vals.mean()); stds.append(vals.std()); Ks.append(K)
            ax.errorbar(Ks, means, yerr=stds, marker="o", capsize=3, label=f"$B_c={cb}$")
        ax.set_title(fam_name, fontsize=10)
        ax.set_xlabel("$K$")
        if fi == 0:
            ax.set_ylabel("tiempo de codificación por frame (s)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_compute_time.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_compute_time.png (split by family)")

    with open(REPO_ROOT / "scripts" / "_e10_compute_time_data.json", "w") as f:
        json.dump({"train": train_summary, "encode": encode_summary}, f, indent=1)

    # -----------------------------------------------------------------
    # Table (item 11): computational cost by (B_p, B_c, K) -- a genuinely different cut than the
    # per-sequence multiframe BD-Rate table: this one is about COST, grouped by the structural
    # parameters that drive it, not by sequence identity.
    # -----------------------------------------------------------------
    print("\nTABLE: costo computacional por (B_p, B_c, K) -- tiempo de codificación por frame, todas las secuencias")
    cost_by_bpbck = []
    for bp, depth_seqs in [(32, [n for n in SEQUENCES if n.endswith("10") or n in DEPTH10_8I]),
                            (16, [n for n in SEQUENCES if n.endswith("9")])]:
        placeholders = ",".join("?" * len(depth_seqs))
        for cb in (4, 8, 16):
            for K in (4, 8, 16, 24, 30):
                rows = c.execute(f"""
                    SELECT fr.frame_runtime_seconds FROM frame_results fr
                    JOIN sequences se ON se.seq_id = fr.seq_id
                    WHERE se.name IN ({placeholders}) AND fr.child_b=? AND fr.K=?
                    AND fr.frame_runtime_seconds IS NOT NULL
                """, (*depth_seqs, cb, K)).fetchall()
                if not rows:
                    continue
                vals = np.array([r[0] for r in rows])
                cost_by_bpbck.append({"B_p": bp, "B_c": cb, "K": K, "n": len(vals),
                                       "mean_s": float(vals.mean()), "std_s": float(vals.std())})
                print(f"  B_p={bp:2d} B_c={cb:2d} K={K:2d}  n={len(vals):3d}  "
                      f"mean={vals.mean():7.2f}s  std={vals.std():6.2f}s")

    with open(REPO_ROOT / "scripts" / "_e10_cost_by_bpbck.json", "w") as f:
        json.dump(cost_by_bpbck, f, indent=1)

    # -----------------------------------------------------------------
    # Multi-frame BD-Rate/BD-PSNR aggregation, GENERALIZED over ALL K (not just K=24, as this
    # section originally hardcoded -- direct query confirmed multi-frame coverage exists at
    # K in {4,16,24,30} for nearly all 14 sequences; K=8 is the actual outlier, only 1 frame
    # at most (seq,B_c) cells except loot/redandblack/soldier). Feeds two table sets in
    # resultados.tex:
    #   - Table Set 1 (tab:e10_bd_ratio1/2/4/8): K=24 only, grouped by ratio B_p/B_c (never
    #     mixing ratios within one table -- the original tab:e10_bd_bc4/8/16 mixed two ratios
    #     per table since the same nominal B_c means a different B_p/B_c for depth-9 vs.
    #     depth-10/8iVFB); BD-Rate + BD-PSNR mean+-std, bold best row per table.
    #   - Table Set 2 (tab:e10_bd_classic_bc4/8/16, +32 longdress-only): "classic" table grouped
    #     by B_c, ALL K in {4,8,16,24,30}, one row per sequence. Multi-frame mean+-std wherever
    #     >=4 per-frame BD-Rate points exist, falling back to the frame_idx=1 single point
    #     (f1/load_f1_rows) where multi-frame coverage is too thin (mainly K=8 cells). Bold the
    #     best K per (sequence,B_c) row.
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Multi-frame BD-Rate/BD-PSNR aggregation, all K")
    print("=" * 100)
    mf_rows = c.execute("""
        SELECT se.name, fr.child_b, fr.K, fr.frame_idx, fr.q_step, fr.frame_bpv, fr.frame_psnr_Y,
               fr.baseline_bpv, fr.baseline_psnr_Y
        FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
        ORDER BY se.name, fr.child_b, fr.K, fr.frame_idx, fr.q_step
    """).fetchall()
    mf_data = defaultdict(list)
    for name, cb, K, fidx, Q, bpv, psnr, bbpv, bpsnr in mf_rows:
        mf_data[(name, cb, K, fidx)].append({"Q": Q, "bpv": bpv, "psnr": psnr, "bbpv": bbpv, "bpsnr": bpsnr})

    def multiframe_bd_stats(name, cb, K):
        """Mean+-std BD-Rate/BD-PSNR across all available frames for (name,cb,K), counting only
        frames with >=4 Q points. Returns None if fewer than 4 valid per-frame BD-Rate points
        end up available (matches the existing small-n-but-real-repeated-measures threshold)."""
        frame_idxs = sorted(set(fidx for (n, c_, k_, fidx) in mf_data if n == name and c_ == cb and k_ == K))
        per_frame_bdr, per_frame_bdp = [], []
        for fidx in frame_idxs:
            qs = sorted(mf_data[(name, cb, K, fidx)], key=lambda r: r["Q"])
            if len(qs) < 4:
                continue
            bbpv = [r["bbpv"] for r in qs]; bpsnr = [r["bpsnr"] for r in qs]
            bpv = [r["bpv"] for r in qs]; psnr = [r["psnr"] for r in qs]
            bd_r = bdrate(bbpv, bpsnr, bpv, psnr)
            bd_p = bdpsnr(bbpv, bpsnr, bpv, psnr)
            if not math.isnan(bd_r):
                per_frame_bdr.append(bd_r)
            if not math.isnan(bd_p):
                per_frame_bdp.append(bd_p)
        if len(per_frame_bdr) < 4:
            return None
        return {
            "bd_rate_mean": float(np.mean(per_frame_bdr)), "bd_rate_std": float(np.std(per_frame_bdr)),
            "bd_psnr_mean": float(np.mean(per_frame_bdp)) if per_frame_bdp else float("nan"),
            "bd_psnr_std": float(np.std(per_frame_bdp)) if per_frame_bdp else float("nan"),
            "n": len(per_frame_bdr), "source": "multiframe",
        }

    def f1_bd_stats(name, cb, K):
        """frame_idx=1 fallback: single-point BD-Rate/BD-PSNR, no std (n=1)."""
        qs = f1.get((name, cb, K))
        if not qs or len(qs) < 4:
            return None
        qs_sorted = sorted(qs, key=lambda r: r["Q"])
        bpv = [r["bpv"] for r in qs_sorted]; psnr = [r["psnr"] for r in qs_sorted]
        bbpv = [r["bbpv"] for r in qs_sorted]; bpsnr = [r["bpsnr"] for r in qs_sorted]
        bd_r = bdrate(bbpv, bpsnr, bpv, psnr); bd_p = bdpsnr(bbpv, bpsnr, bpv, psnr)
        if math.isnan(bd_r):
            return None
        return {"bd_rate_mean": bd_r, "bd_rate_std": None, "bd_psnr_mean": bd_p, "bd_psnr_std": None,
                "n": 1, "source": "frame1"}

    # --- Table Set 1: by ratio B_p/B_c, K=24 only, multi-frame only (no frame1 fallback -- this
    # table set's whole point is genuine repeated-measures mean+-std) ---
    print("\nTABLE SET 1: BD-Rate/BD-PSNR by ratio B_p/B_c, K=24, multi-frame only")
    RATIO_CB_BY_DEPTH = {9: {1: 16, 2: 8, 4: 4}, 10: {2: 16, 4: 8, 8: 4}}
    ratio_tables = {}
    for ratio in (1, 2, 4, 8):
        rows = []
        for name in SEQUENCES:
            depth = 9 if name.endswith("9") else 10
            cb = RATIO_CB_BY_DEPTH[depth].get(ratio)
            if cb is None:
                continue
            stats = multiframe_bd_stats(name, cb, 24)
            if stats is None:
                continue
            rows.append({"name": name, "depth": depth, "B_p": 16 if depth == 9 else 32, "B_c": cb, **stats})
        if rows:
            best = min(rows, key=lambda r: r["bd_rate_mean"])["name"]
            for r in rows:
                r["best"] = (r["name"] == best)
            ratio_tables[ratio] = rows
            print(f"  ratio={ratio}: {len(rows)} sequences, best={best}")
            for r in rows:
                print(f"    {r['name']:12s} B_p={r['B_p']:2d} B_c={r['B_c']:2d} "
                      f"BD-Rate={r['bd_rate_mean']:+.3f}+-{r['bd_rate_std']:.3f}  "
                      f"BD-PSNR={r['bd_psnr_mean']:+.3f}+-{r['bd_psnr_std']:.3f}  n={r['n']}"
                      + ("  [BEST]" if r["best"] else ""))

    with open(REPO_ROOT / "scripts" / "_e10_bd_ratio_tables.json", "w") as f:
        json.dump(ratio_tables, f, indent=1, default=str)

    # --- Table Set 2: "classic" table grouped by B_c, all K, multi-frame-first / frame1-fallback ---
    print("\nTABLE SET 2: 'classic' BD-Rate by B_c, all K, multi-frame-first / frame_idx=1 fallback")
    CLASSIC_KS = [4, 8, 16, 24, 30]
    classic_tables = {}
    for cb in (4, 8, 16):
        rows = []
        for name in SEQUENCES:
            cell = {}
            for K in CLASSIC_KS:
                stats = multiframe_bd_stats(name, cb, K) or f1_bd_stats(name, cb, K)
                if stats is not None:
                    cell[K] = stats
            if cell:
                best_K = min(cell, key=lambda k: cell[k]["bd_rate_mean"])
                rows.append({"name": name, "cells": cell, "best_K": best_K})
        classic_tables[str(cb)] = rows
        print(f"  B_c={cb}: {len(rows)} sequences")
        for r in rows:
            parts = []
            for K in CLASSIC_KS:
                if K in r["cells"]:
                    v = r["cells"][K]
                    tag = "*" if K == r["best_K"] else " "
                    std_str = f"+-{v['bd_rate_std']:.2f}" if v["bd_rate_std"] is not None else "(f1)"
                    parts.append(f"K={K}:{v['bd_rate_mean']:+.2f}{std_str}{tag}")
            print(f"    {r['name']:12s} " + "  ".join(parts))

    # longdress B_c=32 case (excluded from f1/load_f1_rows -- query its frame_idx=1 fallback
    # directly; only K in {16,24} have coverage at all per earlier exploration).
    rows32 = []
    for K in (16, 24):
        stats = multiframe_bd_stats("longdress", 32, K)
        if stats is None:
            r32 = c.execute("""
                SELECT fr.q_step, fr.frame_bpv, fr.frame_psnr_Y, fr.baseline_bpv, fr.baseline_psnr_Y
                FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
                WHERE se.name='longdress' AND fr.child_b=32 AND fr.K=? AND fr.frame_idx=1
                ORDER BY fr.q_step
            """, (K,)).fetchall()
            if len(r32) >= 4:
                bpv = [r[1] for r in r32]; psnr = [r[2] for r in r32]
                bbpv = [r[3] for r in r32]; bpsnr = [r[4] for r in r32]
                bd_r = bdrate(bbpv, bpsnr, bpv, psnr); bd_p = bdpsnr(bbpv, bpsnr, bpv, psnr)
                if not math.isnan(bd_r):
                    stats = {"bd_rate_mean": bd_r, "bd_rate_std": None, "bd_psnr_mean": bd_p,
                              "bd_psnr_std": None, "n": 1, "source": "frame1"}
        if stats:
            rows32.append({"K": K, **stats})
    if rows32:
        classic_tables["32_longdress"] = rows32
        print("  B_c=32 (longdress only): " + "  ".join(f"K={r['K']}:{r['bd_rate_mean']:+.2f}" for r in rows32))

    with open(REPO_ROOT / "scripts" / "_e10_bd_classic_tables.json", "w") as f:
        json.dump(classic_tables, f, indent=1, default=str)

    # -----------------------------------------------------------------
    # Fig: k-means convergence (max centroid shift, not just inertia) + mode-frequency,
    # 3 representative sequences (one per family: 8iVFB, MVUB-9, MVUB-10).
    # Data source: kmeans_iterations, cluster_distribution -- both already fully populated,
    # no new computation logic.
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIG: e10_convergence_frequency.png (centroid shift convergence, 6 sequences)")
    print("=" * 100)
    # 2 per family: the two 8iVFB you named (longdress, loot), plus best/worst BD-Rate (K=24,
    # multiframe -- computed below in this same script) for MVUB-9 and MVUB-10. Mode-frequency
    # (the old bottom row) is dropped here -- it's now covered, weighted by usage and across all
    # 14 sequences, by the codebook-directions figure above (item 18); keeping both was redundant.
    CONV_FIG_SEQS = ["longdress", "loot", "phil9", "ricardo9", "phil10", "ricardo10"]
    CONV_KS = [4, 8, 16, 24, 30]  # codebook K; kmeans_iterations stores K-1 (directional count)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    kcolors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONV_KS)))
    for si, name in enumerate(CONV_FIG_SEQS):
        seq_id = c.execute("SELECT seq_id FROM sequences WHERE name=?", (name,)).fetchone()[0]
        ax = axes[si // 3, si % 3]
        for K, color in zip(CONV_KS, kcolors):
            rows = c.execute("""
                SELECT iteration, max_centroid_shift_deg FROM kmeans_iterations
                WHERE seq_id=? AND K=? ORDER BY iteration
            """, (seq_id, K - 1)).fetchall()
            if not rows:
                continue
            it = [r[0] for r in rows]
            shift = [max(r[1], 1e-7) for r in rows]  # floor for log scale
            ax.plot(it, shift, "-o", color=color, markersize=3, linewidth=1.2, label=f"K={K}")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        if si // 3 == 1:
            ax.set_xlabel("iteración")
        if si % 3 == 0:
            ax.set_ylabel(r"max. desplaz. de centroide ($\log_{10}$, grados)")
        ax.grid(alpha=0.3, which="both")
        if si == 2:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_convergence_frequency.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_convergence_frequency.png (6 sequences: 2 per family)")

    # -----------------------------------------------------------------
    # Table: convergence + self-loop-search summary, ALL 14 sequences (item 19: extend beyond
    # the figure's 6; item 20: fold the adapter-search trace -- previously its own figure panel
    # -- into this same table instead of a separate plot).
    # -----------------------------------------------------------------
    print("\nTABLE: convergencia k-means + resumen de búsqueda de auto-bucle, las 14 secuencias, K=24")
    conv_summary = {}
    for name in SEQUENCES:
        seq_id = c.execute("SELECT seq_id FROM sequences WHERE name=?", (name,)).fetchone()[0]
        krows = c.execute("""
            SELECT iteration, n_reassigned, max_centroid_shift_deg, converged
            FROM kmeans_iterations WHERE seq_id=? AND K=23 ORDER BY iteration
        """, (seq_id,)).fetchall()
        entry = {}
        if krows:
            last = krows[-1]
            entry.update({
                "n_iterations": len(krows), "converged": bool(last[3]),
                "final_n_reassigned": last[1], "final_shift_deg": last[2],
            })
        # Self-loop (adapter) search summary at K=24, Q=24, averaged over directional modes:
        # cost reduction from the start of the coarse grid to the end of golden-section refinement.
        steps = c.execute("""
            SELECT mode_idx, step_type, step_idx, s_lp, s_lw, rd_cost FROM adapter_opt_steps
            WHERE seq_id=? AND K=24 AND Q=24 ORDER BY mode_idx, step_type, step_idx
        """, (seq_id,)).fetchall()
        by_mode = defaultdict(list)
        for mode_idx, step_type, step_idx, s_lp, s_lw, rd_cost in steps:
            by_mode[mode_idx].append((step_type, rd_cost, s_lp, s_lw))
        reductions, n_steps_total = [], 0
        for mode_idx, srows in by_mode.items():
            n_steps_total += len(srows)
            costs = [r[1] for r in srows]
            if len(costs) >= 2 and costs[0] > 0:
                reductions.append((costs[0] - min(costs)) / costs[0] * 100.0)
        if reductions:
            entry["selfloop_search_mean_cost_reduction_pct"] = float(np.mean(reductions))
            entry["selfloop_search_n_steps_total"] = n_steps_total
            entry["selfloop_search_n_modes"] = len(by_mode)
        conv_summary[name] = entry
        conv_line = f"  {name:12s}"
        if "n_iterations" in entry:
            conv_line += (f"  kmeans: n_iter={entry['n_iterations']:2d} converged={str(entry['converged']):5s} "
                          f"final_shift={entry['final_shift_deg']:.2e}deg")
        if "selfloop_search_mean_cost_reduction_pct" in entry:
            conv_line += (f"  |  self-loop search: {entry['selfloop_search_n_modes']} modos, "
                          f"{entry['selfloop_search_n_steps_total']} evals, "
                          f"reducción de costo media={entry['selfloop_search_mean_cost_reduction_pct']:.1f}%")
        print(conv_line)

    with open(REPO_ROOT / "scripts" / "_e10_convergence_data.json", "w") as f:
        json.dump(conv_summary, f, indent=1)

    print("\nDone. See docs/thesis/img/resultados/ for PNGs and scripts/_e10_thesis_table_data.json for table values.")


if __name__ == "__main__":
    main()
