#!/usr/bin/env python3
"""
Multi-sequence rebuild of "reporte_es"'s 6 figures (docs/report/main_es.tex), sourced from
results/exp_e10_multisequence.db (14 sequences) instead of the original single-sequence
results/exp_e9_multi_entropy_suite.db (longdress only, hardcoded).

Overwrites the same 6 filenames under docs/report/figures/ so main_es.tex needs no edits;
recompile main_es.pdf afterward.

Visual style (rcParams, per-K color/marker scheme, seaborn-v0_8-whitegrid, Spanish labels) is
copied verbatim from docs/report/generate_all_report_figures_e9.py on purpose — the user likes
that look and asked to keep it, just extended to more sequences.

Notes on what changed vs. the original, single-sequence figures (see plan / conversation for the
reasoning, confirmed with the user before implementing):
  - fig1 / fig_annex4: a single sequence's RD/switches/overhead curve becomes a mean curve across
    sequences at each K, n=14 (fig1's B_c=32 panel stays longdress-only, n=1 -- that scale was
    never extended to other sequences, said so in the title rather than silently averaging one
    point). PLAIN mean lines, no error bars -- checked the real spread directly (BPV at
    B_c=16,K=24,Q=24 ranges 0.223 to 2.166 across sequences, a 10x spread; trellis switches range
    204-1103, 5.4x) and error bars on that scale just draw dominating whiskers, not a readable
    chart. The spread is real and large; it belongs in the caption as a number, not as a mark that
    swamps the plot.
  - fig7: same 5-control x 4-metric Pearson matrix, now pooling frame_idx=1 rows across all 14
    sequences instead of one sequence's ~64 grid points. Diverging colormap (coolwarm) instead of
    PiYG -- data is signed correlation in [-1,1], coolwarm reads that split more directly.
  - fig_annex1 / fig_annex2: inherently per-object 3D visualizations become small multiples over
    4 representative sequences (longdress, soldier: 8iVFB; andrew10, ricardo9: MVUB, one per depth)
    instead of all 14 (unreadable) or just one.
  - fig_annex3: single-panel now (k-means convergence only). The original plotted fabricated
    placeholder numbers here, not a DB query at all; a first fix replaced it with a real RDO
    search-cost trace (adapter_opt_steps.rd_cost, normalized per-sequence) but that panel didn't
    hold up as a plot either -- 14 mostly-flat, overlapping lines with a few noisy spikes, no real
    read. Dropped rather than kept for the sake of having two panels.
  - fig_annex4: switched source scale from B_c=32 (n=1, same coverage problem as fig1) to B_c=16
    (full 14-sequence coverage at all K) -- a deliberate, disclosed substitution.

Every K series in an aggregated (mean-across-sequences) plot is gated on MIN_N_FULL sequences
actually having complete data (see complete_n()) -- K=30 is currently excluded everywhere below
because the K=16/K=30 background run has only reached 2/14 sequences so far.
"""
import os, sys, sqlite3, zipfile
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # docs/report/<this file> -> repo root
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcadc.pointcloud import MortonBlockPartition
from pcadc.color import Colourist
from exp_e9_multi_entropy_suite import extract_block_tangent
from exp_e10_multisequence_benchmark import (
    SEQUENCES as SEQ_TUPLES, get_bp, load_ply_from_zip, make_pc,
)

FIG_DIR = REPO_ROOT / "docs" / "report" / "figures"
DB = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
FRAMES_DIR = REPO_ROOT / "res" / "frames"
SEQ_META = {n: (zf, d) for (n, zf, d) in SEQ_TUPLES}

ALL_SEQUENCES = ["longdress", "loot", "redandblack", "soldier",
                  "andrew10", "david10", "phil10", "ricardo10", "sarah10",
                  "andrew9", "david9", "phil9", "ricardo9", "sarah9"]
REP_SEQUENCES = ["longdress", "soldier", "andrew10", "ricardo9"]  # 3D small multiples (Q2 answer)
Q_STEPS = [24, 32, 48, 64]
MIN_N_FULL = len(ALL_SEQUENCES)  # 14 -- these figures are captioned "n=14"; a K series that
                                  # doesn't actually have all 14 sequences complete yet (e.g. the
                                  # K=16/K=30 stride-50 run was still in progress) must not be
                                  # silently included under that same "n=14" label.


def complete_n(f1, cb, K):
    """How many sequences have a full 4-Q entry for (child_b=cb, K). Used as the sufficient-runs
    gate before a K series is allowed into an aggregated (mean-across-sequences) plot."""
    return sum(1 for (n, ccb, KK), rs in f1.items() if ccb == cb and KK == K and len(rs) == 4)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "figure.dpi": 300, "axes.grid": True, "grid.alpha": 0.35,
})
COLORS = {24: "#1f77b4", 16: "#2ca02c", 8: "#ff7f0e", 4: "#d62728"}
MARKERS = {24: "o", 16: "s", 8: "^", 4: "v"}


def conn():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def load_f1_rows(c):
    """frame_idx=1 rows, all 14 sequences. Excludes longdress's lone B_c=32 rows (kept as a
    separate n=1 query in fig1) -- same exclusion scripts/generate_e10_thesis_figures.py uses,
    for consistency with the rest of the thesis/report figures this session produced."""
    rows = c.execute("""
        SELECT se.name, fr.child_b, fr.K, fr.q_step, fr.frame_bpv, fr.frame_psnr_Y,
               fr.baseline_bpv, fr.baseline_psnr_Y, fr.tau_gate, fr.n_trellis_switches,
               fr.cabac_overhead_bits, fr.n_voxels, fr.gated_pct
        FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
        WHERE fr.frame_idx = 1 AND NOT (se.name='longdress' AND fr.child_b=32)
        ORDER BY se.name, fr.child_b, fr.K, fr.q_step
    """).fetchall()
    data = defaultdict(list)
    for name, cb, K, Q, bpv, psnr, bbpv, bpsnr, tau, sw, cabac, nvox, gated in rows:
        data[(name, cb, K)].append(dict(Q=Q, bpv=bpv, psnr=psnr, bbpv=bbpv, bpsnr=bpsnr,
                                         tau=tau, sw=sw, cabac=cabac, nvox=nvox, gated=gated))
    return data


# =============================================================================
# 1. FIG 1: RD CURVES, 2x2 BY B_c -- MEAN +/- STD ACROSS SEQUENCES
# =============================================================================
def fig1(c, f1):
    print("[1/6] fig1_rd_curves_zoomed.png (mean RD curve +/- std across sequences)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    axes = axes.flatten()
    for idx, B in enumerate([32, 16, 8, 4]):
        ax = axes[idx]
        if B == 32:
            rows = c.execute("""
                SELECT fr.q_step, fr.K, fr.frame_bpv, fr.frame_psnr_Y, fr.baseline_bpv, fr.baseline_psnr_Y
                FROM frame_results fr JOIN sequences se ON se.seq_id=fr.seq_id
                WHERE se.name='longdress' AND fr.child_b=32 AND fr.frame_idx=1 ORDER BY fr.K, fr.q_step
            """).fetchall()
            by_k = defaultdict(list)
            for Q, K, bpv, psnr, bbpv, bpsnr in rows:
                by_k[K].append((Q, bpv, psnr, bbpv, bpsnr))
            base_plotted = False
            for K in sorted(by_k, reverse=True):
                pts = sorted(by_k[K])
                ax.plot([p[1] for p in pts], [p[2] for p in pts], color=COLORS.get(K, "gray"),
                        marker=MARKERS.get(K, "o"), linewidth=1.9, markersize=7, label=f"PCADC K={K}")
                if not base_plotted:
                    ax.plot([p[3] for p in pts], [p[4] for p in pts], "k*--", linewidth=2.2,
                            markersize=9, label="Linea Base Estructural", zorder=6)
                    base_plotted = True
            ax.set_title(f"Escala de Sub-bloque $B_c={B}$ -- solo longdress ($n=1$)", fontweight="bold")
        else:
            Ks_all = sorted({K for (n, cb, K) in f1 if cb == B}, reverse=True)
            for K in Ks_all:
                n_seq = complete_n(f1, B, K)
                if n_seq < MIN_N_FULL:
                    print(f"    [skip] B_c={B} K={K}: only {n_seq}/{MIN_N_FULL} sequences complete "
                          f"(background run still filling this in) -- excluded from the aggregate")
                    continue
                mean_bpv, std_bpv, mean_psnr, std_psnr = [], [], [], []
                for Q in Q_STEPS:
                    bpvs = [r["bpv"] for (n, cb, KK), rs in f1.items() if cb == B and KK == K
                            for r in rs if r["Q"] == Q]
                    psnrs = [r["psnr"] for (n, cb, KK), rs in f1.items() if cb == B and KK == K
                             for r in rs if r["Q"] == Q]
                    mean_bpv.append(np.mean(bpvs)); std_bpv.append(np.std(bpvs))
                    mean_psnr.append(np.mean(psnrs)); std_psnr.append(np.std(psnrs))
                # Plain mean line -- NOT error bars. Checked directly: cross-sequence BPV spread at
                # this scale is 5-10x (content heterogeneity, not noise), so an xerr/yerr whisker
                # dwarfs the curve. The real spread is printed below for the caption/prose instead.
                ax.plot(mean_bpv, mean_psnr, marker=MARKERS.get(K, "o"), color=COLORS.get(K, "gray"),
                        linewidth=1.9, markersize=7, label=f"PCADC K={K} (media, n={n_seq})")
                if K == max(k for k in Ks_all if complete_n(f1, B, k) >= MIN_N_FULL):
                    cv = [s / m * 100 if m else 0 for s, m in zip(std_bpv, mean_bpv)]
                    print(f"    [sanity] B_c={B} K={K}: BPV coef.-of-variation across sequences by Q = "
                          f"{[f'{v:.0f}%' for v in cv]} (why no error bars)")
            # Baseline is identical across K for a given (seq,B,Q) -- average it from ONE
            # already-threshold-passed K rather than pooling across all K (would otherwise let
            # the still-incomplete K=30 rows contribute extra, non-representative baseline samples).
            ref_K = next((K for K in Ks_all if complete_n(f1, B, K) >= MIN_N_FULL), None)
            base_mean_bpv, base_mean_psnr = [], []
            for Q in Q_STEPS:
                bbpvs = [r["bbpv"] for (n, cb, KK), rs in f1.items() if cb == B and KK == ref_K
                         for r in rs if r["Q"] == Q]
                bpsnrs = [r["bpsnr"] for (n, cb, KK), rs in f1.items() if cb == B and KK == ref_K
                          for r in rs if r["Q"] == Q]
                base_mean_bpv.append(np.mean(bbpvs)); base_mean_psnr.append(np.mean(bpsnrs))
            ax.plot(base_mean_bpv, base_mean_psnr, "k*--", linewidth=2.2, markersize=9,
                    label="Linea Base Estructural (media, n=14)", zorder=6)
            ax.set_title(f"Escala de Sub-bloque $B_c={B}$ (media, $n=14$ secuencias)", fontweight="bold")
        ax.set_xlabel("Tasa Real de Bits por Voxel (BPV)")
        ax.set_ylabel("PSNR Decodificado $Y$ (dB)")
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_rd_curves_zoomed.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 2. FIG 7: SENSITIVITY MATRIX -- POOLED n=14
# =============================================================================
def fig7(c):
    print("[2/6] fig7_algorithmic_sensitivity_matrix.png (pooled, n=14 sequences)...")
    # K=30 excluded: the K=16/K=30 stride-50 run is still in progress (2/14 sequences complete
    # at the time this ran -- see fig1's sanity check), so its rows would be a small, biased,
    # non-representative slice of the pooled grid rather than real n=14 signal.
    rows = c.execute("""
        SELECT fr.tau_gate, fr.K, fr.child_b, fr.q_step, fr.delta_pct, fr.n_trellis_switches,
               fr.cabac_overhead_bits, fr.baseline_bpv, fr.frame_bpv, fr.n_voxels, fr.gated_pct
        FROM frame_results fr JOIN sequences se ON se.seq_id = fr.seq_id
        WHERE fr.frame_idx = 1 AND NOT (se.name='longdress' AND fr.child_b=32) AND fr.K != 30
    """).fetchall()
    (tau, K, B_c, Q, delta_pct, switches, cabac, bbpv, fbpv, nvox, gated) = (
        np.array(x, dtype=float) for x in zip(*rows)
    )
    print(f"    [sanity] pooled rows: {len(rows)} (expect 14 seq x 3 B_c x 4 K x 4 Q = 672, "
          f"minus longdress's extra B_c=32 K in {{16,24}} already excluded above)")
    gamma = np.full_like(tau, 0.001)  # E10 fixes gamma -- zero variance, kept for parity with
                                       # reporte_es's control set; correlation falls out at 0 below.
    savings_bits = (bbpv - fbpv) * nvox
    oh_ratio = np.where(savings_bits > 0, cabac / np.maximum(savings_bits, 1e-9), 0.0)
    gft_coverage = 100.0 - gated  # % of parent blocks NOT gated to mode 0 (actively using GFT)

    controls = {"tau_gate": tau, "K": K, "gamma": gamma, "B_c": B_c, "Q": Q}
    metrics = {"Ahorro Real %": delta_pct, "Conmutaciones Trellis": switches,
               "Ratio OH:Ahorro": oh_ratio, "Cobertura GFT %": gft_coverage}

    sens = np.zeros((len(controls), len(metrics)))
    for i, cv in enumerate(controls.values()):
        for j, mv in enumerate(metrics.values()):
            mask = np.isfinite(cv) & np.isfinite(mv)
            if np.std(cv[mask]) > 1e-12 and np.std(mv[mask]) > 1e-12:
                sens[i, j] = np.corrcoef(cv[mask], mv[mask])[0, 1]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    im = ax.imshow(sens, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title("Matriz de Sensibilidad de Parametros Algoritmicos (E10, $n=14$ secuencias)",
                 fontweight="bold", fontsize=12, pad=10)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(list(metrics), fontweight="bold")
    ax.set_yticks(range(len(controls)))
    ax.set_yticklabels([r"$\tau_{\mathrm{gate}}$", "$K$", r"$\gamma$", r"$B_c$", r"$Q$"],
                       fontweight="bold", fontsize=11)
    for i in range(len(controls)):
        for j in range(len(metrics)):
            v = sens[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontweight="bold", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlacion de Pearson $r$", rotation=270, labelpad=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_algorithmic_sensitivity_matrix.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 3. FIG A1: SPHERICAL CENTROIDS -- SMALL MULTIPLE, 4 SEQUENCES, K=24
# =============================================================================
def fig_annex1(c):
    print("[3/6] fig_annex1_spherical_centroids_3d.png (4 representative sequences, K=24)...")
    fig = plt.figure(figsize=(16, 4.2), dpi=300)
    for idx, name in enumerate(REP_SEQUENCES):
        ax = fig.add_subplot(1, len(REP_SEQUENCES), idx + 1, projection="3d")
        u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.6)

        rows = c.execute("""SELECT cc.mode_idx, cc.cx, cc.cy, cc.cz FROM cluster_centroids cc
                             JOIN sequences se ON se.seq_id=cc.seq_id WHERE se.name=? AND cc.K=24
                             ORDER BY cc.mode_idx""", (name,)).fetchall()
        colors_k = cm.tab20(np.linspace(0, 1, len(rows)))
        for r_idx, (m, cx, cy, cz) in enumerate(rows):
            ax.quiver(0, 0, 0, cx, cy, cz, color=colors_k[r_idx], arrow_length_ratio=0.12,
                      linewidth=1.8, alpha=0.9)
            ax.scatter([cx], [cy], [cz], color=colors_k[r_idx], s=42, edgecolor="black", zorder=5)
        ax.set_title(f"{name} ($K=24$)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("$X$", labelpad=3); ax.set_ylabel("$Y$", labelpad=3); ax.set_zlabel("$Z$", labelpad=3)
        ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
        ax.view_init(elev=22, azim=45)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_annex1_spherical_centroids_3d.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. FIG A2: SPATIAL 3D LAYOUT -- SMALL MULTIPLE, 4 SEQUENCES
# =============================================================================
def _load_pc_frame1(name):
    zip_fn, depth = SEQ_META[name]
    zip_path = FRAMES_DIR / zip_fn
    with zipfile.ZipFile(zip_path, "r") as z:
        names_in_zip = sorted(n2 for n2 in z.namelist() if n2.lower().endswith(".ply"))
    V_raw, A_raw = load_ply_from_zip(zip_path, names_in_zip[1])  # frame_idx=1
    pc = make_pc(V_raw, A_raw, name, depth, 1)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    return pc, depth


def fig_annex2(c):
    print("[4/6] fig_annex2_spatial_3d_pointcloud_layout.png (4 representative sequences)...")
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(15, 4.2 * len(REP_SEQUENCES)), dpi=300)
    for row_idx, name in enumerate(REP_SEQUENCES):
        pc, depth = _load_pc_frame1(name)
        Bp = get_bp(depth)
        _, blocks_p = MortonBlockPartition().partition(pc, bsize=Bp)

        centroid_rows = c.execute("""SELECT cc.mode_idx, cc.cx, cc.cy, cc.cz, cc.theta_spherical
                                      FROM cluster_centroids cc JOIN sequences se ON se.seq_id=cc.seq_id
                                      WHERE se.name=? AND cc.K=24 ORDER BY cc.mode_idx""", (name,)).fetchall()
        cvecs = np.array([[r[1], r[2], r[3]] for r in centroid_rows])
        thetas = np.array([r[4] for r in centroid_rows])

        parent_centers, block_theta = [], []
        for b in blocks_p:
            t = extract_block_tangent(b, pc.V, pc.A)
            b.init_data(pc.V, pc.A)
            parent_centers.append(np.mean(b.Vblock, axis=0))
            b.clear_data()
            dots = np.abs(cvecs @ t["g_norm"])
            block_theta.append(thetas[int(np.argmax(dots))])
        parent_centers = np.array(parent_centers); block_theta = np.array(block_theta)

        ax1 = fig.add_subplot(len(REP_SEQUENCES), 2, 2 * row_idx + 1, projection="3d")
        sample_idx = rng.choice(len(pc.V), size=min(35000, len(pc.V)), replace=False)
        p1 = ax1.scatter(pc.V[sample_idx, 0], pc.V[sample_idx, 2], pc.V[sample_idx, 1],
                         c=pc.A[sample_idx, 0], cmap="viridis", s=1.0, alpha=0.6)
        ax1.set_title(f"(a) {name}: Nube 3D y Luminancia $Y$ ($N={len(pc.V):,}$)".replace(",", "{,}"),
                      fontsize=10, fontweight="bold")
        ax1.set_xlabel("$X$"); ax1.set_ylabel("$Z$"); ax1.set_zlabel("$Y$"); ax1.view_init(elev=15, azim=-60)
        fig.colorbar(p1, ax=ax1, shrink=0.55, pad=0.08)

        ax2 = fig.add_subplot(len(REP_SEQUENCES), 2, 2 * row_idx + 2, projection="3d")
        p2 = ax2.scatter(parent_centers[:, 0], parent_centers[:, 2], parent_centers[:, 1],
                         c=block_theta, cmap="coolwarm", vmin=0, vmax=90, s=42,
                         edgecolor="black", linewidth=0.4)
        ax2.set_title(f"(b) {name}: Inclinacion $\\theta$ ($K=24$, $B_p={Bp}$, $N_p={len(blocks_p)}$)",
                      fontsize=10, fontweight="bold")
        ax2.set_xlabel("$X$"); ax2.set_ylabel("$Z$"); ax2.set_zlabel("$Y$"); ax2.view_init(elev=15, azim=-60)
        fig.colorbar(p2, ax=ax2, shrink=0.55, pad=0.08)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_annex2_spatial_3d_pointcloud_layout.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. FIG A3: K-MEANS CONVERGENCE -- REAL DATA, n=14 (replaces fabricated numbers)
# =============================================================================
def fig_annex3(c):
    print("[5/6] fig_annex3_optimization_cost_evolution.png (real E10 data, n=14)...")
    # Single panel now -- a first fix here (RDO search-cost trace, normalized per-sequence to
    # correct for a ~40x cross-sequence scale mismatch) didn't hold up as a plot either: 14
    # mostly-flat, overlapping lines with a few noisy spikes, no real read. Dropped rather than
    # kept just to have two panels; the k-means convergence panel is the one that's actually
    # informative, so it stands alone.
    fig, ax1 = plt.subplots(figsize=(6.5, 5), dpi=300)
    colors = cm.tab20(np.linspace(0, 1, len(ALL_SEQUENCES)))

    for name, color in zip(ALL_SEQUENCES, colors):
        # kmeans_iterations.K counts DIRECTIONAL centroids only (K_total - 1, mode 0 excluded) --
        # confirmed against the training log ("K-means (K=29 directions)" for a K=30 codebook) and
        # the distinct values actually on disk ({3,7,15,23,29} vs adapter_opt_steps' {4,8,16,24,30}).
        # K=23 directions here == the K=24 codebook everywhere else in this script.
        rows = c.execute("""SELECT ki.iteration, ki.inertia FROM kmeans_iterations ki
                             JOIN sequences se ON se.seq_id=ki.seq_id
                             WHERE se.name=? AND ki.K=23 ORDER BY ki.iteration""", (name,)).fetchall()
        if not rows:
            continue
        ax1.plot([r[0] for r in rows], [r[1] for r in rows], "-", color=color,
                 linewidth=1.3, alpha=0.85, label=name)
    ax1.set_title("Convergencia K-means Esferico ($K=24$, $n=14$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteracion"); ax1.set_ylabel("Inercia (suma de distancias angulares$^2$)")
    ax1.legend(fontsize=6.5, ncol=2); ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_annex3_optimization_cost_evolution.png", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. FIG A4: TRELLIS SWITCHES / CABAC vs Q -- MEAN +/- STD, B_c=16, n=14
# =============================================================================
def fig_annex4(c):
    print("[6/6] fig_annex4_trellis_switches_and_cabac.png (B_c=16, mean+/-std, n=14)...")
    fig, (ax_sw, ax_oh) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    rows = c.execute("""
        SELECT se.name, fr.q_step, fr.K, fr.n_trellis_switches, fr.cabac_overhead_bits
        FROM frame_results fr JOIN sequences se ON se.seq_id=fr.seq_id
        WHERE fr.child_b=16 AND fr.frame_idx=1 ORDER BY fr.K, fr.q_step
    """).fetchall()
    by_k_q = defaultdict(lambda: defaultdict(list))
    oh_by_k_q = defaultdict(lambda: defaultdict(list))
    n_seq_by_k = defaultdict(set)
    for name, Q, K, sw, cabac in rows:
        by_k_q[K][Q].append(sw)
        oh_by_k_q[K][Q].append(cabac / 8.0)  # bits -> bytes
        n_seq_by_k[K].add(name)

    for K in sorted(by_k_q, reverse=True):
        n_seq = len(n_seq_by_k[K])
        if n_seq < MIN_N_FULL:
            print(f"    [skip] K={K}: only {n_seq}/{MIN_N_FULL} sequences complete at B_c=16 "
                  f"-- excluded from the aggregate")
            continue
        sw_mean = [np.mean(by_k_q[K][Q]) for Q in Q_STEPS]; sw_std = [np.std(by_k_q[K][Q]) for Q in Q_STEPS]
        oh_mean = [np.mean(oh_by_k_q[K][Q]) for Q in Q_STEPS]; oh_std = [np.std(oh_by_k_q[K][Q]) for Q in Q_STEPS]
        # Plain mean lines -- same reasoning as fig1: switches range 204-1103 and CABAC bytes
        # 75-529 across sequences at a single (B_c,K,Q) (checked directly), so error bars here
        # would draw the same dominating whiskers. Real spread printed below instead.
        ax_sw.plot(Q_STEPS, sw_mean, marker=MARKERS.get(K, "o"), color=COLORS.get(K, "gray"),
                   linewidth=2.0, markersize=7, label=f"$K={K}$ ($n={n_seq}$)")
        ax_oh.plot(Q_STEPS, oh_mean, marker=MARKERS.get(K, "o"), color=COLORS.get(K, "gray"),
                   linewidth=2.0, markersize=7, label=f"$K={K}$ ($n={n_seq}$)")
        if K == 24:
            print(f"    [sanity] B_c=16 K=24: switches mean={sw_mean} std={sw_std} | "
                  f"cabac_bytes mean={[f'{v:.0f}' for v in oh_mean]} std={[f'{v:.0f}' for v in oh_std]}")

    ax_sw.set_title("(a) Conmutaciones de Modo Trellis Viterbi ($B_c=16$, media, $n=14$)",
                    fontsize=11, fontweight="bold")
    ax_sw.set_xlabel("Paso de Cuantizacion $Q$"); ax_sw.set_ylabel("Numero de Cambios de Modo")
    ax_sw.set_xticks(Q_STEPS); ax_sw.grid(True, linestyle="--", alpha=0.6); ax_sw.legend(fontsize=9)

    ax_oh.set_title("(b) Sobrecarga Fisica CABAC Orden-1 ($B_c=16$, media, $n=14$)",
                    fontsize=11, fontweight="bold")
    ax_oh.set_xlabel("Paso de Cuantizacion $Q$"); ax_oh.set_ylabel("Tamano del Bitstream CABAC (Bytes)")
    ax_oh.set_xticks(Q_STEPS); ax_oh.grid(True, linestyle="--", alpha=0.6); ax_oh.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_annex4_trellis_switches_and_cabac.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c = conn().cursor()
    f1 = load_f1_rows(c)
    fig1(c, f1)
    fig7(c)
    fig_annex1(c)
    fig_annex2(c)
    fig_annex3(c)
    fig_annex4(c)
    print("Listo -- las 6 figuras de docs/report/figures/ fueron regeneradas con datos "
          "multi-secuencia (E10, n=14 salvo donde se indica).")


if __name__ == "__main__":
    main()
