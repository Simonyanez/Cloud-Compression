#!/usr/bin/env python3
"""
Regenerates the Oracle-experiment figures for docs/thesis/resultados.tex (§Análisis de Límites
Teóricos).

- item 2 (still true, kept): confirms via direct query that the oracle experiment used the FULL
  longdress frame (765,821 points, not sampled) and that its "Rate" axis is a nonzero-coefficient
  PROXY (Decider._RDcost's sparsity term), not real RLGR-encoded bits.
- Real-RLGR RD curve (Linear Oracle, no CABAC/signaling overhead): reads
  scripts/_oracle_real_rlgr.json, produced by scripts/oracle_real_rlgr_and_compaction.py, which
  recomputes real per-block GFT coefficients at the oracle's already-chosen (best_p, best_w) and
  RLGR-encodes them for real. Overlaid against the proxy curve computed directly from
  linear_oracle_upper_bound.db, on the same (bsize, q_step) grid, to make the real-vs-proxy gap a
  concrete, plotted number instead of an assertion.
- Sensitivity analysis (replaces the old, uninformative "Figura 5.2" gain-by-blocksize histogram):
  reads scripts/_oracle_compaction_records.json (Linear Oracle, Q=32, real spectral compaction
  computed alongside the RLGR pass above) and correlates block characteristics (Y_std, planar-fit
  residual, gradient magnitude, point count) against compaction/gain/best_p/best_w, per block
  size -- which characteristics actually drive the ideal-plane case's performance.

Data sources (read-only): results/oracle_upper_bound.db, results/linear_oracle_upper_bound.db
(populated by absolute_upper_bound_full.py / linear_oracle_upper_bound_full.py -- this script
never re-runs the (p,w) search itself), plus the two JSON files named above.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "thesis" / "img" / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                      "legend.fontsize": 7.5, "figure.dpi": 300})


class Cols:
    """Minimal column-store standing in for a pandas DataFrame (pandas isn't installed in this
    venv) -- just enough dict-of-arrays + boolean masking for what this script needs."""
    def __init__(self, names, rows):
        arr = np.array(rows, dtype=object)
        self._d = {n: np.array(arr[:, i].tolist()) for i, n in enumerate(names)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._d[key]
        mask = key  # boolean array -> filtered copy
        rows = list(zip(*[self._d[n][mask] for n in self._d]))
        return Cols(list(self._d.keys()), rows)

    def col_eq(self, name, value):
        return self._d[name] == value

    def unique(self, name):
        return sorted(set(self._d[name].tolist()))


def load(db_name):
    conn = sqlite3.connect(f"file:{REPO_ROOT / 'results' / db_name}?mode=ro", uri=True)
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(results)")]
    rows = cur.execute(f"SELECT {','.join(cols)} FROM results").fetchall()
    conn.close()
    return Cols(cols, rows)


def part1_real_rd():
    """Real-RLGR RD curves for BOTH oracles vs. the real structural (p=0) baseline -- all
    computed by the same real encode+RLGR pass (scripts/oracle_real_rlgr_and_compaction.py, run
    once per oracle), no proxy anywhere in this figure. The baseline (p=0, no self-loops) is
    identical code for both oracles and numerically identical in the data (confirmed bit-for-bit
    equal across both JSON outputs), so it's plotted once per panel, not duplicated. Also
    computes real BD-Rate/BD-PSNR per block size for both oracles, reusing the project's own
    Bjontegaard implementation."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from calculate_bjontegaard_metrics import bjontegaard_metric  # noqa: E402  (exact reuse)

    def load_by_bs(fname):
        real = json.load(open(REPO_ROOT / "scripts" / fname))
        by_bs = {}
        for r in real:
            by_bs.setdefault(r["bsize"], []).append(r)
        for bs in by_bs:
            by_bs[bs].sort(key=lambda r: r["q_step"])
        return by_bs

    oracles = [("Linear Oracle", "_oracle_real_rlgr.json", "#C0392B", "o"),
               ("Absolute Oracle", "_oracle_real_rlgr_absolute.json", "#2E86AB", "^")]
    data = {name: load_by_bs(fname) for name, fname, _, _ in oracles}
    block_sizes = sorted(set().union(*[d.keys() for d in data.values()]))

    fig, axes = plt.subplots(1, len(block_sizes), figsize=(4.6 * len(block_sizes), 4.6),
                              sharey=True)
    if len(block_sizes) == 1:
        axes = [axes]

    bd_report = {}  # {oracle_name: {bsize: (bd_rate, bd_psnr)}}
    for name, _, color, marker in oracles:
        bd_report[name] = {}

    for pi, (ax, bs) in enumerate(zip(axes, block_sizes)):
        # Baseline: identical for both oracles, take it from whichever has this bsize.
        ref_pts = next(d[bs] for d in data.values() if bs in d)
        bbpv = np.array([p["base_bpv"] for p in ref_pts])
        bpsnr = np.array([p["base_psnr_y"] for p in ref_pts])
        ax.plot(bbpv, bpsnr, "s--", color="gray", linewidth=1.3, markersize=4, alpha=0.7,
                label="Baseline estructural (real)")

        for name, _, color, marker in oracles:
            pts = data[name].get(bs)
            if pts is None:
                continue
            obpv = np.array([p["oracle_bpv"] for p in pts])
            opsnr = np.array([p["oracle_psnr_y"] for p in pts])
            ax.plot(obpv, opsnr, marker + "-", color=color, linewidth=1.8, markersize=5,
                    label=f"{name} (real)")
            bd_rate = bjontegaard_metric(bbpv, bpsnr, obpv, opsnr, mode="rate")
            bd_psnr = bjontegaard_metric(bbpv, bpsnr, obpv, opsnr, mode="psnr")
            bd_report[name][bs] = (bd_rate, bd_psnr)

        ax.set_title(f"$B_c={bs}$", fontsize=10)
        ax.set_xlabel("bpv")
        if pi == 0:
            ax.set_ylabel("PSNR$_Y$ (dB)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "oracle_real_rd.png", dpi=300)
    plt.close(fig)
    print("  [+] oracle_real_rd.png")

    print("\nReal BD-Rate/BD-PSNR vs. real structural baseline, both oracles:")
    for name, _, _, _ in oracles:
        for bs in block_sizes:
            if bs in bd_report[name]:
                bd_rate, bd_psnr = bd_report[name][bs]
                print(f"  {name:16s} B_c={bs:2d}: BD-Rate={bd_rate:+.2f}%  BD-PSNR={bd_psnr:+.3f}dB")
    return bd_report


def part2_sensitivity():
    """Sensitivity analysis replacing the old gain-by-blocksize histogram: real per-block
    PERFORMANCE (rows: bpv, PSNR, spectral compaction, coefficient sparsity -- all real, not
    proxy) against candidate DRIVERS (columns: Y_std, planar-fit residual, point density,
    geometric planarity), at a fixed representative Q (=32, matching the
    sec:e10_info_diagnostics convention), split by block size so the already-established
    block-size effect doesn't swamp the others."""
    records_path = REPO_ROOT / "scripts" / "_oracle_compaction_records.json"
    recs = json.load(open(records_path))
    by_bs = {}
    for r in recs:
        by_bs.setdefault(r["bsize"], []).append(r)

    # Rows = performance (what we care about); columns = drivers (what might explain it) --
    # matched to how the reader will actually ask the question ("does X predict performance Y").
    performance = [("bpv", lambda r: r["real_bpv"]),
                   ("PSNR", lambda r: r["psnr_y"]),
                   ("compaction", lambda r: r["compaction"]),
                   ("sparsity", lambda r: r["sparsity"])]
    drivers = [("Y_std", lambda r: r["y_var"] ** 0.5),
               ("fit_rmse", lambda r: r["fit_rmse"]),
               ("point_count", lambda r: r["point_count"]),
               ("planarity", lambda r: r["planarity"])]

    block_sizes = sorted(by_bs.keys())
    fig, axes = plt.subplots(1, len(block_sizes), figsize=(4.3 * len(block_sizes), 4.6),
                              sharey=True)
    if len(block_sizes) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.12)

    im = None
    for pi, (ax, bs) in enumerate(zip(axes, block_sizes)):
        rs = [r for r in by_bs[bs] if r["psnr_y"] is not None]
        P = np.array([[f(r) for r in rs] for _, f in performance])
        D = np.array([[f(r) for r in rs] for _, f in drivers])
        sens = np.zeros((len(performance), len(drivers)))
        for i in range(len(performance)):
            for j in range(len(drivers)):
                p, d = P[i], D[j]
                if np.std(p) > 1e-12 and np.std(d) > 1e-12:
                    sens[i, j] = np.corrcoef(p, d)[0, 1]
        im = ax.imshow(sens, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(drivers)))
        ax.set_xticklabels([d[0] for d in drivers], rotation=30, ha="right")
        ax.set_yticks(range(len(performance)))
        if pi == 0:
            ax.set_yticklabels([p[0] for p in performance])
        else:
            ax.tick_params(axis="y", left=False)
        ax.set_title(f"$B_c={bs}$ (n={len(rs)})", fontsize=10)
        for i in range(len(performance)):
            for j in range(len(drivers)):
                v = sens[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 else "black")
        print(f"\nSensitivity B_c={bs} (n={len(rs)}):")
        for i, (pname, _) in enumerate(performance):
            for j, (dname, _) in enumerate(drivers):
                print(f"  {pname:12s} vs {dname:12s}: r={sens[i, j]:+.3f}")

    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson $r$")
    fig.savefig(OUT_DIR / "oracle_sensitivity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n  [+] oracle_sensitivity_matrix.png")


def main():
    abs_df = load("oracle_upper_bound.db")
    lin_df = load("linear_oracle_upper_bound.db")

    # --- item 2: confirm full-cloud + proxy-rate facts the thesis text will cite ---
    print(f"Sample block_id: {abs_df['block_id'][0]}")
    min_q = min(abs_df["q_step"])
    for bs in abs_df.unique("bsize"):
        mask = abs_df.col_eq("bsize", bs) & abs_df.col_eq("q_step", min_q)
        n_pts = int(sum(abs_df["point_count"][mask]))
        print(f"  bsize={bs}: total points at min q_step = {n_pts}")
    print("(765,821 points at every block size == the FULL longdress frame 1051, not sampled)")

    part1_real_rd()
    part2_sensitivity()

    # --- item 3b (still Figura 5.3, unchanged): 2D best_p vs best_w distribution ---
    # bins="log": the (p=0, w=0) "no gain / near-flat block" cell dominates linearly (>50k of
    # ~92k blocks) and would otherwise wash out everything else on a linear color scale.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, df, title, cmap in [(axes[0], abs_df, "Absolute Oracle", "Blues"),
                                 (axes[1], lin_df, "Linear Oracle", "Reds")]:
        hb = ax.hexbin(df["best_p"].astype(float), df["best_w"].astype(float), gridsize=30,
                        cmap=cmap, mincnt=1, bins="log")
        ax.set_title(title)
        ax.set_xlabel("$p$ (fracción de nodos seleccionados)")
        ax.set_ylabel("$w$ (peso de auto-lazo)")
        fig.colorbar(hb, ax=ax, shrink=0.8, label="bloques")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "oracle_p_w_distribution.png", dpi=300)
    plt.close(fig)
    print("  [+] oracle_p_w_distribution.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
