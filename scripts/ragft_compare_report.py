#!/usr/bin/env python3
"""
Comparison report: RA-GFT baseline (results/ragft_baseline.db, this branch) vs. PCADC-adaptive
(K=24) vs. PCADC-baseline (K=0 structural), read from results/exp_e10_multisequence.db in
READ-ONLY mode (never opened for writing -- enforced here via a mode=ro URI connection, not
just convention).

Unlike the RAHT comparison, this is the real, meaningful external comparison: RA-GFT is a
genuine multiresolution graph transform, already a stronger baseline by design (and PCADC's own
graph/color code is a direct descendant of the RA-GFT reference implementation -- see
docs/thesis/ra_gft_compatibility_note.md), not an easy win either direction the way RAHT was.

DISCLOSED APPROXIMATIONS (restated on every run, not just in source comments):
  1. RA-GFT here is BLOCK-LOCAL (see src/pcadc/ragft.py module docstring and
     scripts/exp_ragft_baseline.py) -- applied within each PCADC B_c=16 block via a genuine
     4-level cascade (bsize_per_level=[2,2,2,2]), not full-cloud multiresolution as the
     original paper's own results demonstrate (which cascades across the WHOLE point cloud,
     capturing cross-block redundancy this block-local version cannot). Expect this to
     understate full-cloud RA-GFT's true ceiling -- same direction of caveat as the RAHT
     baseline, for the same structural reason.
  2. Both RA-GFT and PCADC use the same PyRLGR entropy backend in this repo (a deliberate
     choice, for a fair PCADC-vs-RA-GFT comparison *within this codebase*) -- neither side's
     absolute bpv should be read as directly comparable to any published reference numbers,
     which use different entropy coders.

This is a 3-sequence sanity/feasibility comparison, not a full 14-sequence sweep.
"""
import sys
import sqlite3
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_e10_multisequence_benchmark import bdrate

RAGFT_DB = REPO_ROOT / "results" / "ragft_baseline.db"
PCADC_DB = REPO_ROOT / "results" / "exp_e10_multisequence.db"
OUT_JSON = REPO_ROOT / "results" / "ragft_baseline_comparison.json"
OUT_PNG = REPO_ROOT / "results" / "ragft_baseline_comparison.png"

B_C = 16
K_ADAPTIVE = 24
SEQUENCES = ["longdress", "phil9", "phil10"]

DISCLOSURE = (
    "RA-GFT is block-local (a genuine 4-level cascade within each PCADC block, not full-cloud "
    "multiresolution as the original paper demonstrates) -- expect this to understate true "
    "RA-GFT performance. Both RA-GFT and PCADC use the same PyRLGR entropy backend here, for a "
    "fair within-repo comparison only. 3-sequence sanity test, not a full sweep."
)


def read_ragft(seq_name):
    with sqlite3.connect(str(RAGFT_DB)) as conn:
        rows = conn.execute("""
            SELECT r.q_step, r.aggregate_bpv, r.aggregate_psnr_Y
            FROM aggregate_results_ragft r JOIN sequences_ragft s ON s.seq_id = r.seq_id
            WHERE s.name=? AND r.child_b=?
            ORDER BY r.q_step
        """, (seq_name, B_C)).fetchall()
    return [{"q": q, "bpv": bpv, "psnr": psnr} for q, bpv, psnr in rows]


def read_pcadc(seq_name):
    uri = f"file:{PCADC_DB}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute("""
            SELECT a.q_step, a.aggregate_bpv, a.aggregate_psnr_Y, a.baseline_bpv, a.baseline_psnr_Y
            FROM aggregate_results a JOIN sequences s ON s.seq_id = a.seq_id
            WHERE s.name=? AND a.child_b=? AND a.K=?
            ORDER BY a.q_step
        """, (seq_name, B_C, K_ADAPTIVE)).fetchall()
    adaptive = [{"q": q, "bpv": abpv, "psnr": apsnr} for q, abpv, apsnr, _, _ in rows]
    baseline = [{"q": q, "bpv": bbpv, "psnr": bpsnr} for q, _, _, bbpv, bpsnr in rows]
    return adaptive, baseline


def read_raht(seq_name):
    """Also pull the already-computed RAHT numbers (previous branch's DB, if present) so the
    report can place RA-GFT alongside them -- both baselines on the same sanity subset."""
    raht_db = REPO_ROOT / "results" / "raht_baseline.db"
    if not raht_db.exists():
        return []
    with sqlite3.connect(str(raht_db)) as conn:
        rows = conn.execute("""
            SELECT r.q_step, r.aggregate_bpv, r.aggregate_psnr_Y
            FROM aggregate_results_raht r JOIN sequences_raht s ON s.seq_id = r.seq_id
            WHERE s.name=? AND r.child_b=?
            ORDER BY r.q_step
        """, (seq_name, B_C)).fetchall()
    return [{"q": q, "bpv": bpv, "psnr": psnr} for q, bpv, psnr in rows]


def main():
    report = {"disclosure": DISCLOSURE, "b_c": B_C, "k_adaptive": K_ADAPTIVE, "sequences": {}}

    for seq in SEQUENCES:
        ragft = read_ragft(seq)
        pcadc_adaptive, pcadc_baseline = read_pcadc(seq)
        raht = read_raht(seq)
        if not ragft or not pcadc_adaptive:
            print(f"[WARN] missing data for {seq}, skipping")
            continue

        bpv_ragft = [d["bpv"] for d in ragft]
        psnr_ragft = [d["psnr"] for d in ragft]
        bpv_base = [d["bpv"] for d in pcadc_baseline]
        psnr_base = [d["psnr"] for d in pcadc_baseline]
        bpv_adapt = [d["bpv"] for d in pcadc_adaptive]
        psnr_adapt = [d["psnr"] for d in pcadc_adaptive]

        bd_ragft_vs_base = bdrate(bpv_base, psnr_base, bpv_ragft, psnr_ragft)
        bd_ragft_vs_adapt = bdrate(bpv_adapt, psnr_adapt, bpv_ragft, psnr_ragft)
        bd_ragft_vs_raht = (
            bdrate([d["bpv"] for d in raht], [d["psnr"] for d in raht], bpv_ragft, psnr_ragft)
            if raht else float("nan")
        )

        report["sequences"][seq] = {
            "ragft": ragft,
            "raht": raht,
            "pcadc_adaptive_K24": pcadc_adaptive,
            "pcadc_baseline_K0": pcadc_baseline,
            "bd_rate_ragft_vs_pcadc_baseline_pct": bd_ragft_vs_base,
            "bd_rate_ragft_vs_pcadc_adaptive_pct": bd_ragft_vs_adapt,
            "bd_rate_ragft_vs_raht_pct": bd_ragft_vs_raht,
        }
        raht_str = f" | vs RAHT: {bd_ragft_vs_raht:+7.3f}%" if raht else ""
        print(f"{seq:>12s} | BD-Rate RA-GFT vs PCADC-baseline(K=0): {bd_ragft_vs_base:+7.3f}% | "
              f"vs PCADC-adaptive(K=24): {bd_ragft_vs_adapt:+7.3f}%{raht_str}")

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # --- figure: 3-panel RD curves, one subplot per sequence ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(report["sequences"])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, (seq, d) in zip(axes, report["sequences"].items()):
        ax.plot([p["bpv"] for p in d["pcadc_adaptive_K24"]], [p["psnr"] for p in d["pcadc_adaptive_K24"]],
                "o-", label="PCADC adaptive (K=24)", color="tab:green")
        ax.plot([p["bpv"] for p in d["pcadc_baseline_K0"]], [p["psnr"] for p in d["pcadc_baseline_K0"]],
                "s--", label="PCADC baseline (K=0)", color="tab:gray")
        ax.plot([p["bpv"] for p in d["ragft"]], [p["psnr"] for p in d["ragft"]],
                "D-", label="RA-GFT (block-local)", color="tab:blue")
        if d["raht"]:
            ax.plot([p["bpv"] for p in d["raht"]], [p["psnr"] for p in d["raht"]],
                    "^:", label="RAHT (block-local)", color="tab:red", alpha=0.7)
        ax.set_title(f"{seq}, $B_c$={B_C}")
        ax.set_xlabel("bpv")
        ax.set_ylabel("PSNR$_Y$ (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("PCADC vs. block-local RA-GFT baseline -- sanity subset, real RLGR coding all sides\n"
                 + DISCLOSURE, fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(str(OUT_PNG), dpi=200)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
