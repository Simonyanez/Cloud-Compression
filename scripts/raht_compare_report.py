#!/usr/bin/env python3
"""
Comparison report: RAHT baseline (results/raht_baseline.db, this branch) vs. PCADC-adaptive
(K=24) vs. PCADC-baseline (K=0 structural), read from results/exp_e10_multisequence.db in
READ-ONLY mode (never opened for writing -- enforced here via a mode=ro URI connection, not
just convention).

DISCLOSED APPROXIMATIONS (restated on every run, not just in source comments):
  1. RAHT here is BLOCK-LOCAL (see src/pcadc/raht.py module docstring) -- rooted at each
     PCADC block's own corner, not full-cloud as in the real G-PCC/TMC13 reference software.
     Expect this to understate true RAHT's rate-distortion performance.
  2. Both RAHT and PCADC use the same PyRLGR entropy backend in this repo (a deliberate
     choice, for a fair PCADC-vs-RAHT comparison *within this codebase*) -- neither side's
     absolute bpv should be read as directly comparable to published TMC13 numbers, which use
     a different entropy coder entirely.

This is a 3-sequence sanity/feasibility comparison, not the full 14-sequence sweep the thesis
originally deferred.
"""
import sys
import sqlite3
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_e10_multisequence_benchmark import bdrate

RAHT_DB = REPO_ROOT / "results" / "raht_baseline.db"
PCADC_DB = REPO_ROOT / "results" / "exp_e10_multisequence.db"
OUT_JSON = REPO_ROOT / "results" / "raht_baseline_comparison.json"
OUT_PNG = REPO_ROOT / "results" / "raht_baseline_comparison.png"

B_C = 16
K_ADAPTIVE = 24
SEQUENCES = ["longdress", "phil9", "phil10"]

DISCLOSURE = (
    "RAHT is block-local (rooted per PCADC block, not full-cloud as in real G-PCC/TMC13) -- "
    "expect this to understate true RAHT performance. Both RAHT and PCADC use the same PyRLGR "
    "entropy backend here, for a fair within-repo comparison only; absolute bpv is not directly "
    "comparable to published TMC13 numbers. 3-sequence sanity test, not the full 14-sequence sweep."
)


def read_raht(seq_name):
    with sqlite3.connect(str(RAHT_DB)) as conn:
        rows = conn.execute("""
            SELECT r.q_step, r.aggregate_bpv, r.aggregate_psnr_Y
            FROM aggregate_results_raht r JOIN sequences_raht s ON s.seq_id = r.seq_id
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


def main():
    report = {"disclosure": DISCLOSURE, "b_c": B_C, "k_adaptive": K_ADAPTIVE, "sequences": {}}

    for seq in SEQUENCES:
        raht = read_raht(seq)
        pcadc_adaptive, pcadc_baseline = read_pcadc(seq)
        if not raht or not pcadc_adaptive:
            print(f"[WARN] missing data for {seq}, skipping")
            continue

        bpv_raht = [d["bpv"] for d in raht]
        psnr_raht = [d["psnr"] for d in raht]
        bpv_base = [d["bpv"] for d in pcadc_baseline]
        psnr_base = [d["psnr"] for d in pcadc_baseline]
        bpv_adapt = [d["bpv"] for d in pcadc_adaptive]
        psnr_adapt = [d["psnr"] for d in pcadc_adaptive]

        bd_raht_vs_base = bdrate(bpv_base, psnr_base, bpv_raht, psnr_raht)
        bd_raht_vs_adapt = bdrate(bpv_adapt, psnr_adapt, bpv_raht, psnr_raht)

        report["sequences"][seq] = {
            "raht": raht,
            "pcadc_adaptive_K24": pcadc_adaptive,
            "pcadc_baseline_K0": pcadc_baseline,
            "bd_rate_raht_vs_pcadc_baseline_pct": bd_raht_vs_base,
            "bd_rate_raht_vs_pcadc_adaptive_pct": bd_raht_vs_adapt,
        }
        print(f"{seq:>12s} | BD-Rate RAHT vs PCADC-baseline(K=0): {bd_raht_vs_base:+7.3f}% | "
              f"BD-Rate RAHT vs PCADC-adaptive(K=24): {bd_raht_vs_adapt:+7.3f}%")

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
        ax.plot([p["bpv"] for p in d["raht"]], [p["psnr"] for p in d["raht"]],
                "^-", label="RAHT (block-local)", color="tab:red")
        ax.set_title(f"{seq}, $B_c$={B_C}")
        ax.set_xlabel("bpv")
        ax.set_ylabel("PSNR$_Y$ (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("PCADC vs. block-local RAHT baseline -- sanity subset, real RLGR coding both sides\n"
                 + DISCLOSURE, fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(str(OUT_PNG), dpi=200)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
