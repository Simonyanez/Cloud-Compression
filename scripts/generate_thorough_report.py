import json
import numpy as np
from pathlib import Path
import sqlite3

def analyze():
    root = Path("sweep_results_20260524_210626")
    runs = ["B4_C4_Q24", "B8_C4_Q24", "B8_C8_Q24", "B16_C4_Q24"]
    baselines = {"B4": 72816.5995, "B8": 49682.9118, "B16": 90174.2800}
    
    report = []
    report.append("# DETAILED SENSITIVITY ANALYSIS DATASET (B4/B8/B16)")
    report.append(f"Timestamp: 2026-05-24 | Samples: 0.1% | Q-Step: 24\n")

    # 1. Executive Table
    report.append("## SECTION 1: GLOBAL PERFORMANCE MATRIX")
    report.append("| Run ID | Baseline Cost | Final Cost | Gain % | Iters | Active Clusters | Hamming Total |")
    report.append("|---|---|---|---|---|---|---|")
    
    for run in runs:
        res_file = root / f"result_{run}.json"
        if not res_file.exists(): continue
        with open(res_file, 'r') as f:
            res = json.load(f)
        
        b_key = run.split('_')[0]
        base = baselines.get(b_key, 1.0)
        gain = (res['final_cost'] - base) / base * 100
        
        # Calculate Total Hamming from iterations
        iter_files = sorted(list(root.glob(f"{run}_iter_*.json")))
        total_hamming = 0
        prev_labels = None
        for it_f in iter_files:
            with open(it_f, 'r') as f:
                d = json.load(f)
                if prev_labels is not None:
                    total_hamming += np.sum(np.array(prev_labels) != np.array(d['labels']))
                prev_labels = d['labels']

        report.append(f"| {run} | {base:,.2f} | {res['final_cost']:,.2f} | {gain:+.2f}% | {res['iterations']} | {res['active_clusters']}/{res['total_clusters']} | {total_hamming} |")

    # 2. Convergence Logs
    report.append("\n## SECTION 2: CONVERGENCE & HAMMING EVOLUTION")
    for run in runs:
        report.append(f"### Run: {run}")
        report.append("| Iter | RD Cost | Entropy | Rate | Dist | Hamming Δ |")
        report.append("|---|---|---|---|---|---|")
        
        iter_files = sorted(list(root.glob(f"{run}_iter_*.json")))
        prev_labels = None
        for it_f in iter_files:
            with open(it_f, 'r') as f:
                d = json.load(f)
            
            it_num = d.get('iteration', 0)
            cost = d.get('total_cost')
            h = d.get('cluster_entropy')
            r = d.get('avg_rate')
            dist = d.get('avg_distortion')
            
            h_str = f"{h:.4f}" if h is not None else "N/A"
            r_str = f"{r:.4f}" if r is not None else "N/A"
            d_str = f"{dist:.4f}" if dist is not None else "N/A"
            
            ham = 0
            if prev_labels is not None:
                ham = np.sum(np.array(prev_labels) != np.array(d['labels']))
            prev_labels = d['labels']
            
            cost_str = f"{cost:,.2f}" if cost else "N/A"
            report.append(f"| {it_num} | {cost_str} | {h_str} | {r_str} | {d_str} | {ham} |")
        report.append("\n")

    # 3. Parameter Distributions
    report.append("## SECTION 3: FINAL PARAMETER RANGES")
    report.append("| Run ID | SLW (Avg ± Std) | SLW [Min, Max] | SLP (Avg ± Std) | SLP [Min, Max] | Slope Norm (Avg) |")
    report.append("|---|---|---|---|---|---|")
    
    for run in runs:
        res_file = root / f"result_{run}.json"
        if not res_file.exists(): continue
        with open(res_file, 'r') as f:
            res = json.load(f)
        
        slw = np.array(res['final_slw'])
        slp = np.array(res['final_slp'])
        slopes = np.array(res['final_slopes'])
        
        # Exclude index 0 (Structural DC) for pure adaptive stats
        if len(slw) > 1:
            slw_a = slw[1:]
            slp_a = slp[1:]
            slopes_a = slopes[1:]
        else:
            slw_a, slp_a, slopes_a = slw, slp, slopes

        snorms = [np.linalg.norm(s) for s in slopes_a]
        
        report.append(f"| {run} | {np.mean(slw_a):.3f} ± {np.std(slw_a):.3f} | [{np.min(slw_a):.2f}, {np.max(slw_a):.2f}] | {np.mean(slp_a):.3f} ± {np.std(slp_a):.3f} | [{np.min(slp_a):.2f}, {np.max(slp_a):.2f}] | {np.mean(snorms):.3f} |")

    # 4. Cluster Stability / Vanishing
    report.append("\n## SECTION 4: STABILITY ANALYSIS")
    for run in runs:
        res_file = root / f"result_{run}.json"
        if not res_file.exists(): continue
        with open(res_file, 'r') as f:
            res = json.load(f)
        
        if res['is_vanishing']:
            report.append(f"- **{run}**: CRITICAL VANISHING. Active: {res['active_clusters']}/{res['total_clusters']}. Empty Indices: {res['empty_clusters']}")
        else:
            report.append(f"- **{run}**: STABLE. All clusters active.")

    with open("THOROUGH_EXPERIMENT_REPORT.md", "w") as f:
        f.write("\n".join(report))
    print("[+] Report generated: THOROUGH_EXPERIMENT_REPORT.md")

if __name__ == "__main__":
    analyze()
