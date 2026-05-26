import json
import sqlite3
import numpy as np
from pathlib import Path

def analyze_sweep(folder_path):
    root = Path(folder_path)
    # Get all result files
    result_files = list(root.glob("result_*.json"))
    
    all_res = []
    
    # Baselines for comparison (we don't have C=1 runs in this sweep, so we'll just report absolute values and internal gains)
    # However, to be useful, I'll try to find the earliest cost (Iter 0) as baseline for EACH run.
    
    for rf in result_files:
        with open(rf, 'r') as f:
            res_meta = json.load(f)
        
        run_id = res_meta['experiment_code']
        db_file = root / f"{run_id}.db"
        
        history = []
        if db_file.exists():
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("""SELECT iteration, total_cost, avg_rate, avg_distortion, cluster_entropy, 
                           hamming_distance_from_previous, self_loop_weights, self_loop_percentages 
                           FROM clustering_history ORDER BY iteration ASC""")
            history = cur.fetchall()
            conn.close()
        
        if not history:
            continue
            
        init = history[0]
        # Find first valid cost if Iter 0 is None
        for row in history:
            if row[1] is not None:
                init = row
                break
        
        final = history[-1]
        
        def parse_json(s):
            try: return np.array(json.loads(s))
            except: return np.array([])

        total_hamming = sum(row[5] for row in history if row[5] is not None)
        slw_final = parse_json(final[6])
        slp_final = parse_json(final[7])
        
        # Parse B, C, Q from run_id
        parts = run_id.split('_')
        b_val = int(parts[0][1:])
        c_val = int(parts[1][1:])
        q_val = int(parts[2][1:])

        all_res.append({
            'B': b_val, 'C': c_val, 'Q': q_val,
            'cost_init': init[1], 'cost_final': final[1],
            'rate_final': final[2], 'dist_final': final[3],
            'ent_final': final[4], 'hamming': total_hamming,
            'iters': len(history) - 1,
            'slw_avg': np.mean(slw_final) if slw_final.size > 0 else 0,
            'slp_avg': np.mean(slp_final) if slp_final.size > 0 else 0,
            'run_id': run_id
        })

    # Sort by B, then Q, then C
    all_res.sort(key=lambda x: (x['B'], x['Q'], x['C']))
    
    print(f"### FAST VALIDATED SWEEP ANALYSIS: {folder_path} ###\n")
    print("| Run ID | Final RD Cost | Internal Δ % | Rate (R) | Dist (D) | Entropy | Ham Σ | Iters | Avg SLW | Avg SLP |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    
    for r in all_res:
        delta = (r['cost_final'] - r['cost_init']) / r['cost_init'] * 100 if r['cost_init'] else 0
        print(f"| {r['run_id']:12s} | {r['cost_final']:12.2f} | {delta:+10.2f}% | {r['rate_final']:8.4f} | {r['dist_final']:8.4f} | {r['ent_final']:7.4f} | {r['hamming']:5d} | {r['iters']:5d} | {r['slw_avg']:7.3f} | {r['slp_avg']:7.3f} |")

if __name__ == "__main__":
    analyze_sweep("fast_validated_sweep_20260524_235753")
