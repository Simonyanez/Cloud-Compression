import json
import sqlite3
import numpy as np
from pathlib import Path
import os

def get_latest_results():
    results_root = Path("results")
    bsize = 4
    all_res = {} # (C, Q) -> data
    
    # Iterate through all timestamped folders to gather ALL results
    run_folders = sorted([d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("20260524")], reverse=True)
    
    for run in run_folders:
        vanish_dirs = [d for d in run.iterdir() if d.is_dir() and d.name.startswith(f"VANISH_B{bsize}")]
        for vdir in vanish_dirs:
            # Parse C and Q from name: VANISH_B4_C8_Q12
            parts = vdir.name.split('_')
            try:
                c_val = int(parts[2][1:])
                q_val = int(parts[3][1:])
            except:
                continue
                
            # If we already have this (C, Q) from a newer run, skip
            if (c_val, q_val) in all_res:
                continue
                
            json_file = list(vdir.glob("test_result_*.json"))
            if not json_file:
                continue
                
            with open(json_file[0], 'r') as f:
                data = json.load(f)
            
            db_file = list(vdir.glob("*.db"))
            history_data = []
            if db_file:
                conn = sqlite3.connect(db_file[0])
                cur = conn.cursor()
                cur.execute("""SELECT iteration, total_cost, avg_rate, avg_distortion, cluster_entropy, 
                               hamming_distance_from_previous, self_loop_weights, self_loop_percentages, slopes 
                               FROM clustering_history ORDER BY iteration ASC""")
                history_data = cur.fetchall()
                conn.close()
            
            if not history_data:
                continue

            # IDENTIFY OPTIMIZATION BASELINE (First row with valid cost)
            init_opt_row = None
            for row in history_data:
                if row[1] is not None:
                    init_opt_row = row
                    break
            
            # If no row has a cost (C=1), use the only row available
            if init_opt_row is None:
                init_opt_row = history_data[0]

            final_row = history_data[-1]
            
            def parse_json(s):
                try: return np.array(json.loads(s))
                except: return np.array([])

            slw_final = parse_json(final_row[6])
            slp_final = parse_json(final_row[7])
            slopes_final = parse_json(final_row[8])
            
            all_res[(c_val, q_val)] = {
                'cost_init': init_opt_row[1],
                'cost_final': final_row[1],
                'rate_init': init_opt_row[2],
                'rate_final': final_row[2],
                'dist_init': init_opt_row[3],
                'dist_final': final_row[3],
                'ent_init': init_opt_row[4],
                'ent_final': final_row[4],
                'active': data['active_clusters'],
                'iters': len(history_data) - 1 if len(history_data) > 1 else 1,
                'total_hamming': sum(row[5] for row in history_data if row[5] is not None),
                'slw_stats': (np.mean(slw_final), np.std(slw_final), np.min(slw_final), np.max(slw_final)) if slw_final.size > 0 else (0,0,0,0),
                'slp_stats': (np.mean(slp_final), np.std(slp_final), np.min(slp_final), np.max(slp_final)) if slp_final.size > 0 else (0,0,0,0),
                'slope_norm': np.mean([np.linalg.norm(s) for s in slopes_final]) if slopes_final.size > 0 else 0
            }
            
    return all_res

def main():
    results = get_latest_results()
    if not results:
        print("No results found.")
        return
        
    c_vals = sorted(list(set(k[0] for k in results.keys())))
    q_vals = sorted(list(set(k[1] for k in results.keys())))
    
    print("### RAW NUMERICAL DATASET: RD-CLUSTERING (B=4, N=135) ###\n")
    
    for q_idx, q in enumerate(q_vals):
        letter = chr(97 + q_idx) # a, b, c...
        print(f"--- [ QUANTIZATION STEP Q = {q} ] ---")
        baseline = results.get((1, q))
        b_cost = baseline['cost_final'] if baseline else 1.0
        
        print(f"\n[ Table 1{letter}: End-State Performance & Coding Gain ]")
        print("| C | Final Cost     | Rate (R) | Dist (D) | Entropy (H) | Active | Gain %  | Eff % (Gain/H) |")
        print("|---|----------------|----------|----------|-------------|--------|---------|----------------|")
        for c in c_vals:
            res = results.get((c, q))
            if not res: continue
            gain = (res['cost_final'] - b_cost) / b_cost * 100
            eff = gain / res['ent_final'] if res['ent_final'] > 0.01 else 0.0
            print(f"| {c:1d} | {res['cost_final']:14.4f} | {res['rate_final']:8.4f} | {res['dist_final']:8.4f} | {res['ent_final']:11.4f} | {res['active']:1d}/{c:1d} | {gain:+7.4f}% | {eff:+14.4f} |")

        print(f"\n[ Table 2{letter}: Optimization Progress (First Assignment -> Final) ]")
        print("| C | Cost Δ % | Rate Δ % | Dist Δ % | H Δ     | Hamming Σ | Opt Iters | Avg Ham/Iter |")
        print("|---|----------|----------|----------|---------|-----------|-----------|--------------|")
        for c in c_vals:
            res = results.get((c, q))
            if not res: continue
            
            def safe_delta_pct(f, i):
                if f is not None and i is not None and i != 0: return (f - i) / i * 100
                return 0.0
            
            def safe_sub(f, i):
                if f is not None and i is not None: return f - i
                return 0.0

            c_delta = safe_delta_pct(res['cost_final'], res['cost_init'])
            r_delta = safe_delta_pct(res['rate_final'], res['rate_init'])
            d_delta = safe_delta_pct(res['dist_final'], res['dist_init'])
            h_delta = safe_sub(res['ent_final'], res['ent_init'])
            avg_ham = res['total_hamming'] / max(1, res['iters'])
            print(f"| {c:1d} | {c_delta:+8.4f}% | {r_delta:+8.4f}% | {d_delta:+8.4f}% | {h_delta:+7.4f} | {res['total_hamming']:9d} | {res['iters']:9d} | {avg_ham:12.4f} |")

        print(f"\n[ Table 3{letter}: Parameter Distribution Statistics (Final) ]")
        print("| C | Avg SLW | Std SLW | SLW [Min, Max] | Avg SLP | Std SLP | SLP [Min, Max] | Slope Norm |")
        print("|---|---------|---------|----------------|---------|---------|----------------|------------|")
        for c in c_vals:
            res = results.get((c, q))
            if not res: continue
            w = res['slw_stats']
            p = res['slp_stats']
            print(f"| {c:1d} | {w[0]:7.4f} | {w[1]:7.4f} | [{w[2]:4.2f}, {w[3]:4.2f}] | {p[0]:7.4f} | {p[1]:7.4f} | [{p[2]:4.2f}, {p[3]:4.2f}] | {res['slope_norm']:10.4f} |")
        print("\n" + "="*120 + "\n")

if __name__ == "__main__":
    main()
