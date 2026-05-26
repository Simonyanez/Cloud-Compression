import json
import sqlite3
import numpy as np
from pathlib import Path
import os

def get_results_from_folders(folder_paths):
    all_res = {} # (B, C, Q) -> data
    
    for folder in folder_paths:
        root = Path(folder)
        if not root.exists():
            continue
            
        # Find all result_*.json files directly in the root of the folder
        result_files = list(root.glob("result_*.json"))
        for rf in result_files:
            # result_B4_C2_Q24.json
            parts = rf.stem.split('_')
            try:
                b_val = int(parts[1][1:])
                c_val = int(parts[2][1:])
                q_val = int(parts[3][1:])
            except:
                continue
                
            with open(rf, 'r') as f:
                data = json.load(f)
            
            run_id = f"B{b_val}_C{c_val}_Q{q_val}"
            db_file = root / f"{run_id}.db"
            
            history_data = []
            if db_file.exists():
                conn = sqlite3.connect(db_file)
                cur = conn.cursor()
                cur.execute("""SELECT iteration, total_cost, avg_rate, avg_distortion, cluster_entropy, 
                               hamming_distance_from_previous, self_loop_weights, self_loop_percentages, slopes 
                               FROM clustering_history ORDER BY iteration ASC""")
                history_data = cur.fetchall()
                conn.close()
            
            if not history_data:
                continue

            init_opt_row = None
            for row in history_data:
                if row[1] is not None:
                    init_opt_row = row
                    break
            
            if init_opt_row is None:
                init_opt_row = history_data[0]

            final_row = history_data[-1]
            
            def parse_json(s):
                try: return np.array(json.loads(s))
                except: return np.array([])

            slw_final = parse_json(final_row[6])
            slp_final = parse_json(final_row[7])
            slopes_final = parse_json(final_row[8])
            
            all_res[(b_val, c_val, q_val)] = {
                'cost_init': init_opt_row[1],
                'cost_final': final_row[1],
                'rate_final': final_row[2],
                'dist_final': final_row[3],
                'ent_final': final_row[4],
                'active': data.get('active_clusters', 0),
                'iters': len(history_data) - 1 if len(history_data) > 1 else 1,
                'total_hamming': sum(row[5] for row in history_data if row[5] is not None),
                'slw_stats': (np.mean(slw_final), np.std(slw_final)) if slw_final.size > 0 else (0,0),
                'slp_stats': (np.mean(slp_final), np.std(slp_final)) if slp_final.size > 0 else (0,0),
                'slope_norm': np.mean([np.linalg.norm(s) for s in slopes_final]) if slopes_final.size > 0 else 0
            }
            
    return all_res

def main():
    folders = [
        "fast_validated_sweep_20260524_235753",
        "afternoon_validated_sweep_20260525_152602"
    ]
    
    results = get_results_from_folders(folders)
    if not results:
        print("No results found.")
        return
        
    b_vals = sorted(list(set(k[0] for k in results.keys())))
    q_vals = sorted(list(set(k[2] for k in results.keys())))
    
    print("### COMBINED NUMERICAL DATASET: RD-CLUSTERING SENSITIVITY ###\n")
    
    for b in b_vals:
        print(f"==========================================================================================")
        print(f"BLOCK SIZE B = {b}")
        print(f"==========================================================================================")
        
        for q in q_vals:
            current_q_res = {k: v for k, v in results.items() if k[0] == b and k[2] == q}
            if not current_q_res: continue
            
            print(f"\n--- Quantization Step Q = {q} ---")
            print("| C | Final Cost | Rate (R) | Dist (D) | Entropy (H) | Active | Iters | Avg Ham/Iter | SLW Avg | SLP Avg |")
            print("|---|------------|----------|----------|-------------|--------|-------|--------------|---------|---------|")
            
            c_sorted = sorted(current_q_res.keys(), key=lambda x: x[1])
            for k in c_sorted:
                res = current_q_res[k]
                c = k[1]
                avg_ham = res['total_hamming'] / max(1, res['iters'])
                print(f"| {c:2d} | {res['cost_final']:10.2f} | {res['rate_final']:8.4f} | {res['dist_final']:8.4f} | {res['ent_final']:11.4f} | {res['active']:1d}/{c:1d} | {res['iters']:5d} | {avg_ham:12.2f} | {res['slw_stats'][0]:7.3f} | {res['slp_stats'][0]:7.3f} |")

if __name__ == "__main__":
    main()
