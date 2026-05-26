import json
import sqlite3
import numpy as np
from pathlib import Path

def get_results():
    folders = ["fast_validated_sweep_20260524_235753", "afternoon_validated_sweep_20260525_152602"]
    all_res = {}
    
    for folder in folders:
        root = Path(folder)
        if not root.exists(): continue
        result_files = list(root.glob("result_*.json"))
        for rf in result_files:
            parts = rf.stem.split('_')
            b, c, q = int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
            with open(rf, 'r') as f:
                data = json.load(f)
            
            run_id = f"B{b}_C{c}_Q{q}"
            db_file = root / f"{run_id}.db"
            
            rate, dist, entropy, total_ham = 0, 0, 0, 0
            if db_file.exists():
                conn = sqlite3.connect(db_file)
                cur = conn.cursor()
                # Get final state metrics
                cur.execute("SELECT avg_rate, avg_distortion, cluster_entropy FROM clustering_history WHERE iteration = (SELECT MAX(iteration) FROM clustering_history)")
                row = cur.fetchone()
                if row:
                    rate, dist, entropy = row
                
                # Get total hamming
                cur.execute("SELECT SUM(hamming_distance_from_previous) FROM clustering_history")
                total_ham = cur.fetchone()[0] or 0
                conn.close()

            key = (b, c, q, "AFT" if "afternoon" in folder else "FST")
            all_res[key] = {
                'cost': data['final_cost'],
                'rate': rate,
                'dist': dist,
                'entropy': entropy,
                'ham': total_ham,
                'iters': data['iterations'],
                'active': data['active_clusters']
            }
    return all_res

def to_latex_row(label, res):
    return f"{label} & {res['cost']:,.2f} & {res['rate']:.4f} & {res['dist']:.4f} & {res['entropy']:.4f} & {res['active']} \\\\"

def main():
    data = get_results()
    
    print("\n% Table: B=4, Q=24, Capacity Sweep")
    for c in [2, 4, 6, 8, 10]:
        res = data.get((4, c, 24, "AFT")) or data.get((4, c, 24, "FST"))
        if res: print(to_latex_row(f"C={c}", res))

    print("\n% Table: B=4, C=4, Quantization Sweep")
    for q in [12, 24, 36, 44, 48, 64]:
        res = data.get((4, 4, q, "AFT")) or data.get((4, 4, q, "FST"))
        if res: print(to_latex_row(f"Q={q}", res))

    print("\n% Table: Q=24, C=4, Resolution Sweep")
    for b in [4, 8, 16]:
        res = data.get((b, 4, 24, "AFT")) or data.get((b, 4, 24, "FST"))
        if res: print(to_latex_row(f"B={b}", res))

    print("\n% Table: 40-iteration Comparison (B=4, C=6, Q=24)")
    fst = data.get((4, 6, 24, "FST"))
    aft = data.get((4, 6, 24, "AFT"))
    if fst and aft:
        print(f"Standard (20 iters) & {fst['cost']:,.2f} & {fst['ham']} \\\\")
        print(f"Prolonged (40 iters) & {aft['cost']:,.2f} & {aft['ham']} \\\\")

if __name__ == "__main__":
    main()
