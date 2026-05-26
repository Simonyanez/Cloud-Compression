import json
import numpy as np
from pathlib import Path

def analyze_dynamics(folder_paths):
    all_runs = []
    
    for folder in folder_paths:
        root = Path(folder)
        if not root.exists():
            continue
            
        # Group iteration files by experiment
        iter_files = list(root.glob("*_iter_*.json"))
        run_groups = {}
        for f in iter_files:
            # B4_C10_Q24_iter_000.json -> B4_C10_Q24
            exp_id = "_".join(f.name.split("_")[:3])
            if exp_id not in run_groups:
                run_groups[exp_id] = []
            run_groups[exp_id].append(f)
            
        for exp_id, files in run_groups.items():
            # Sort by iteration
            files.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
            
            history = []
            prev_labels = None
            
            for f in files:
                with open(f, 'r') as jf:
                    try:
                        data = json.load(jf)
                    except:
                        continue
                
                labels = data.get('labels', [])
                it_num = data.get('iteration', 0)
                cost = data.get('total_cost')
                
                ham = 0
                if prev_labels is not None and labels:
                    ham = int(np.sum(np.array(prev_labels) != np.array(labels)))
                
                history.append({
                    'iter': it_num,
                    'cost': cost,
                    'ham': ham
                })
                prev_labels = labels
            
            # Source tag to distinguish afternoon vs fast
            source = "AFT" if "afternoon" in folder else "FST"
            all_runs.append({
                'id': f"{exp_id}_{source}",
                'history': history,
                'total_ham': sum(h['ham'] for h in history),
                'n_blocks': len(prev_labels) if prev_labels else 0
            })

    # Sort runs alphabetically by ID
    all_runs.sort(key=lambda x: x['id'])

    print("### DYNAMIC EVOLUTION ANALYSIS: RD-COST & HAMMING TRAJECTORY ###\n")

    for run in all_runs:
        print(f"#### Run: {run['id']} (N={run['n_blocks']} blocks)")
        print("| Iter | RD Cost     | Cost Δ % | Hamming Δ | Effort % |")
        print("|------|-------------|----------|-----------|----------|")
        
        hist = run['history']
        first_cost = None
        for h in hist:
            if h['cost'] is not None:
                first_cost = h['cost']
                break
        
        prev_cost = None
        for h in hist:
            cost_str = f"{h['cost']:11.2f}" if h['cost'] is not None else "    N/A    "
            
            c_delta_pct = 0.0
            if h['cost'] is not None and prev_cost is not None and prev_cost != 0:
                c_delta_pct = (h['cost'] - prev_cost) / prev_cost * 100
            
            eff_pct = (h['ham'] / run['n_blocks'] * 100) if run['n_blocks'] > 0 else 0
            
            print(f"| {h['iter']:4d} | {cost_str} | {c_delta_pct:+7.2f}% | {h['ham']:9d} | {eff_pct:7.2f}% |")
            prev_cost = h['cost']
            
        print(f"\n**Cumulative Reassignments:** {run['total_ham']} | **Final Stability:** {run['history'][-1]['ham']} blocks changed\n")

if __name__ == "__main__":
    analyze_dynamics([
        "fast_validated_sweep_20260524_235753",
        "afternoon_validated_sweep_20260525_152602"
    ])
