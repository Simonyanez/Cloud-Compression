import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to data
FOLDERS = [
    "fast_validated_sweep_20260524_235753",
    "afternoon_validated_sweep_20260525_152602"
]
OUTPUT_DIR = Path("docs/thesis/img/resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_data():
    all_experiments = {} 
    
    for folder in FOLDERS:
        root = Path(folder)
        if not root.exists(): continue
        
        result_files = list(root.glob("result_*.json"))
        for rf in result_files:
            with open(rf, 'r') as f:
                meta = json.load(f)
            
            run_id = meta['experiment_code']
            source = "AFT" if "afternoon" in folder else "FST"
            unique_id = f"{run_id}_{source}"
            
            # Load iteration history
            iter_files = sorted(list(root.glob(f"{run_id}_iter_*.json")), 
                               key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
            
            history = []
            prev_labels = None
            for it_f in iter_files:
                with open(it_f, 'r') as f:
                    d = json.load(f)
                
                weights = np.array(d.get('self_loop_weights', [0]))
                percs = np.array(d.get('self_loop_percentages', [0]))
                
                # Exclude index 0 (DC) if multiple clusters exist
                if len(weights) > 1:
                    weights = weights[1:]
                    percs = percs[1:]
                
                history.append({
                    'iter': d.get('iteration', 0),
                    'cost': d.get('total_cost'),
                    'slw_avg': np.mean(weights),
                    'slw_min': np.min(weights),
                    'slw_max': np.max(weights),
                    'slp_avg': np.mean(percs),
                    'slp_min': np.min(percs),
                    'slp_max': np.max(percs),
                    'ham': int(np.sum(np.array(prev_labels) != np.array(d['labels']))) if prev_labels is not None else 0,
                    'entropy': d.get('cluster_entropy', 0)
                })
                prev_labels = d['labels']
                
            all_experiments[unique_id] = {
                'meta': meta,
                'history': history,
                'b': int(run_id.split('_')[0][1:]),
                'c': int(run_id.split('_')[1][1:]),
                'q': int(run_id.split('_')[2][1:])
            }
            
    return all_experiments

def plot_cost_evolution(data):
    """One plot per block size showing cost evolution."""
    b_sizes = [4, 8, 16]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist if h['cost'] is not None]
            costs = [h['cost'] for h in hist if h['cost'] is not None]
            plt.plot(iters, costs, 'o-', label=f"C={r['c']}")
        
        plt.title(f'RD Cost Evolution (Block Size B={b}, Q=24)')
        plt.xlabel('Iteration')
        plt.ylabel('Total RD Cost')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"cost_evolution_B{b}.png", dpi=300)
        plt.close()

def plot_hamming_evolution(data):
    """One plot per block size showing Hamming distance."""
    b_sizes = [4, 8, 16]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist][1:] # Skip iter 0
            hams = [h['ham'] for h in hist][1:]
            plt.plot(iters, hams, '.-', label=f"C={r['c']}")
            
        plt.title(f'Convergence Speed: Hamming Δ (Block Size B={b}, Q=24)')
        plt.xlabel('Iteration')
        plt.ylabel('Label Changes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"hamming_evolution_B{b}.png", dpi=300)
        plt.close()

def plot_slp_evolution(data):
    """Min/Max SLP Trajectories per block size."""
    b_sizes = [4, 8, 16]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist]
            mins = [h['slp_min'] for h in hist]
            maxs = [h['slp_max'] for h in hist]
            
            line, = plt.plot(iters, mins, '--', alpha=0.5)
            plt.plot(iters, maxs, '-', color=line.get_color(), label=f"C={r['c']}")
            plt.fill_between(iters, mins, maxs, alpha=0.1, color=line.get_color())
            
        plt.title(f'Self-Loop Percentage (SLP) Bounds Evolution (B={b})')
        plt.xlabel('Iteration')
        plt.ylabel('SLP [min, max]')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"slp_bounds_B{b}.png", dpi=300)
        plt.close()

def plot_slw_evolution(data):
    """Min/Max SLW Trajectories per block size."""
    b_sizes = [4, 8, 16]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist]
            mins = [h['slw_min'] for h in hist]
            maxs = [h['slw_max'] for h in hist]
            
            line, = plt.plot(iters, mins, '--', alpha=0.5)
            plt.plot(iters, maxs, '-', color=line.get_color(), label=f"C={r['c']}")
            plt.fill_between(iters, mins, maxs, alpha=0.1, color=line.get_color())
            
        plt.title(f'Self-Loop Weight (SLW) Bounds Evolution (B={b})')
        plt.xlabel('Iteration')
        plt.ylabel('SLW [min, max]')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"slw_bounds_B{b}.png", dpi=300)
        plt.close()

def main():
    data = get_data()
    print("[*] Generating Cost Evolution plots...")
    plot_cost_evolution(data)
    print("[*] Generating Hamming Evolution plots...")
    plot_hamming_evolution(data)
    print("[*] Generating SLP Bounds plots...")
    plot_slp_evolution(data)
    print("[*] Generating SLW Bounds plots...")
    plot_slw_evolution(data)
    print(f"[+] Multi-scale analysis plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
