import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths to data - ONLY THE LAST FOLDER
FOLDERS = [
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
                
                # Exclude index 0 (DC) if multiple clusters exist for pure adaptive stats
                if len(weights) > 1:
                    weights = weights[1:]
                    percs = percs[1:]
                
                history.append({
                    'iter': d.get('iteration', 0),
                    'cost': d.get('total_cost'),
                    'slw_avg': np.mean(weights),
                    'slw_std': np.std(weights),
                    'slw_min': np.min(weights),
                    'slw_max': np.max(weights),
                    'slp_avg': np.mean(percs),
                    'slp_std': np.std(percs),
                    'slp_min': np.min(percs),
                    'slp_max': np.max(percs),
                    'ham': int(np.sum(np.array(prev_labels) != np.array(d['labels']))) if prev_labels is not None else 0,
                    'entropy': d.get('cluster_entropy', 0)
                })
                prev_labels = d['labels']
            
            # Identify Q step from run_id or meta
            parts = run_id.split('_')
            
            all_experiments[unique_id] = {
                'meta': meta,
                'history': history,
                'b': int(parts[0][1:]),
                'c': int(parts[1][1:]),
                'q': int(parts[2][1:]),
                'run_id': run_id
            }
            
    return all_experiments

def plot_cost_evolution(data):
    """One plot per block size showing cost evolution. Excludes 40-iter run."""
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        # Filter for this B and Q=24. EXCLUDE B4_C6_Q24 (the 40 iter one)
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and d['run_id'] != "B4_C6_Q24"]
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
    """One plot per block size showing Hamming distance. Excludes 40-iter run."""
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and d['run_id'] != "B4_C6_Q24"]
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
    """Min/Max and Avg/Std SLP Trajectories per block size. Excludes 40-iter run."""
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and d['run_id'] != "B4_C6_Q24"]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist]
            mins = [h['slp_min'] for h in hist]
            maxs = [h['slp_max'] for h in hist]
            avgs = [h['slp_avg'] for h in hist]
            stds = [h['slp_std'] for h in hist]
            
            line, = plt.plot(iters, avgs, '-', label=f"C={r['c']} (Avg)", linewidth=2)
            plt.plot(iters, mins, ':', color=line.get_color(), alpha=0.6)
            plt.plot(iters, maxs, ':', color=line.get_color(), alpha=0.6)
            
            # std bounds
            upper = np.array(avgs) + np.array(stds)
            lower = np.array(avgs) - np.array(stds)
            plt.plot(iters, upper, '--', color=line.get_color(), alpha=0.4)
            plt.plot(iters, lower, '--', color=line.get_color(), alpha=0.4)
            
        plt.title(f'SLP Evolution: Avg ± Std and [Min, Max] (B={b})')
        plt.xlabel('Iteration')
        plt.ylabel('Self-Loop Percentage (SLP)')
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize='small')
        plt.savefig(OUTPUT_DIR / f"slp_bounds_B{b}.png", dpi=300)
        plt.close()

def plot_slw_evolution(data):
    """Min/Max and Avg/Std SLW Trajectories per block size. Excludes 40-iter run."""
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and d['run_id'] != "B4_C6_Q24"]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist]
            mins = [h['slw_min'] for h in hist]
            maxs = [h['slw_max'] for h in hist]
            avgs = [h['slw_avg'] for h in hist]
            stds = [h['slw_std'] for h in hist]
            
            line, = plt.plot(iters, avgs, '-', label=f"C={r['c']} (Avg)", linewidth=2)
            plt.plot(iters, mins, ':', color=line.get_color(), alpha=0.6)
            plt.plot(iters, maxs, ':', color=line.get_color(), alpha=0.6)
            
            # std bounds
            upper = np.array(avgs) + np.array(stds)
            lower = np.array(avgs) - np.array(stds)
            plt.plot(iters, upper, '--', color=line.get_color(), alpha=0.4)
            plt.plot(iters, lower, '--', color=line.get_color(), alpha=0.4)
            
        plt.title(f'SLW Evolution: Avg ± Std and [Min, Max] (B={b})')
        plt.xlabel('Iteration')
        plt.ylabel('Self-Loop Weight (SLW)')
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize='small')
        plt.savefig(OUTPUT_DIR / f"slw_bounds_B{b}.png", dpi=300)
        plt.close()

def plot_4_efficiency_frontier(data):
    """Plot 4: The Informational Efficiency Frontier with clear Q labels."""
    plt.figure(figsize=(12, 8))
    
    # We use Q=12 from somewhere? No, in afternoon sweep we have Q24, Q36, Q48.
    # Q12 was in Fast Sweep. User said ONLY last experiment.
    
    colors = {24: 'blue', 36: 'red', 48: 'green'}
    
    for unique_id, run in data.items():
        hist = run['history']
        if len(hist) < 2: continue
        
        iter1 = None
        for h in hist:
            if h['cost'] is not None:
                iter1 = h
                break
        if not iter1: continue
        
        final = hist[-1]
        gain = (final['cost'] - iter1['cost']) / iter1['cost'] * 100
        entropy = final['entropy']
        
        c = colors.get(run['q'], 'black')
        plt.scatter(entropy, gain, color=c, s=100, alpha=0.7)
        label = f"B{run['b']}_C{run['c']}_Q{run['q']}"
        plt.annotate(label, (entropy, gain), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
    plt.title('Informational Efficiency Frontier (Afternoon Sweep)', fontsize=14)
    plt.xlabel('Codebook Entropy (H) [bits]', fontsize=12)
    plt.ylabel('Internal Gain (Δ%) vs Initialization', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Legend for Q steps
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Q=24', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Q=36', markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Q=48', markerfacecolor='green', markersize=10)]
    plt.legend(handles=legend_elements, title="Quantization Step")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_frontier.png", dpi=300)
    plt.close()

def main():
    print("[*] Gathering data from the latest sweep...")
    data = get_data()
    
    print("[*] Generating Cost Evolution plots...")
    plot_cost_evolution(data)
    
    print("[*] Generating Hamming Evolution plots...")
    plot_hamming_evolution(data)
    
    print("[*] Generating SLP Bounds plots...")
    plot_slp_evolution(data)
    
    print("[*] Generating SLW Bounds plots...")
    plot_slw_evolution(data)
    
    print("[*] Generating Efficiency Frontier plot...")
    plot_4_efficiency_frontier(data)
    
    # Remove capacity saturation plot if it exists
    sat_file = OUTPUT_DIR / "capacity_saturation.png"
    if sat_file.exists():
        sat_file.unlink()
        print("[*] Removed capacity saturation plot.")
    
    print(f"[+] All updated plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
