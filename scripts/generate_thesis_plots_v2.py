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
                
            all_experiments[unique_id] = {
                'meta': meta,
                'history': history,
                'b': int(run_id.split('_')[0][1:]),
                'c': int(run_id.split('_')[1][1:]),
                'q': int(run_id.split('_')[2][1:]),
                'source': source,
                'run_id': run_id
            }
            
    return all_experiments

def plot_cost_evolution(data):
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        # Exclude 40-iter run (B4_C6_Q24 from Afternoon source)
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and not (d['b']==4 and d['c']==6 and d['source']=='AFT')]
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
    b_sizes = [4, 8]
    for b in b_sizes:
        plt.figure(figsize=(10, 6))
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24 and not (d['b']==4 and d['c']==6 and d['source']=='AFT')]
        if not runs: continue
        
        for r in sorted(runs, key=lambda x: x['c']):
            hist = r['history']
            iters = [h['iter'] for h in hist][1:]
            hams = [h['ham'] for h in hist][1:]
            plt.plot(iters, hams, '.-', label=f"C={r['c']}")
            
        plt.title(f'Convergence Speed: Hamming Δ (Block Size B={b}, Q=24)')
        plt.xlabel('Iteration')
        plt.ylabel('Label Changes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(OUTPUT_DIR / f"hamming_evolution_B{b}.png", dpi=300)
        plt.close()

def plot_parameter_split(data, param_name, b_size):
    """Generates 4 subplots: Min, Max, Avg, Std for a given parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{param_name.upper()} Evolution Statistics (Block Size B={b_size}, Q=24)', fontsize=16)
    
    runs = [d for d in data.values() if d['b'] == b_size and d['q'] == 24 and not (d['b']==4 and d['c']==6 and d['source']=='AFT')]
    if not runs: return
    
    for r in sorted(runs, key=lambda x: x['c']):
        hist = r['history']
        iters = [h['iter'] for h in hist]
        
        # Mapping param names to hist keys
        p_key = param_name.lower() # 'slp' or 'slw'
        
        axes[0, 0].plot(iters, [h[f'{p_key}_avg'] for h in hist], label=f"C={r['c']}")
        axes[0, 1].plot(iters, [h[f'{p_key}_std'] for h in hist], label=f"C={r['c']}")
        axes[1, 0].plot(iters, [h[f'{p_key}_min'] for h in hist], label=f"C={r['c']}")
        axes[1, 1].plot(iters, [h[f'{p_key}_max'] for h in hist], label=f"C={r['c']}")
        
    axes[0, 0].set_title('Average')
    axes[0, 1].set_title('Standard Deviation')
    axes[1, 0].set_title('Minimum')
    axes[1, 1].set_title('Maximum')
    
    for ax in axes.flat:
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param_name.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small', ncol=2)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / f"{p_key}_split_B{b_size}.png", dpi=300)
    plt.close()

def plot_efficiency_frontier(data):
    plt.figure(figsize=(12, 8))
    
    # Use all data points except 40-iter
    runs = [d for d in data.values() if not (d['b']==4 and d['c']==6 and d['source']=='AFT')]
    
    q_colors = {12: 'cyan', 24: 'blue', 36: 'red', 48: 'green', 64: 'black'}
    
    for run in runs:
        hist = run['history']
        if len(hist) < 2: continue
        
        iter1 = next((h for h in hist if h['cost'] is not None), None)
        if not iter1: continue
        
        final = hist[-1]
        gain = (final['cost'] - iter1['cost']) / iter1['cost'] * 100
        entropy = final['entropy']
        
        plt.scatter(entropy, gain, color=q_colors.get(run['q'], 'gray'), s=100, alpha=0.7)
        plt.annotate(f"B{run['b']}C{run['c']}Q{run['q']}", (entropy, gain), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
    plt.title('Informational Efficiency Frontier (All Combined Runs)', fontsize=14)
    plt.xlabel('Codebook Entropy (H) [bits]')
    plt.ylabel('Internal Gain (%) vs Iter 1')
    plt.grid(True, alpha=0.3)
    
    # Legend for Q
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Q={q}', markerfacecolor=c, markersize=10) 
                       for q, c in sorted(q_colors.items())]
    plt.legend(handles=legend_elements, title="Quantization")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_frontier.png", dpi=300)
    plt.close()

def main():
    data = get_data()
    print("[*] Generating split evolution plots...")
    for b in [4, 8]:
        plot_cost_evolution(data)
        plot_hamming_evolution(data)
        plot_parameter_split(data, 'SLP', b)
        plot_parameter_split(data, 'SLW', b)
    
    print("[*] Generating Efficiency Frontier...")
    plot_efficiency_frontier(data)
    
    print(f"[+] All updated plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
