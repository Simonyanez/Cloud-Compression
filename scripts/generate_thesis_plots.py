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

def get_data():
    all_experiments = {} # exp_id -> {meta: {}, history: []}
    
    for folder in FOLDERS:
        root = Path(folder)
        if not root.exists(): continue
        
        # result_*.json files
        result_files = list(root.glob("result_*.json"))
        for rf in result_files:
            with open(rf, 'r') as f:
                meta = json.load(f)
            
            run_id = meta['experiment_code']
            # Source key for disambiguation if needed
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
                
                # Extract slopes/slw/slp means
                slw = np.mean(d.get('self_loop_weights', [0]))
                slp = np.mean(d.get('self_loop_percentages', [0]))
                cost = d.get('total_cost')
                labels = d.get('labels', [])
                it_num = d.get('iteration', 0)
                entropy = d.get('cluster_entropy', 0)
                
                ham = 0
                if prev_labels is not None and labels:
                    ham = int(np.sum(np.array(prev_labels) != np.array(labels)))
                
                history.append({
                    'iter': it_num,
                    'cost': cost,
                    'slw': slw,
                    'slp': slp,
                    'ham': ham,
                    'entropy': entropy
                })
                prev_labels = labels
                
            all_experiments[unique_id] = {
                'meta': meta,
                'history': history,
                'b': int(run_id.split('_')[0][1:]),
                'c': int(run_id.split('_')[1][1:]),
                'q': int(run_id.split('_')[2][1:])
            }
            
    return all_experiments

def plot_1_saturation(data):
    """Plot 1: Rate-Distortion & Capacity Saturation Curve"""
    plt.figure(figsize=(10, 6))
    
    # We focus on Q=24
    b_sizes = sorted(list(set(d['b'] for d in data.values())))
    
    for b in b_sizes:
        # Filter runs for this B and Q=24
        runs = [d for d in data.values() if d['b'] == b and d['q'] == 24]
        runs.sort(key=lambda x: x['c'])
        
        if not runs: continue
        
        c_vals = [r['c'] for r in runs]
        costs = [r['meta']['final_cost'] for r in runs]
        
        plt.plot(c_vals, costs, 'o-', label=f'Block Size B={b}', linewidth=2, markersize=8)
        
    plt.title('Capacity Saturation Curve (Q=24)', fontsize=14)
    plt.xlabel('Cluster Size (C)', fontsize=12)
    plt.ylabel('Final RD Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "capacity_saturation.png", dpi=300)
    plt.close()

def plot_2_parameter_drifts(data):
    """Plot 2: Graph Sparsification Trajectory"""
    # Pick a representative run: B4_C6_Q24_AFT (long run)
    target = data.get('B4_C6_Q24_AFT')
    if not target: return
    
    hist = target['history']
    iters = [h['iter'] for h in hist]
    slp = [h['slp'] for h in hist]
    slw = [h['slw'] for h in hist]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Optimization Iteration')
    ax1.set_ylabel('Avg Self-Loop Percentage (SLP)', color=color)
    ax1.plot(iters, slp, 'o-', color=color, label='SLP (Sparsity)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Self-Loop Weight (SLW)', color=color)
    ax2.plot(iters, slw, 's-', color=color, label='SLW (Weight)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Graph Sparsification Trajectory (B=4, C=6, Q=24)', fontsize=14)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_drifts.png", dpi=300)
    plt.close()

def plot_3_convergence(data):
    """Plot 3: Convergence Velocity & The 'Iter 2' Breakthrough"""
    plt.figure(figsize=(10, 6))
    
    # Plot Hamming Distance evolution for a few B4 runs
    runs_to_plot = ['B4_C4_Q24_FST', 'B4_C6_Q24_AFT', 'B8_C4_Q24_FST']
    
    for rid in runs_to_plot:
        run = data.get(rid)
        if not run: continue
        
        hist = run['history']
        iters = [h['iter'] for h in hist][1:] # Skip iter 0
        hams = [h['ham'] for h in hist][1:]
        
        plt.plot(iters, hams, '.-', label=rid.replace('_FST','').replace('_AFT',''))
        
    plt.axvline(x=2, color='gray', linestyle='--', alpha=0.7, label='Initial Refinement')
    plt.title('Convergence Velocity (Hamming Distance Evolution)', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Block Label Changes (Hamming Δ)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence_velocity.png", dpi=300)
    plt.close()

def plot_4_efficiency_frontier(data):
    """Plot 4: The Informational Efficiency Frontier"""
    plt.figure(figsize=(10, 6))
    
    # We need internal gain %
    # Delta = (Final - Iter 1) / Iter 1 * 100
    
    b_sizes = {4: 'blue', 8: 'orange', 16: 'green'}
    
    for rid, run in data.items():
        hist = run['history']
        if len(hist) < 2: continue
        
        # Get Iter 1 as baseline for internal gain
        iter1 = None
        for h in hist:
            if h['cost'] is not None:
                iter1 = h
                break
        if not iter1: continue
        
        final = hist[-1]
        gain = (final['cost'] - iter1['cost']) / iter1['cost'] * 100
        entropy = final['entropy']
        
        color = b_sizes.get(run['b'], 'gray')
        plt.scatter(entropy, gain, c=color, s=run['c']*20, alpha=0.6)
        plt.annotate(f"C{run['c']}_B{run['b']}", (entropy, gain), fontsize=8, alpha=0.8)
        
    plt.title('Informational Efficiency Frontier', fontsize=14)
    plt.xlabel('Codebook Entropy (H) [bits]', fontsize=12)
    plt.ylabel('Internal Gain (Δ%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Dummy handles for legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='B=4', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='B=8', markerfacecolor='orange', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='B=16', markerfacecolor='green', markersize=10)]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "efficiency_frontier.png", dpi=300)
    plt.close()

def main():
    print("[*] Gathering data from sweeps...")
    data = get_data()
    
    print("[*] Generating Plot 1: Capacity Saturation...")
    plot_1_saturation(data)
    
    print("[*] Generating Plot 2: Parameter Drifts...")
    plot_2_parameter_drifts(data)
    
    print("[*] Generating Plot 3: Convergence Velocity...")
    plot_3_convergence(data)
    
    print("[*] Generating Plot 4: Efficiency Frontier...")
    plot_4_efficiency_frontier(data)
    
    print(f"[+] All plots saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
