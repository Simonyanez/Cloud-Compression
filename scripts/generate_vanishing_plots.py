import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_evolution(db_path, output_path, experiment_code):
    """Generates a multi-panel plot showing the evolution of metrics over iterations."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Get all columns from clustering_history
    cur.execute("SELECT iteration, total_cost, cluster_entropy, avg_rate, avg_distortion, hamming_distance_from_previous FROM clustering_history")
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return

    # Convert to numpy array for easier indexing, handling potential None/NaN
    data = []
    for row in rows:
        if row[1] is not None: # Ensure total_cost is not None
            data.append(row)
    
    if not data:
        return
        
    data = np.array(data)
    
    iterations = data[:, 0]
    total_cost = data[:, 1]
    entropy = data[:, 2]
    rate = data[:, 3]
    distortion = data[:, 4]
    hamming = data[:, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Clustering Evolution: {experiment_code}", fontsize=16)
    
    # 1. Total Cost
    axes[0, 0].plot(iterations, total_cost, 'o-', color='blue')
    axes[0, 0].set_title('Total RD Cost Evolution')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Entropy
    axes[0, 1].plot(iterations, entropy, 'o-', color='green')
    axes[0, 1].set_title('Cluster Entropy Evolution')
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hamming Distance
    axes[1, 0].bar(iterations, hamming, color='orange', alpha=0.7)
    axes[1, 0].set_title('Label Changes (Hamming Distance)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rate & Distortion (Dual Axis)
    ax4_2 = axes[1, 1].twinx()
    p1, = axes[1, 1].plot(iterations, rate, 's-', color='red', label='Rate')
    p2, = ax4_2.plot(iterations, distortion, '^-', color='purple', label='Distortion')
    axes[1, 1].set_title('Rate and Distortion')
    axes[1, 1].set_ylabel('Avg Rate (bits/block)')
    ax4_2.set_ylabel('Avg Distortion (MSE)')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].legend(handles=[p1, p2])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    results_root = Path("results")
    vanish_dirs = sorted(list(results_root.glob("VANISH_*")))
    
    comparison_data = []
    
    for vdir in vanish_dirs:
        # Find the database file
        db_files = list(vdir.glob("*.db"))
        if not db_files:
            continue
        db_path = db_files[0]
        
        # Find the result json
        json_files = list(vdir.glob("test_result_*.json"))
        if not json_files:
            continue
        
        with open(json_files[0], 'r') as f:
            res = json.load(f)
            comparison_data.append(res)
        
        # Plot evolution for this specific experiment
        evolution_plot_path = vdir / "evolution_plot.png"
        print(f"[*] Generating evolution plot for {vdir.name}...")
        plot_evolution(db_path, evolution_plot_path, vdir.name)
        
    if not comparison_data:
        print("[!] No comparison data found.")
        return
        
    # Sort by total_clusters
    comparison_data.sort(key=lambda x: x['total_clusters'])
    
    total_clusters = [x['total_clusters'] for x in comparison_data]
    final_costs = [x['final_cost'] for x in comparison_data]
    active_clusters = [x['active_clusters'] for x in comparison_data]
    
    # 1. Final Cost vs Clusters
    plt.figure(figsize=(10, 6))
    plt.plot(total_clusters, final_costs, 'o--', markersize=8, linewidth=2)
    plt.title('Final RD Cost vs Number of Clusters')
    plt.xlabel('Number of Clusters (C)')
    plt.ylabel('Total Cost')
    plt.grid(True, alpha=0.3)
    plt.xticks(total_clusters)
    
    # Add percentage labels
    base_cost = final_costs[0]
    for i, c in enumerate(total_clusters):
        cost = final_costs[i]
        pct = (cost - base_cost) / base_cost * 100
        plt.annotate(f"{pct:+.2f}%", (c, cost), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                     
    plt.savefig(results_root / "comparative_cost_vs_clusters.png", dpi=300)
    plt.close()
    
    # 2. Vanishing Status Summary (Bar)
    plt.figure(figsize=(10, 4))
    indices = np.arange(len(total_clusters))
    plt.bar(indices, active_clusters, color='skyblue', label='Active')
    vanished = [tc - ac for tc, ac in zip(total_clusters, active_clusters)]
    plt.bar(indices, vanished, bottom=active_clusters, color='salmon', label='Vanished')
    plt.title('Cluster Stability (Active vs Vanished)')
    plt.xlabel('Total Clusters Requested')
    plt.ylabel('Number of Clusters')
    plt.xticks(indices, total_clusters)
    plt.legend()
    plt.savefig(results_root / "comparative_cluster_stability.png", dpi=300)
    plt.close()
    
    print(f"[*] Comparative plots saved in {results_root}/")

if __name__ == "__main__":
    main()
