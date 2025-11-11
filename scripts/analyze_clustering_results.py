import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import ast # To safely evaluate string representations of lists/arrays

def analyze_clustering_results():
    """
    Analyzes the clustering results from the JSON files in the results/temp directory.
    """
    results_path = Path("results/temp")
    json_files = sorted(results_path.glob("iteration_*.json"))

    if not json_files:
        print("No iteration JSON files found in results/temp.")
        return

    # Parse the data from the JSON files
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            iteration_data = json.load(f)
            iteration_data['iteration'] = int(file.stem.split('_')[1])
            data.append(iteration_data)

    df = pd.DataFrame(data)

    # --- Generate Plots ---

    # Plot 1: Total cost vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['total_cost'], marker='o')
    plt.title('Total RD Cost vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Total RD Cost')
    plt.grid(True)
    plt.savefig(results_path / "total_cost_vs_iteration.png")
    plt.close()

    # Plot 2: Number of blocks per cluster vs. iteration
    cluster_assignments = df['labels'].apply(lambda x: np.bincount(x, minlength=df['slopes'].iloc[0].__len__()))
    cluster_assignments = pd.DataFrame(cluster_assignments.tolist(), index=df.index)
    
    plt.figure(figsize=(10, 6))
    for i in range(cluster_assignments.shape[1]):
        plt.plot(df['iteration'], cluster_assignments[i], marker='o', label=f'Cluster {i}')
    plt.title('Number of Blocks per Cluster vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Blocks')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_path / "blocks_per_cluster_vs_iteration.png")
    plt.close()

    # Plot 3: Average gain for each cluster vs. iteration
    cluster_gains = df['cluster_gains'].apply(pd.Series)
    
    plt.figure(figsize=(10, 6))
    for col in cluster_gains.columns:
        plt.plot(df['iteration'], cluster_gains[col], marker='o', label=f'Cluster {col}')
    plt.title('Average Gain per Cluster vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Average Gain')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_path / "gain_per_cluster_vs_iteration.png")
    plt.close()

    # Plot 4: Cluster entropy vs. iteration
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['cluster_entropy'], marker='o')
    plt.title('Cluster Entropy vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.grid(True)
    plt.savefig(results_path / "entropy_vs_iteration.png")
    plt.close()

    # --- Generate Tables ---

    # Table 1: Final slopes and number of blocks
    final_iteration = df.iloc[-1]
    final_slopes = final_iteration['slopes']
    final_assignments = cluster_assignments.iloc[-1]

    print("--- Final Clustering Results ---")
    print("Final Slopes:")
    for i, slope in enumerate(final_slopes):
        print(f"  Cluster {i}: {slope}")
    
    print("\nFinal Cluster Assignments:")
    for i, count in enumerate(final_assignments):
        print(f"  Cluster {i}: {count} blocks")


if __name__ == "__main__":
    analyze_clustering_results()