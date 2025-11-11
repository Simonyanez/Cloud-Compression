import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def analyze_iterations(results_dir: Path):
    """
    Analyzes the iteration files in the results directory.

    Args:
        results_dir: The path to the directory containing the iteration files.
    """

    iteration_files = sorted(results_dir.glob("iteration_*.json"))
    if not iteration_files:
        print("No iteration files found.")
        return

    # Load the data from the iteration files
    iteration_data = []
    for iteration_file in iteration_files:
        with open(iteration_file, "r") as f:
            try:
                iteration_data.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {iteration_file}")
                continue

    # Get the labels from each iteration
    labels_per_iteration = [np.array(data["labels"]) for data in iteration_data]

    # --- Analysis 1: Cluster distribution per iteration ---
    num_clusters = len(np.unique(labels_per_iteration[0]))
    cluster_counts_per_iteration = []
    for labels in labels_per_iteration:
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        # Ensure all clusters are represented, even if with 0 count
        full_counts = {i: cluster_counts.get(i, 0) for i in range(num_clusters)}
        cluster_counts_per_iteration.append(full_counts)

    print("--- Cluster Distribution per Iteration ---")
    for i, counts in enumerate(cluster_counts_per_iteration):
        print(f"Iteration {i}: {counts}")

    # --- Analysis 2: Block stability ---
    num_blocks = len(labels_per_iteration[0])
    block_changes = np.zeros(num_blocks, dtype=int)
    for i in range(len(labels_per_iteration) - 1):
        block_changes += (labels_per_iteration[i] != labels_per_iteration[i+1])

    print("\n--- Block Stability ---")
    print(f"Number of blocks that never changed cluster: {np.sum(block_changes == 0)}")
    print(f"Number of blocks that changed cluster at least once: {np.sum(block_changes > 0)}")
    print(f"Average number of changes per block: {np.mean(block_changes):.2f}")
    
    # --- Analysis 3: Convergence ---
    changes_per_iteration = []
    for i in range(len(labels_per_iteration) - 1):
        changes = np.sum(labels_per_iteration[i] != labels_per_iteration[i+1])
        changes_per_iteration.append(changes)

    print("\n--- Convergence ---")
    for i, changes in enumerate(changes_per_iteration):
        print(f"Iteration {i+1}: {changes} blocks changed cluster.")

    # --- Plotting ---
    # Plot cluster distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(cluster_counts_per_iteration))
    for i in range(num_clusters):
        counts = [counts[i] for counts in cluster_counts_per_iteration]
        ax.bar(range(len(counts)), counts, label=f"Cluster {i}", bottom=bottom)
        bottom += counts
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Blocks")
    ax.set_title("Cluster Distribution Over Iterations")
    ax.legend()
    plt.savefig(results_dir / "cluster_distribution.png")
    print(f"\nSaved cluster distribution plot to {results_dir / 'cluster_distribution.png'}")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(changes_per_iteration) + 1), changes_per_iteration, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Blocks Changed")
    ax.set_title("Convergence Over Iterations")
    plt.savefig(results_dir / "convergence.png")
    print(f"Saved convergence plot to {results_dir / 'convergence.png'}")


if __name__ == "__main__":
    results_directory = Path("/home/simao/Repositories/Cloud-Compression/results/temp")
    analyze_iterations(results_directory)