import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def analyze_iterations(results_dir: Path):
    """
    Analyzes the iteration files in the results directory.

    Args:
        results_dir: The path to the directory containing the iteration files.
    """
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    iteration_files = sorted(results_dir.glob("iteration_*.json"))
    if not iteration_files:
        print("No iteration files found.")
        return

    # Load the data from the iteration files
    iteration_data = []
    for iteration_file in iteration_files:
        with open(iteration_file, "r") as f:
            try:
                data = json.load(f)
                # Make sure all keys are present
                for key in ["labels", "slopes", "total_cost", "avg_rate", "avg_distortion", "cluster_entropy", "cluster_gains"]:
                    if key not in data:
                        data[key] = None
                iteration_data.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {iteration_file}")
                continue

    # --- Plotting ---
    sns.set_theme(style="whitegrid")

    plot_cluster_distribution(iteration_data, analysis_dir)
    plot_convergence(iteration_data, analysis_dir)
    plot_rd_metrics(iteration_data, analysis_dir)
    plot_slope_movement(iteration_data, analysis_dir)
    plot_cluster_gains(iteration_data, analysis_dir)


def plot_cluster_distribution(iteration_data, output_dir):
    labels_per_iteration = [np.array(data["labels"]) for data in iteration_data if data["labels"] is not None]
    if not labels_per_iteration:
        return
        
    num_clusters = len(np.unique(labels_per_iteration[0]))
    cluster_counts_per_iteration = []
    for labels in labels_per_iteration:
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        full_counts = {i: cluster_counts.get(i, 0) for i in range(num_clusters)}
        cluster_counts_per_iteration.append(full_counts)

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
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_distribution.png")
    print(f"Saved cluster distribution plot to {output_dir / 'cluster_distribution.png'}")
    plt.close(fig)

def plot_convergence(iteration_data, output_dir):
    labels_per_iteration = [np.array(data["labels"]) for data in iteration_data if data["labels"] is not None]
    if len(labels_per_iteration) < 2:
        return

    changes_per_iteration = []
    for i in range(len(labels_per_iteration) - 1):
        changes = np.sum(labels_per_iteration[i] != labels_per_iteration[i+1])
        changes_per_iteration.append(changes)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(changes_per_iteration) + 1), changes_per_iteration, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Blocks Changed")
    ax.set_title("Convergence: Block Changes Per Iteration")
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png")
    print(f"Saved convergence plot to {output_dir / 'convergence.png'}")
    plt.close(fig)

def plot_rd_metrics(iteration_data, output_dir):
    iterations = [d['iteration'] for d in iteration_data if d['iteration'] is not None]
    total_cost = [d['total_cost'] for d in iteration_data if d['total_cost'] is not None]
    avg_rate = [d['avg_rate'] for d in iteration_data if d['avg_rate'] is not None]
    avg_distortion = [d['avg_distortion'] for d in iteration_data if d['avg_distortion'] is not None]
    cluster_entropy = [d['cluster_entropy'] for d in iteration_data if d['cluster_entropy'] is not None]

    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    axs[0].plot(iterations, total_cost, marker='o', color='r')
    axs[0].set_ylabel("Total RD Cost")
    axs[0].set_title("Rate-Distortion Metrics Over Iterations")

    axs[1].plot(iterations, avg_rate, marker='o', color='g')
    axs[1].set_ylabel("Average Rate (Sparsity)")

    axs[2].plot(iterations, avg_distortion, marker='o', color='b')
    axs[2].set_ylabel("Average Distortion")

    axs[3].plot(iterations, cluster_entropy, marker='o', color='purple')
    axs[3].set_ylabel("Cluster Entropy")
    axs[3].set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig(output_dir / "rd_metrics.png")
    print(f"Saved RD metrics plot to {output_dir / 'rd_metrics.png'}")
    plt.close(fig)

def plot_slope_movement(iteration_data, output_dir):
    slopes_per_iteration = [np.array(data["slopes"]) for data in iteration_data if data["slopes"] is not None]
    if not slopes_per_iteration:
        return

    num_clusters = slopes_per_iteration[0].shape[0]
    
    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for k in range(1, num_clusters): # Skip structural cluster
        slopes = np.array([slopes[k] for slopes in slopes_per_iteration])
        ax.plot(slopes[:, 0], slopes[:, 1], slopes[:, 2], marker='o', label=f"Cluster {k}")

    ax.set_xlabel("Slope X")
    ax.set_ylabel("Slope Y")
    ax.set_zlabel("Slope Z")
    ax.set_title("Slope Movement in 3D Space")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "slope_movement_3d.png")
    print(f"Saved 3D slope movement plot to {output_dir / 'slope_movement_3d.png'}")
    plt.close(fig)

    # 2D plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    iterations = range(len(slopes_per_iteration))
    for k in range(1, num_clusters): # Skip structural cluster
        slopes_x = [s[k][0] for s in slopes_per_iteration]
        slopes_y = [s[k][1] for s in slopes_per_iteration]
        slopes_z = [s[k][2] for s in slopes_per_iteration]
        axs[0].plot(iterations, slopes_x, marker='o', label=f"Cluster {k}")
        axs[1].plot(iterations, slopes_y, marker='o', label=f"Cluster {k}")
        axs[2].plot(iterations, slopes_z, marker='o', label=f"Cluster {k}")

    axs[0].set_ylabel("Slope X")
    axs[0].set_title("Slope Components Over Iterations")
    axs[0].legend()
    
    axs[1].set_ylabel("Slope Y")
    axs[1].legend()

    axs[2].set_ylabel("Slope Z")
    axs[2].set_xlabel("Iteration")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "slope_movement_2d.png")
    print(f"Saved 2D slope movement plot to {output_dir / 'slope_movement_2d.png'}")
    plt.close(fig)

def plot_cluster_gains(iteration_data, output_dir):
    gains_per_iteration = [d['cluster_gains'] for d in iteration_data if d['cluster_gains'] is not None]
    if not gains_per_iteration:
        return
        
    num_clusters = len(gains_per_iteration[0].keys())
    iterations = range(len(gains_per_iteration))

    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(1, num_clusters):
        gains = [gains[str(k)]['avg_gain'] for gains in gains_per_iteration]
        ax.plot(iterations, gains, marker='o', label=f"Cluster {k}")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Gain")
    ax.set_title("Average Gain per Dynamic Cluster Over Iterations")
    ax.legend()
    ax.axhline(0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_gains.png")
    print(f"Saved cluster gains plot to {output_dir / 'cluster_gains.png'}")
    plt.close(fig)


if __name__ == "__main__":
    results_directory = Path("/home/simao/Repositories/Cloud-Compression/results/temp")
    analyze_iterations(results_directory)
