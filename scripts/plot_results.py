import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import json

def plot_evolution(data: pd.DataFrame, y_column: str, title: str, xlabel: str, ylabel: str, output_path: Path, baseline_data: Optional[pd.DataFrame] = None):
    plt.figure(figsize=(10, 6))
    # Plot the line connecting all points for current experiment
    sns.lineplot(data=data, x="iteration", y=y_column, color="gray", sort=False, label="Current Experiment")
    # Plot the points with hue for q_step for current experiment
    sns.scatterplot(data=data, x="iteration", y=y_column, hue="q_step", palette="viridis", s=100)
    
    if baseline_data is not None:
        # Plot the line for baseline
        sns.lineplot(data=baseline_data, x="iteration", y=y_column, color="red", linestyle="--", sort=False, label="Baseline")
        # Plot the points for baseline (optional, can be removed if too cluttered)
        sns.scatterplot(data=baseline_data, x="iteration", y=y_column, color="red", marker="x", s=100, legend=False)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend() # Ensure legend is shown
    plt.savefig(output_path)
    plt.close()

def plot_rd_curve(data: pd.DataFrame, baseline_data: Optional[pd.DataFrame], title: str, output_path: Path, overhead_bpv: Optional[float] = None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="bpv", y="psnr", marker="o", label="Current Experiment")
    if baseline_data is not None:
        sns.lineplot(data=baseline_data, x="bpv", y="psnr", marker="o", label="Baseline")
    
    plot_title = title
    if overhead_bpv is not None:
        plot_title += f"\nCluster ID Overhead (approx.): {overhead_bpv:.4f} bpv"
    plt.title(plot_title)

    plt.xlabel("Bits per Voxel (bpv)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    for i, point in data.iterrows():
        plt.text(point['bpv'], point['psnr'], f"q={point['q_step']}")
    if baseline_data is not None:
        for i, point in baseline_data.iterrows():
            plt.text(point['bpv'], point['psnr'], f"q={point['q_step']}")
    plt.savefig(output_path)
    plt.close()

def plot_huffman_analysis(data: pd.DataFrame, title: str, output_path: Path, baseline_data: Optional[pd.DataFrame] = None):
    if data.empty and (baseline_data is None or baseline_data.empty):
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    # Prepare data for plotting
    plot_data = []
    expected_length_current = None
    if not data.empty:
        data['Experiment'] = 'Current Experiment'
        # Simplify cluster IDs to "Cluster 1", "Cluster 2", etc.
        unique_clusters = data['cluster_id'].unique()
        cluster_map = {cluster: f'Cluster {i+1}' for i, cluster in enumerate(unique_clusters)}
        data['cluster_id'] = data['cluster_id'].map(cluster_map)
        data = data.sort_values('cluster_id')
        plot_data.append(data)
        expected_length_current = (data['probability'] * data['code_length']).sum()

    expected_length_baseline = None
    if baseline_data is not None and not baseline_data.empty:
        baseline_data['Experiment'] = 'Baseline'
        # Simplify cluster IDs to "Cluster 1", "Cluster 2", etc.
        unique_clusters_baseline = baseline_data['cluster_id'].unique()
        cluster_map_baseline = {cluster: f'Cluster {i+1}' for i, cluster in enumerate(unique_clusters_baseline)}
        baseline_data['cluster_id'] = baseline_data['cluster_id'].map(cluster_map_baseline)
        baseline_data = baseline_data.sort_values('cluster_id')
        plot_data.append(baseline_data)
        expected_length_baseline = (baseline_data['probability'] * baseline_data['code_length']).sum()

    if not plot_data:
        return

    combined_data = pd.concat(plot_data)

    # Plot probabilities
    sns.barplot(ax=axes[0], data=combined_data, x="cluster_id", y="probability", hue="Experiment", palette="viridis")
    axes[0].set_title(f"Cluster Probabilities - {title}")
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Probability")
    axes[0].legend(title="Experiment Type")

    # Plot code lengths
    sns.barplot(ax=axes[1], data=combined_data, x="cluster_id", y="code_length", hue="Experiment", palette="viridis")
    
    # Add expected length to the title
    plot_title = f"Huffman Code Lengths - {title}"
    if expected_length_current is not None:
        plot_title += f"\nE[L_current] = {expected_length_current:.4f} bits"
    if expected_length_baseline is not None:
        plot_title += f"\nE[L_baseline] = {expected_length_baseline:.4f} bits"
    axes[1].set_title(plot_title)
    
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Code Length (bits)")
    axes[1].legend(title="Experiment Type")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import json

def analyze_experiment(experiment_code: str, baseline_code: Optional[str] = None):
    db_path = Path(f"results/PCADC_TEST/{experiment_code}/{experiment_code}.db")
    images_dir = Path(f"results/PCADC_TEST/{experiment_code}/images")
    images_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    
    # Load baseline data if provided
    baseline_encoding_df = None
    baseline_history_df = None
    baseline_huffman_df = None

    if baseline_code:
        baseline_db_path = Path(f"results/PCADC_TEST/{baseline_code}/{baseline_code}.db")
        baseline_conn = sqlite3.connect(baseline_db_path)
        try:
            baseline_encoding_df = pd.read_sql_query("SELECT * FROM encoding", baseline_conn)
            baseline_history_df = pd.read_sql_query("SELECT * FROM clustering_history", baseline_conn)
            baseline_huffman_df = pd.read_sql_query("SELECT * FROM cluster_codes", baseline_conn)
        except pd.io.sql.DatabaseError as e:
            print(f"Could not read baseline data from {baseline_db_path}: {e}")
        finally:
            baseline_conn.close()

    # Clustering history
    try:
        history_df = pd.read_sql_query("SELECT * FROM clustering_history", conn)
        if not history_df.empty:
            plot_evolution(history_df, "cluster_entropy", f"Cluster Entropy Evolution - {experiment_code}", "Iteration", "Cluster Entropy", images_dir / "entropy_evolution.png", baseline_data=baseline_history_df)
            plot_evolution(history_df, "hamming_distance_from_previous", f"Hamming Distance Evolution - {experiment_code}", "Iteration", "Hamming Distance", images_dir / "hamming_distance_evolution.png", baseline_data=baseline_history_df)
            plot_evolution(history_df, "total_cost", f"Total Cost Evolution - {experiment_code}", "Iteration", "Total Cost", images_dir / "cost_evolution.png", baseline_data=baseline_history_df)
            plot_evolution(history_df, "avg_rate", f"Average Rate Evolution - {experiment_code}", "Iteration", "Average Rate", images_dir / "rate_evolution.png", baseline_data=baseline_history_df)
            plot_evolution(history_df, "avg_distortion", f"Average Distortion Evolution - {experiment_code}", "Iteration", "Average Distortion", images_dir / "distortion_evolution.png", baseline_data=baseline_history_df)
    except pd.io.sql.DatabaseError as e:
        print(f"Could not read clustering_history from {db_path}: {e}")

    # Huffman code analysis
    expected_length = None
    overhead_bpv = None
    try:
        huffman_df = pd.read_sql_query("SELECT * FROM cluster_codes", conn)
        if not huffman_df.empty:
            expected_length = (huffman_df['probability'] * huffman_df['code_length']).sum()
            
            config_path = Path(f"results/PCADC_TEST/{experiment_code}/config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    block_size = config.get('sequential_params', {}).get('block_size')
                    if block_size:
                        overhead_bpv = expected_length / block_size
    except pd.io.sql.DatabaseError as e:
        print(f"Could not read cluster_codes from {db_path}: {e}")

    # Encoding results
    try:
        encoding_df = pd.read_sql_query("SELECT * FROM encoding", conn)
        if not encoding_df.empty:
            plot_rd_curve(encoding_df, baseline_encoding_df, f"Rate-Distortion Curve - {experiment_code}", images_dir / "rd_curve.png", overhead_bpv)
    except pd.io.sql.DatabaseError as e:
        print(f"Could not read encoding from {db_path}: {e}")

    # Huffman code analysis
    try:
        huffman_df = pd.read_sql_query("SELECT * FROM cluster_codes", conn)
        if not huffman_df.empty:
            plot_huffman_analysis(huffman_df, f"Huffman Analysis - {experiment_code}", images_dir / "huffman_analysis.png", baseline_data=baseline_huffman_df)
    except pd.io.sql.DatabaseError as e:
        print(f"Could not read cluster_codes from {db_path}: {e}")

    conn.close()

if __name__ == "__main__":
    experiments = ["RD-Fast-B16", "RD-Fast-B8", "RD-Fast-B4"]
    baselines = ["Baseline-B16", "Baseline-B8", "Baseline-B4"]
    for exp, baseline in zip(experiments, baselines):
        analyze_experiment(exp, baseline)
