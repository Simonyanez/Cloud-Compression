from argparse import ArgumentParser
import sys
import os
from pathlib import Path
import sqlite3
import json
import dataclasses
from datetime import datetime
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import logging
from pcadc.parameters import load_experiment_config, ExperimentConfig
from pcadc.pointcloud import PointCloud
from pcadc.color import Colourist
from pcadc.pointcloud import MortonBlockPartition, Sampler
from pcadc.rd_cluster.main import run_rd_clustering
from pcadc.rd_cluster.states import RDClusterState, ClusteringHistory

# Setup logging
logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass
class VanishingTestResult:
    experiment_code: str
    active_clusters: int
    total_clusters: int
    is_vanishing: bool
    empty_clusters: List[int]
    final_cost: Optional[float]
    final_slopes: List[List[float]]
    final_slw: List[float]
    final_slp: List[float]
    iterations: int
    timestamp: str

def save_history_to_db(db_path: Path, experiment_code: str, block_size: int, history: ClusteringHistory):
    """Manually save clustering history to a SQLite database without using observers."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS clustering_history(
        experiment_code TEXT,
        block_size INT,
        iteration INT,
        q_step INT,
        total_cost REAL,
        cluster_entropy REAL,
        avg_rate REAL,
        avg_distortion REAL,
        hamming_distance_from_previous INT,
        slopes TEXT,
        self_loop_weights TEXT,
        self_loop_percentages TEXT
    )""")
    
    previous_labels = None
    for state in history.states:
        hamming_dist = 0
        if previous_labels is not None:
            hamming_dist = int(np.sum(previous_labels != state.labels))
        
        cur.execute(
            "INSERT INTO clustering_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_code,
                block_size,
                state.iteration,
                state.qstep_value,
                state.total_cost,
                state.cluster_entropy,
                state.avg_rate,
                state.avg_distortion,
                hamming_dist,
                json.dumps(state.slopes.tolist()),
                json.dumps(state.self_loop_weights.tolist()),
                json.dumps(state.self_loop_percentages.tolist())
            )
        )
        previous_labels = state.labels
    conn.commit()
    conn.close()

def test_vanishing_clusters():
    # Setup argument parser
    parser = ArgumentParser(description="Test vanishing clusters with RD clustering")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters to test")
    parser.add_argument("--block_size", type=int, default=8, help="Block size for partitioning")
    parser.add_argument("--sample_frac", type=float, default=0.1, help="Sample percentage (0.0 to 1.0)")
    parser.add_argument("--max_iters", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to base configuration")
    args = parser.parse_args()

    print(f"[*] Loading configuration from {args.config}...")
    config_path = Path(args.config)
    params = load_experiment_config(config_path)
    
    # Override with command line arguments
    params.clustering_params.max_iterations = args.max_iters
    params.clustering_params.number_of_clusters = args.clusters
    params.sequential_params.block_size = args.block_size
    params.sequential_params.sample_percentage = args.sample_frac

    # Setup experiment naming
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    indicators = f"B{params.sequential_params.block_size}_C{params.clustering_params.number_of_clusters}"
    experiment_code = f"VANISH_{indicators}"
    params.metadata.experiment_code = experiment_code
    
    # Ensure results directory exists (organized by date and run time)
    results_dir = Path("results") / run_timestamp / experiment_code
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Update paths to be unique per experiment to avoid collisions
    params.metadata.export_folder = results_dir
    params.metadata.temp_folder = results_dir / "temp"
    params.metadata.temp_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Experiment Code: {experiment_code}")
    print(f"[*] Saving results to: {results_dir}")

    # Use indicators for the timestamp field in the result dataclass for uniqueness if needed, 
    # but here we use the actual time of execution.
    exec_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[*] Loading point cloud: {params.sequential_params.point_cloud_path}")
    colourist = Colourist()
    point_cloud = PointCloud.from_file(
        path=params.sequential_params.point_cloud_path, 
        fmt="ply", 
        metadata=params.pointcloud
    )
    point_cloud.transform_attributes(colourist._RGBtoYUV)
    
    vertices = point_cloud.vertices
    attributes = point_cloud.attributes
    
    print("[*] Partitioning blocks...")
    mortonpartition = MortonBlockPartition()
    _, blocks = mortonpartition.partition(point_cloud, bsize=params.sequential_params.block_size)
    
    print(f"[*] Total blocks: {len(blocks)}")
    
    print(f"[*] Subsampling blocks (ratio: {params.sequential_params.sample_percentage})...")
    sampler = Sampler(params.sequential_params.sample_percentage, n_strata=5, seed=42)
    sampled_blocks = sampler(vertices, attributes, blocks)
    
    # Run clustering
    print("[*] Starting RD clustering experiment...")
    # run_rd_clustering returns (final_state, history)
    final_state, history = run_rd_clustering(sampled_blocks, vertices, attributes, params)
    
    # Manual save to SQLite
    db_path = results_dir / f"{experiment_code}.db"
    print(f"[*] Saving history to {db_path}...")
    save_history_to_db(db_path, experiment_code, params.sequential_params.block_size, history)
    
    # Check for empty clusters
    unique_labels = np.unique(final_state.labels)
    active_clusters = len(unique_labels)
    total_clusters = params.clustering_params.number_of_clusters
    is_vanishing = active_clusters < total_clusters
    empty_clusters = [k for k in range(total_clusters) if k not in unique_labels]
    
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Final State: {final_state}")
    print(f"Active clusters: {active_clusters} / {total_clusters}")
    
    if is_vanishing:
        print("[!] WARNING: Clusters are vanishing!")
        print(f"Empty clusters: {empty_clusters}")
    else:
        print("[+] SUCCESS: All clusters have assignments.")

    # Create and save result dataclass
    result = VanishingTestResult(
        experiment_code=experiment_code,
        active_clusters=active_clusters,
        total_clusters=total_clusters,
        is_vanishing=is_vanishing,
        empty_clusters=empty_clusters,
        final_cost=final_state.total_cost,
        final_slopes=final_state.slopes.tolist(),
        final_slw=final_state.self_loop_weights.tolist(),
        final_slp=final_state.self_loop_percentages.tolist(),
        iterations=len(history.states),
        timestamp=exec_timestamp
    )
    
    # Save result as JSON
    result_path = results_dir / f"test_result_{indicators}.json"
    with open(result_path, "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=4)
    
    # Save config as JSON
    config_save_path = results_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(params.to_dict(), f, indent=4)
        
    print(f"[*] Results saved successfully in {results_dir}")

if __name__ == "__main__":
    test_vanishing_clusters()
