import sys
import os
from pathlib import Path
import sqlite3
import json
import dataclasses
from datetime import datetime
from argparse import ArgumentParser
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import logging
from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.rd_cluster.main import run_rd_clustering
from pcadc.rd_cluster.clusterer import RDClusterer

# Setup logging
logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass
class ExpResult:
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

def save_history_to_db(db_path: Path, experiment_code: str, block_size: int, history):
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

# Global to store current experiment code for the monkeypatch
CURRENT_EXP_CODE = "state"

# Monkeypatch RDClusterer to avoid subfolders and use prefixed iteration files
def flattened_save_intermediate_state(self, state):
    global CURRENT_EXP_CODE
    filepath = self.temp_folder / f"{CURRENT_EXP_CODE}_iter_{state.iteration:03d}.json"
    with open(filepath, 'w') as f:
        json.dump(state.to_dict(), f, indent=4)

RDClusterer._save_intermediate_state = flattened_save_intermediate_state

def main():
    global CURRENT_EXP_CODE
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--q_step", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--mode", type=str, choices=["draft", "production"], default="production")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_experiment_config(Path(args.config))
    
    # Apply Overrides
    params.clustering_params.max_iterations = args.max_iters
    params.clustering_params.number_of_clusters = args.clusters
    params.sequential_params.block_size = args.block_size
    params.sequential_params.sample_percentage = args.sample_rate
    params.sequential_params.quantization_steps = [args.q_step]
    params.clustering_params.optimization_mode = args.mode
    
    experiment_code = f"B{args.block_size}_C{args.clusters}_Q{args.q_step}"
    CURRENT_EXP_CODE = experiment_code
    params.metadata.experiment_code = experiment_code
    params.metadata.export_folder = out_dir
    params.metadata.temp_folder = out_dir # Flattened: temp is same as out

    # Load Data
    colourist = Colourist()
    point_cloud = PointCloud.from_file(
        path=params.sequential_params.point_cloud_path, 
        fmt="ply", 
        metadata=params.pointcloud
    )
    point_cloud.transform_attributes(colourist._RGBtoYUV)
    
    vertices = point_cloud.vertices
    attributes = point_cloud.attributes
    
    mortonpartition = MortonBlockPartition()
    _, blocks = mortonpartition.partition(point_cloud, bsize=params.sequential_params.block_size)
    
    sampler = Sampler(params.sequential_params.sample_percentage, n_strata=5, seed=42)
    sampled_blocks = sampler(vertices, attributes, blocks)
    
    # Run
    final_state, history = run_rd_clustering(sampled_blocks, vertices, attributes, params)
    
    # Save Outputs directly to out_dir
    db_path = out_dir / f"{experiment_code}.db"
    save_history_to_db(db_path, experiment_code, args.block_size, history)
    
    unique_labels = np.unique(final_state.labels)
    active_clusters = len(unique_labels)
    is_vanishing = active_clusters < args.clusters
    empty_clusters = [k for k in range(args.clusters) if k not in unique_labels]
    
    result = ExpResult(
        experiment_code=experiment_code,
        active_clusters=active_clusters,
        total_clusters=args.clusters,
        is_vanishing=is_vanishing,
        empty_clusters=empty_clusters,
        final_cost=final_state.total_cost,
        final_slopes=final_state.slopes.tolist(),
        final_slw=final_state.self_loop_weights.tolist(),
        final_slp=final_state.self_loop_percentages.tolist(),
        iterations=len(history.states),
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    with open(out_dir / f"result_{experiment_code}.json", "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=4)
    
    with open(out_dir / f"config_{experiment_code}.json", "w") as f:
        json.dump(params.to_dict(), f, indent=4)

if __name__ == "__main__":
    main()
