import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import logging
from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud
from pcadc.color import Colourist
from pcadc.pointcloud import MortonBlockPartition
from pcadc.rd_cluster.main import run_rd_clustering

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_vanishing_clusters():
    print("[*] Loading configuration...")
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    
    # Override for testing
    params.clustering_params.max_iterations = 10
    params.clustering_params.number_of_clusters = 4
    params.sequential_params.sample_percentage = 0.1 # Use 10% of blocks for training
    
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
    
    # Run clustering
    print("[*] Starting RD clustering experiment...")
    final_state, history = run_rd_clustering(blocks, vertices, attributes, params)
    
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Final State: {final_state}")
    
    # Check for empty clusters
    unique_labels = np.unique(final_state.labels)
    active_clusters = len(unique_labels)
    print(f"Active clusters: {active_clusters} / {params.clustering_params.number_of_clusters}")
    
    if active_clusters < params.clustering_params.number_of_clusters:
        print("[!] WARNING: Clusters are vanishing!")
        empty_clusters = [k for k in range(params.clustering_params.number_of_clusters) if k not in unique_labels]
        print(f"Empty clusters: {empty_clusters}")
    else:
        print("[+] SUCCESS: All clusters have assignments.")

if __name__ == "__main__":
    test_vanishing_clusters()
