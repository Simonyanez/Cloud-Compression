import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, PointCloudMetadata, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.parameters import load_experiment_config
from pcadc.transforms import GFTStrategyWraper
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph

def test_ideal_rmse_sweep():
    # 1. Setup
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    approximator = Approximator()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=params.sequential_params.block_size)
    
    # 5% sample for speed in the sweep
    sampler = Sampler(ratio=0.05, n_strata=5)
    blocks = sampler(pc.V, pc.A, all_blocks)
    
    # 2. Pre-extract Oracle Slopes and Structural Coefficients
    print(f"Pre-processing {len(blocks)} blocks...")
    oracle_slopes = []
    structural_coeffs = []
    
    for block in tqdm(blocks, desc="Extracting features"):
        block.init_data(pc.V, pc.A)
        # Oracle slope
        fit = approximator(block)
        oracle_slopes.append(fit.coeffs[1:])
        
        # Structural coeffs (independent of qstep/clusters)
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs = gft_computer(block, s_graph)
        structural_coeffs.append(coeffs)
        
        block.clear_data()
        
    oracle_slopes = np.array(oracle_slopes)
    
    q_steps = params.sequential_params.quantization_steps
    cluster_counts = [4, 8, 16, 32, 64]
    
    results = {} # (n_clusters, qstep) -> improvement%

    # 3. Sweep
    for n_clusters in cluster_counts:
        print(f"\nEvaluating n_clusters={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(oracle_slopes)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Pre-compute all Adaptive Coefficients for this cluster set
        # (This is the slow part, but only done once per n_clusters)
        adaptive_coeffs_list = []
        for i, block in enumerate(tqdm(blocks, desc=f"Computing Adaptive GFTs (K={n_clusters})")):
            block.init_data(pc.V, pc.A)
            best_centroid = centroids[labels[i]]
            
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            
            attr_graph = AttributeGraph(
                s_graph, 
                best_centroid, 
                labels[i] + 1, 
                params.sequential_params.self_loop_threshold,
                params.sequential_params.self_loop_weight
            )
            attr_graph.set_data(block.Vblock, block.Ablock)
            _, coeffs = gft_computer(block, attr_graph)
            adaptive_coeffs_list.append(coeffs)
            block.clear_data()

        for q in q_steps:
            decider = Decider(mode=params.sequential_params.decider_mode, 
                              lagrange_proportional=params.sequential_params.lagrange_proportional)
            decider._set_vars(q)
            
            total_dc_cost = 0
            total_ideal_cost = 0
            
            for i in range(len(blocks)):
                # Cost for DC
                rd_dc, _, _ = decider._RDcost(structural_coeffs[i])
                total_dc_cost += rd_dc
                
                # Cost for Adaptive
                rd_ideal, _, _ = decider._RDcost(adaptive_coeffs_list[i])
                total_ideal_cost += rd_ideal
            
            improvement = ((total_dc_cost - total_ideal_cost) / total_dc_cost) * 100
            results[(n_clusters, q)] = improvement
            print(f"  Q={q}: Improvement = {improvement:+.2f}%")

    # 4. Final Reporting & Plotting
    print("\n--- Summary Table (Improvement %) ---")
    header = "K \\ Q | " + " | ".join([f"{q:2d}" for q in q_steps])
    print(header)
    print("-" * len(header))
    for k in cluster_counts:
        row = f"{k:5d} | " + " | ".join([f"{results[(k, q)]:+5.2f}" for q in q_steps])
        print(row)

    # Plotting
    plt.figure(figsize=(10, 6))
    for k in cluster_counts:
        imps = [results[(k, q)] for q in q_steps]
        plt.plot(q_steps, imps, marker='o', label=f'K={k}')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Ideal Linear Representation Improvement Sweep")
    plt.xlabel("Quantization Step (Q)")
    plt.ylabel("Improvement over Structural GFT (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("logs/ideal_rmse_sweep.png")
    print("\nSweep plot saved to logs/ideal_rmse_sweep.png")

if __name__ == "__main__":
    test_ideal_rmse_sweep()
