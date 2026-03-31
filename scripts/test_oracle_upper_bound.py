import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.transforms import GFTStrategyWraper
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph

def test_oracle_for_block_size(bsize, q_step=48, sample_ratio=0.05):
    # 1. Setup
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # Sample for the high-res sweep
    sampler = Sampler(ratio=sample_ratio, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    # 2. Fine Sweep Parameters
    decider = Decider(mode=params.sequential_params.decider_mode, 
                      lagrange_proportional=params.sequential_params.lagrange_proportional)
    decider._set_vars(q_step)

    thresholds = np.linspace(0.1, 0.95, 10)
    weights = np.linspace(0.1, 3.0, 10)

    print(f"\n{'='*50}")
    print(f"Starting FINAL High-Res Sweep for BSIZE={bsize}")
    print(f"Sampled {len(sampled_blocks)}/{len(all_blocks)} blocks ({sample_ratio*100:.1f}%)")
    print(f"Weights range: [{weights[0]:.2f}, {weights[-1]:.2f}]")
    print(f"Thresholds range: [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]")
    print(f"{'='*50}")

    # 3. Pre-compute and Evaluate
    gain_matrix = np.zeros((len(thresholds), len(weights)))
    sup_matrix = np.zeros((len(thresholds), len(weights)))

    # Pre-compute structural for the sampled set
    structural_results = []
    for block in tqdm(sampled_blocks, desc=f"Pre-computing Structural (B={bsize})"):
        block.init_data(pc.V, pc.A)
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs = gft_computer(block, s_graph)
        rd_s, _, _ = decider._RDcost(coeffs)
        structural_results.append((rd_s, s_graph)) # Keep graph for reuse if possible, but actually we need to clear data
        block.clear_data()

    for i, t in enumerate(thresholds):
        for j, w in enumerate(weights):
            total_dc_rd = 0
            total_hyb_rd = 0
            better_count = 0
            
            for b_idx, block in enumerate(sampled_blocks):
                block.init_data(pc.V, pc.A)
                rd_s, _ = structural_results[b_idx]
                
                # 2. Adaptive RD Cost
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                attr_graph = AttributeGraph(s_graph, np.zeros(3), 1, t, float(w))
                attr_graph.set_data(block.Vblock, block.Ablock)
                _, coeffs_a = gft_computer(block, attr_graph)
                rd_a, _, _ = decider._RDcost(coeffs_a)
                
                # Safeguards for infinite/NaN
                if not np.isfinite(rd_a):
                    rd_a = 1e18 # Effectively infinite cost
                
                total_dc_rd += rd_s
                if rd_a < rd_s:
                    better_count += 1
                    total_hyb_rd += rd_a
                else:
                    total_hyb_rd += rd_s
                block.clear_data()
            
            # Final gain calculation with safety
            gain = ((total_dc_rd - total_hyb_rd) / total_dc_rd * 100) if total_dc_rd > 0 else 0.0
            sup = (better_count / len(sampled_blocks) * 100)
            
            gain_matrix[i, j] = gain
            sup_matrix[i, j] = sup
            
            print(f"  [T={t:.3f}, W={w:.2f}] -> Gain: {gain:6.3f}% | Superiority: {sup:5.1f}%")

    # 4. Reporting Best result
    best_idx = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
    print(f"\n--- Best Configuration Found for BSIZE={bsize} ---")
    print(f"  Threshold: {thresholds[best_idx[0]]:.4f}")
    print(f"  Weight:    {weights[best_idx[1]]:.4f}")
    print(f"  Max Gain:  {gain_matrix[best_idx]:.4f}%")
    print(f"  Superiority: {sup_matrix[best_idx]:.2f}%")

    # 5. Visualization
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(gain_matrix, annot=True, fmt=".2f", 
                xticklabels=[f"{w:1.1f}" for w in weights], 
                yticklabels=[f"{t:1.2f}" for t in thresholds])
    plt.title(f"Hybrid RD Gain % (Oracle, Q={q_step}, B={bsize})")
    plt.xlabel("Self-loop Weight")
    plt.ylabel("Self-loop Threshold")

    plt.subplot(1, 2, 2)
    sns.heatmap(sup_matrix, annot=True, fmt=".0f", 
                xticklabels=[f"{w:1.1f}" for w in weights], 
                yticklabels=[f"{t:1.2f}" for t in thresholds])
    plt.title(f"Superiority Ratio % (Oracle, Q={q_step}, B={bsize})")
    plt.xlabel("Self-loop Weight")
    plt.ylabel("Self-loop Threshold")

    plt.tight_layout()
    save_path = f"logs/oracle_sweep_B{bsize}.png"
    plt.savefig(save_path)
    print(f"\nHeatmaps saved to {save_path}")

    return gain_matrix, sup_matrix, thresholds, weights

def main():
    block_sizes = [4, 8, 16]
    q_step = 48
    
    # Optional: adjust sample ratio for smaller blocks to avoid long runtimes
    # B=4 has many more blocks than B=16
    sample_ratios = {4: 0.01, 8: 0.02, 16: 0.05}
    
    results = {}
    for bsize in block_sizes:
        res = test_oracle_for_block_size(bsize, q_step, sample_ratios.get(bsize, 0.05))
        results[bsize] = res

    # Summary plot of best gains across block sizes
    plt.figure(figsize=(10, 6))
    best_gains = [np.max(results[bs][0]) for bs in block_sizes]
    plt.bar([str(bs) for bs in block_sizes], best_gains, color='skyblue')
    plt.xlabel('Block Size')
    plt.ylabel('Max Hybrid Gain (%)')
    plt.title(f'Oracle Upper Bound Gain across Block Sizes (Q={q_step})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("logs/oracle_summary_blocks.png")
    print("\nSummary plot saved to logs/oracle_summary_blocks.png")

if __name__ == "__main__":
    main()
