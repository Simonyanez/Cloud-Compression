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

class PercentageAttributeGraph(AttributeGraph):
    def __init__(self, structural_graph, luminance_centroid, centroid_label, percentage, self_loop_weight):
        # We temporarily pass 0.0 as threshold to keep AttributeGraph init happy
        super().__init__(structural_graph, luminance_centroid, centroid_label, 0.0, self_loop_weight)
        self.percentage = percentage

    def _self_loops_selection(self):
        """Select top % nodes based on sink vector S."""
        n_nodes = len(self.S)
        n_sl = max(1, int(n_nodes * self.percentage))
        
        # S contains sink counts. We want the top sinks.
        self.selected_nodes = np.argsort(self.S)[::-1][:n_sl]
        
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

def run_experiment():
    # 1. Setup
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    bsize = 8
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # Sample 0.5% for speed in this comparison
    sampler = Sampler(ratio=0.005, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    q_steps = [24, 32, 48, 64, 80]
    percentages = [0.05, 0.10, 0.20, 0.30]
    weights = [0.8, 1.6, 2.4]
    
    # Store results: {method: {q: gain}}
    results = {"Percentage": {}, "Threshold": {}}
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    # Best threshold found previously for B=8 was T=0.86, W=1.39 (approx)
    best_t = 0.86
    best_w_t = 1.39

    for q in q_steps:
        decider._set_vars(q)
        print(f"\n--- Testing Q={q} ---")
        
        # 1. Baseline: Structural
        total_cost_s = 0
        structural_coeffs = []
        for block in sampled_blocks:
            block.init_data(pc.V, pc.A)
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            _, coeffs = gft_computer(block, s_graph)
            cost, _, _ = decider._RDcost(coeffs)
            total_cost_s += cost
            structural_coeffs.append((cost, s_graph))
            block.clear_data()

        # 2. Percentage Sweep
        best_p_gain = -1e9
        for p in percentages:
            for w in weights:
                total_cost_p = 0
                for b_idx, block in enumerate(sampled_blocks):
                    block.init_data(pc.V, pc.A)
                    cost_s, s_graph = structural_coeffs[b_idx]
                    
                    p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                    p_graph.set_data(block.Vblock, block.Ablock)
                    _, coeffs_p = gft_computer(block, p_graph)
                    cost_p, _, _ = decider._RDcost(coeffs_p)
                    
                    total_cost_p += min(cost_s, cost_p) # Oracle choice per block
                    block.clear_data()
                
                gain = (total_cost_s - total_cost_p) / total_cost_s * 100
                if gain > best_p_gain:
                    best_p_gain = gain
                print(f"  Percentage P={p:.2f}, W={w:.1f} -> Gain: {gain:.4f}%")
        results["Percentage"][q] = best_p_gain

        # 3. Threshold (Best fixed)
        total_cost_t = 0
        for b_idx, block in enumerate(sampled_blocks):
            block.init_data(pc.V, pc.A)
            cost_s, s_graph = structural_coeffs[b_idx]
            
            t_graph = AttributeGraph(s_graph, np.zeros(3), 1, best_t, best_w_t)
            t_graph.set_data(block.Vblock, block.Ablock)
            _, coeffs_t = gft_computer(block, t_graph)
            cost_t, _, _ = decider._RDcost(coeffs_t)
            
            total_cost_t += min(cost_s, cost_t)
            block.clear_data()
        
        gain_t = (total_cost_s - total_cost_t) / total_cost_s * 100
        results["Threshold"][q] = gain_t
        print(f"  Threshold T={best_t:.2f}, W={best_w_t:.1f} -> Gain: {gain_t:.4f}%")

    # Final Plot
    plt.figure(figsize=(10, 6))
    plt.plot(q_steps, [results["Percentage"][q] for q in q_steps], 'o-', label='Best Percentage (Top Sinks)')
    plt.plot(q_steps, [results["Threshold"][q] for q in q_steps], 's--', label='Best Threshold (T=0.86)')
    plt.title("Percentage vs Threshold Self-Loop RD Gains (Oracle Selection)")
    plt.xlabel("Quantization Step (Q)")
    plt.ylabel("RD Gain (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/percentage_vs_threshold.png")
    print("\nComparison plot saved to logs/percentage_vs_threshold.png")

if __name__ == "__main__":
    run_experiment()
