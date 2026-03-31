import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        super().__init__(structural_graph, luminance_centroid, centroid_label, 0.0, self_loop_weight)
        self.percentage = percentage

    def _self_loops_selection(self):
        """Select top % nodes based on sink vector S."""
        n_nodes = len(self.S)
        n_sl = max(1, int(n_nodes * self.percentage))
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
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64, 80]
    
    # Sweep parameters
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    percentages = [0.02, 0.05, 0.10, 0.20, 0.30]
    weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    plt.figure(figsize=(18, 6))

    for i, bsize in enumerate(block_sizes):
        print(f"\n--- Processing B={bsize} ---")
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        # Sample for speed
        sample_ratio = 0.002 if bsize == 4 else 0.005 if bsize == 8 else 0.01
        sampler = Sampler(ratio=sample_ratio, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        res_struct = {"r": [], "d": []}
        res_threshold = {"r": [], "d": []}
        res_percentage = {"r": [], "d": []}

        for q in q_steps:
            decider._set_vars(q)
            print(f"  Testing Q={q}...")
            
            total_v = 0
            t_r_s, t_d_s = 0, 0
            t_r_t, t_d_t = 0, 0
            t_r_p, t_d_p = 0, 0
            
            for block in tqdm(sampled_blocks, desc=f"Sweeping Blocks (Q={q})", leave=False):
                block.init_data(pc.V, pc.A)
                nv = block.Vblock.shape[0]
                total_v += nv
                
                # 1. Structural
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                _, coeffs_s = gft_computer(block, s_graph)
                cost_s, r_s, d_s = decider._RDcost(coeffs_s)
                t_r_s += r_s
                t_d_s += d_s
                
                # 2. Threshold Oracle Local
                best_cost_t = cost_s
                best_rd_t = (r_s, d_s)
                for t in thresholds:
                    for w in weights:
                        t_graph = AttributeGraph(s_graph, np.zeros(3), 1, t, w)
                        t_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_t = gft_computer(block, t_graph)
                        cost, r, d = decider._RDcost(coeffs_t)
                        if cost < best_cost_t:
                            best_cost_t = cost
                            best_rd_t = (r, d)
                t_r_t += best_rd_t[0]
                t_d_t += best_rd_t[1]

                # 3. Percentage Oracle Local
                best_cost_p = cost_s
                best_rd_p = (r_s, d_s)
                for p in percentages:
                    for w in weights:
                        p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                        p_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_p = gft_computer(block, p_graph)
                        cost, r, d = decider._RDcost(coeffs_p)
                        if cost < best_cost_p:
                            best_cost_p = cost
                            best_rd_p = (r, d)
                t_r_p += best_rd_p[0]
                t_d_p += best_rd_p[1]
                
                block.clear_data()

            res_struct["r"].append(t_r_s / total_v)
            res_struct["d"].append(20 * np.log10(255 / (t_d_s / total_v)))
            res_threshold["r"].append(t_r_t / total_v)
            res_threshold["d"].append(20 * np.log10(255 / (t_d_t / total_v)))
            res_percentage["r"].append(t_r_p / total_v)
            res_percentage["d"].append(20 * np.log10(255 / (t_d_p / total_v)))

        # Plotting
        plt.subplot(1, 3, i+1)
        plt.plot(res_struct["r"], res_struct["d"], 'o-', label='Structural', alpha=0.6)
        plt.plot(res_threshold["r"], res_threshold["d"], 's--', label='Oracle Local (Threshold)', alpha=0.8)
        plt.plot(res_percentage["r"], res_percentage["d"], 'd:', label='Oracle Local (Percentage)', color='red', linewidth=2)
        
        # Add Q labels
        for j, q in enumerate(q_steps):
            plt.text(res_percentage["r"][j], res_percentage["d"][j], f"Q={q}", fontsize=9, verticalalignment='bottom')

        plt.title(f"RD Curve B={bsize}")
        plt.xlabel("Rate (Non-zero / Vertex)")
        plt.ylabel("PSNR (Approximated)")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig("logs/rd_percentage_oracle.png")
    print("\nUltra-adaptive Percentage vs Threshold plot saved to logs/rd_percentage_oracle.png")

if __name__ == "__main__":
    run_experiment()
