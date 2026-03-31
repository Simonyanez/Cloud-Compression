import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.integrate import quad

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
        n_nodes = len(self.S)
        n_sl = max(1, int(n_nodes * self.percentage))
        self.selected_nodes = np.argsort(self.S)[::-1][:n_sl]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

def bjontegaard_metric(R1, PSNR1, R2, PSNR2, mode='rate'):
    lR1, lR2 = np.log10(R1), np.log10(R2)
    # Ensure PSNR is monotonically increasing for BD-Rate
    if mode == 'rate':
        idx = np.argsort(PSNR1)
        R1, PSNR1 = R1[idx], PSNR1[idx]
        idx = np.argsort(PSNR2)
        R2, PSNR2 = R2[idx], PSNR2[idx]
        lR1, lR2 = np.log10(R1), np.log10(R2)
        min_int, max_int = max(min(PSNR1), min(PSNR2)), min(max(PSNR1), max(PSNR2))
        f1, f2 = interp1d(PSNR1, lR1, kind='cubic'), interp1d(PSNR2, lR2, kind='cubic')
        return (10**((quad(f2, min_int, max_int)[0] - quad(f1, min_int, max_int)[0]) / (max_int - min_int)) - 1) * 100
    else:
        min_int, max_int = max(min(lR1), min(lR2)), min(max(lR1), max(lR2))
        f1, f2 = interp1d(lR1, PSNR1, kind='cubic'), interp1d(lR2, PSNR2, kind='cubic')
        return (quad(f2, min_int, max_int)[0] - quad(f1, min_int, max_int)[0]) / (max_int - min_int)

def run_b16_fix():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    bsize = 16
    q_steps = [24, 32, 48, 64, 80]
    
    # NEW INCLUSIVE RANGE
    percentages = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    weights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # Increase sample ratio for better precision
    sampler = Sampler(ratio=0.02, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    rd_struct = {"r": [], "d": []}
    rd_oracle = {"r": [], "d": []}
    chosen_p = []
    chosen_w = []

    for q in q_steps:
        decider._set_vars(q)
        t_v, t_r_s, t_d_s, t_r_o, t_d_o = 0, 0, 0, 0, 0
        
        for block in tqdm(sampled_blocks, desc=f"B16 Fix Q={q}"):
            block.init_data(pc.V, pc.A)
            nv = block.Vblock.shape[0]
            t_v += nv
            
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            _, coeffs_s = gft_computer(block, s_graph)
            cost_s, r_s, d_s = decider._RDcost(coeffs_s)
            t_r_s += r_s
            t_d_s += d_s
            
            best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0, 0
            for p in percentages:
                for w in weights:
                    p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                    p_graph.set_data(block.Vblock, block.Ablock)
                    _, coeffs_p = gft_computer(block, p_graph)
                    c, r, d = decider._RDcost(coeffs_p)
                    if c < best_c:
                        best_c, best_r, best_d, best_p, best_w = c, r, d, p, w
            
            t_r_o += best_r
            t_d_o += best_d
            if best_p > 0:
                chosen_p.append(best_p)
                chosen_w.append(best_w)
            block.clear_data()

        rd_struct["r"].append(t_r_s / t_v)
        rd_struct["d"].append(20 * np.log10(255 / (t_d_s / t_v)))
        rd_oracle["r"].append(t_r_o / t_v)
        rd_oracle["d"].append(20 * np.log10(255 / (t_d_o / t_v)))

    # BD Metrics
    R_s, D_s = np.array(rd_struct["r"]), np.array(rd_struct["d"])
    R_o, D_o = np.array(rd_oracle["r"]), np.array(rd_oracle["d"])
    bd_rate = bjontegaard_metric(R_s, D_s, R_o, D_o, mode='rate')
    bd_psnr = bjontegaard_metric(R_s, D_s, R_o, D_o, mode='psnr')

    print(f"\n--- B16 FIXED RESULTS ---")
    print(f"BD-Rate:  {bd_rate:.4f}%")
    print(f"BD-PSNR:  {bd_psnr:.4f} dB")

    # Final Joint Distribution Heatmap
    plt.figure(figsize=(10, 8))
    hist, xedges, yedges = np.histogram2d(chosen_p, chosen_w, bins=[len(percentages), len(weights)])
    sns.heatmap(hist.T, annot=True, fmt=".0f", 
                xticklabels=[f"{p*100:.0f}%" for p in percentages], 
                yticklabels=[f"{w:.1f}" for w in weights])
    plt.title("Joint Distribution of Chosen P and W (B=16 Oracle)")
    plt.xlabel("Percentage of Self-Loops")
    plt.ylabel("Self-Loop Weight")
    plt.savefig("logs/b16_joint_distribution.png")
    print("\nJoint distribution heatmap saved to logs/b16_joint_distribution.png")

if __name__ == "__main__":
    run_b16_fix()
