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
        """Select top % nodes based on sink vector S."""
        n_nodes = len(self.S)
        n_sl = int(round(n_nodes * self.percentage))
        if n_sl < 1 and self.percentage > 0:
            n_sl = 1
        
        if n_sl == 0:
            return # No self-loops to add

        self.selected_nodes = np.argsort(self.S)[::-1][:n_sl]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

def bjontegaard_metric(R1, PSNR1, R2, PSNR2, mode='rate'):
    lR1, lR2 = np.log10(R1), np.log10(R2)
    # Sort by PSNR for rate metric
    if mode == 'rate':
        idx1, idx2 = np.argsort(PSNR1), np.argsort(PSNR2)
        r1, p1 = lR1[idx1], PSNR1[idx1]
        r2, p2 = lR2[idx2], PSNR2[idx2]
        min_int, max_int = max(min(p1), min(p2)), min(max(p1), max(p2))
        f1, f2 = interp1d(p1, r1, kind='cubic'), interp1d(p2, r2, kind='cubic')
        return (10**((quad(f2, min_int, max_int)[0] - quad(f1, min_int, max_int)[0]) / (max_int - min_int)) - 1) * 100
    else:
        min_int, max_int = max(min(lR1), min(lR2)), min(max(lR1), max(lR2))
        f1, f2 = interp1d(lR1, PSNR1, kind='cubic'), interp1d(lR2, PSNR2, kind='cubic')
        return (quad(f2, min_int, max_int)[0] - quad(f1, min_int, max_int)[0]) / (max_int - min_int)

def run_experiment():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64, 80]
    
    percentages = np.arange(0.0, 1.025, 0.025)
    weights = np.arange(0.5, 4.5, 0.5)
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    print(f"\n{'='*90}")
    print(f"{'BSize':<6} | {'BD-Rate %':<12} | {'BD-PSNR dB':<12} | {'Hybrid Selection %':<20}")
    print(f"{'-'*90}")

    for bsize in block_sizes:
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        # Sampling adjusted for speed
        sample_ratio = 0.0015 if bsize == 4 else 0.004 if bsize == 8 else 0.01
        sampler = Sampler(ratio=sample_ratio, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        rd_struct = {"r": [], "d": []}
        rd_oracle = {"r": [], "d": []}
        chosen_p, chosen_w = [], []
        total_blocks_count = 0
        hybrid_chosen_count = 0

        for q in q_steps:
            decider._set_vars(q)
            t_v, t_r_s, t_d_s, t_r_o, t_d_o = 0, 0, 0, 0, 0
            
            for block in tqdm(sampled_blocks, desc=f"B={bsize}, Q={q}", leave=False):
                block.init_data(pc.V, pc.A)
                nv = block.Vblock.shape[0]
                t_v += nv
                total_blocks_count += 1
                
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                _, coeffs_s = gft_computer(block, s_graph)
                cost_s, r_s, d_s = decider._RDcost(coeffs_s)
                t_r_s += r_s
                t_d_s += d_s
                
                best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0.0, 0.0

                for p in percentages:
                    if p == 0: continue # p=0 is structural
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
                    hybrid_chosen_count += 1
                
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
        hyb_perc = (hybrid_chosen_count / total_blocks_count) * 100
        print(f"{bsize:<6} | {bd_rate:>12.4f}% | {bd_psnr:>12.4f} dB | {hyb_perc:>18.2f}%")

        # Distribution Heatmap
        plt.figure(figsize=(12, 10))
        hist, xedges, yedges = np.histogram2d(chosen_p, chosen_w, 
                                             bins=[percentages[percentages>0], weights])
        sns.heatmap(hist.T, annot=True, fmt=".0f", 
                    xticklabels=[f"{p*100:.1f}%" for p in percentages[1:]], 
                    yticklabels=[f"{w:.1f}" for w in weights])
        plt.title(f"Joint Distribution of Chosen P and W (B={bsize} Oracle)")
        plt.xlabel("Percentage of Self-Loops")
        plt.ylabel("Self-Loop Weight")
        plt.savefig(f"logs/ultra_granular_B{bsize}_joint.png")
        plt.close()

        # RD Curve for this block size
        plt.figure(figsize=(10, 6))
        plt.plot(rd_struct["r"], rd_struct["d"], 'o-', label='Structural Baseline')
        plt.plot(rd_oracle["r"], rd_oracle["d"], 'd--', label='Ultra Oracle (Percentage)', color='red')
        for j, q in enumerate(q_steps):
            plt.text(rd_oracle["r"][j], rd_oracle["d"][j], f"Q={q}", verticalalignment='bottom')
        plt.title(f"RD Curve B={bsize} (Ultra Granular Oracle)")
        plt.xlabel("Rate (Non-zero / Vertex)")
        plt.ylabel("PSNR (Approximated)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"logs/ultra_granular_B{bsize}_rd.png")
        plt.close()

    print(f"\nAll results (heatmaps and RD curves) saved to logs/ultra_granular_B*")

if __name__ == "__main__":
    run_experiment()
