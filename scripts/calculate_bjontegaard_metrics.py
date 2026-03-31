import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
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
    """
    Calculates BD-PSNR (dB) or BD-Rate (%) between two RD curves.
    """
    # Log transform of rates
    lR1 = np.log10(R1)
    lR2 = np.log10(R2)

    # Use interpolation to find common overlap in PSNR or Rate
    if mode == 'psnr':
        # BD-PSNR: Average PSNR difference over a range of log-rates
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))
        
        f1 = interp1d(lR1, PSNR1, kind='cubic')
        f2 = interp1d(lR2, PSNR2, kind='cubic')
        
        int1 = quad(f1, min_int, max_int)[0]
        int2 = quad(f2, min_int, max_int)[0]
        
        return (int2 - int1) / (max_int - min_int)
    
    else:
        # BD-Rate: Average bitrate difference over a range of PSNRs
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))
        
        # We need monotonic PSNR for interpolation
        idx1 = np.argsort(PSNR1)
        idx2 = np.argsort(PSNR2)
        
        f1 = interp1d(PSNR1[idx1], lR1[idx1], kind='cubic')
        f2 = interp1d(PSNR2[idx2], lR2[idx2], kind='cubic')
        
        int1 = quad(f1, min_int, max_int)[0]
        int2 = quad(f2, min_int, max_int)[0]
        
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        return (10**avg_exp_diff - 1) * 100

def run_experiment():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64, 80]
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    percentages = [0.02, 0.05, 0.10, 0.20]
    weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    print(f"\n{'='*85}")
    print(f"{'BSize':<6} | {'Metric':<10} | {'Threshold vs Struct':<20} | {'Percentage vs Struct':<20}")
    print(f"{'-'*85}")

    for bsize in block_sizes:
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        sample_ratio = 0.003 if bsize == 4 else 0.008 if bsize == 8 else 0.015
        sampler = Sampler(ratio=sample_ratio, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        data = {"struct": {"r": [], "d": []}, "thresh": {"r": [], "d": []}, "perc": {"r": [], "d": []}}

        for q in q_steps:
            decider._set_vars(q)
            total_v = 0
            t_r_s, t_d_s, t_r_t, t_d_t, t_r_p, t_d_p = 0, 0, 0, 0, 0, 0
            
            for block in tqdm(sampled_blocks, desc=f"B={bsize}, Q={q}", leave=False):
                block.init_data(pc.V, pc.A)
                nv = block.Vblock.shape[0]
                total_v += nv
                
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                _, coeffs_s = gft_computer(block, s_graph)
                cost_s, r_s, d_s = decider._RDcost(coeffs_s)
                t_r_s += r_s
                t_d_s += d_s
                
                # Threshold Oracle
                bt_c, bt_r, bt_d = cost_s, r_s, d_s
                for t in thresholds:
                    for w in weights:
                        t_graph = AttributeGraph(s_graph, np.zeros(3), 1, t, w)
                        t_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_t = gft_computer(block, t_graph)
                        c, r, d = decider._RDcost(coeffs_t)
                        if c < bt_c: bt_c, bt_r, bt_d = c, r, d
                t_r_t += bt_r
                t_d_t += bt_d

                # Percentage Oracle
                bp_c, bp_r, bp_d = cost_s, r_s, d_s
                for p in percentages:
                    for w in weights:
                        p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                        p_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_p = gft_computer(block, p_graph)
                        c, r, d = decider._RDcost(coeffs_p)
                        if c < bp_c: bp_c, bp_r, bp_d = c, r, d
                t_r_p += bp_r
                t_d_p += bp_d
                
                block.clear_data()

            data["struct"]["r"].append(t_r_s / total_v)
            data["struct"]["d"].append(20 * np.log10(255 / (t_d_s / total_v)))
            data["thresh"]["r"].append(t_r_t / total_v)
            data["thresh"]["d"].append(20 * np.log10(255 / (t_d_t / total_v)))
            data["perc"]["r"].append(t_r_p / total_v)
            data["perc"]["d"].append(20 * np.log10(255 / (t_d_p / total_v)))

        # Calculate Bjontegaard Metrics
        R_s, D_s = np.array(data["struct"]["r"]), np.array(data["struct"]["d"])
        R_t, D_t = np.array(data["thresh"]["r"]), np.array(data["thresh"]["d"])
        R_p, D_p = np.array(data["perc"]["r"]), np.array(data["perc"]["d"])

        bd_rate_t = bjontegaard_metric(R_s, D_s, R_t, D_t, mode='rate')
        bd_psnr_t = bjontegaard_metric(R_s, D_s, R_t, D_t, mode='psnr')
        
        bd_rate_p = bjontegaard_metric(R_s, D_s, R_p, D_p, mode='rate')
        bd_psnr_p = bjontegaard_metric(R_s, D_s, R_p, D_p, mode='psnr')

        print(f"{bsize:<6} | {'BD-Rate %':<10} | {bd_rate_t:>19.4f}% | {bd_rate_p:>19.4f}%")
        print(f"{'':<6} | {'BD-PSNR dB':<10} | {bd_psnr_t:>19.4f}  | {bd_psnr_p:>19.4f} ")
        print(f"{'-'*85}")

if __name__ == "__main__":
    run_experiment()
