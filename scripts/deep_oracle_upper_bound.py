import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
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
        n_sl = int(round(n_nodes * self.percentage))
        if n_sl < 1 and self.percentage > 0: n_sl = 1
        if n_sl == 0: return
        self.selected_nodes = np.argsort(self.S)[::-1][:n_sl]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

def bjontegaard_metric(R1, PSNR1, R2, PSNR2):
    """Calculates BD-Rate."""
    lR1, lR2 = np.log10(R1), np.log10(R2)
    idx1, idx2 = np.argsort(PSNR1), np.argsort(PSNR2)
    f1, f2 = interp1d(PSNR1[idx1], lR1[idx1], kind='cubic'), interp1d(PSNR2[idx2], lR2[idx2], kind='cubic')
    min_p, max_p = max(min(PSNR1), min(PSNR2)), min(max(PSNR1), max(PSNR2))
    return (10**((quad(f2, min_p, max_p)[0] - quad(f1, min_p, max_p)[0]) / (max_p - min_p)) - 1) * 100

def optimize_block(args):
    """Worker function for parallel optimization of a single block."""
    Vblock, Ablock, metadata, q_step, lagrange_proportional = args
    
    decider = Decider(mode="0", lagrange_proportional=lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    
    # 1. Structural Baseline
    s_graph = StructuralGraph(metadata)
    s_graph.set_data(Vblock)
    _, coeffs_s = gft_computer(None, s_graph, Vblock, Ablock) # Manual GFT call
    cost_s, r_s, d_s = decider._RDcost(coeffs_s)
    
    best_c, best_r, best_d = cost_s, r_s, d_s
    
    # 2. Stage 1: Coarse Grid
    p_coarse = np.linspace(0.01, 0.4, 8)
    w_coarse = np.linspace(0.5, 3.5, 6)
    
    best_p_c, best_w_c = 0, 0
    for p in p_coarse:
        for w in w_coarse:
            p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
            p_graph.set_data(Vblock, Ablock)
            _, coeffs_p = gft_computer(None, p_graph, Vblock, Ablock)
            c, r, d = decider._RDcost(coeffs_p)
            if c < best_c:
                best_c, best_r, best_d, best_p_c, best_w_c = c, r, d, p, w
                
    # 3. Stage 2: Fine Refinement around winner (if hybrid won)
    if best_p_c > 0:
        p_fine = np.linspace(max(0.005, best_p_c - 0.05), min(0.5, best_p_c + 0.05), 5)
        w_fine = np.linspace(max(0.25, best_w_c - 0.4), min(4.0, best_w_c + 0.4), 5)
        for p in p_fine:
            for w in w_fine:
                p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                p_graph.set_data(Vblock, Ablock)
                _, coeffs_p = gft_computer(None, p_graph, Vblock, Ablock)
                c, r, d = decider._RDcost(coeffs_p)
                if c < best_c:
                    best_c, best_r, best_d = c, r, d

    return (r_s, d_s, best_r, best_d, Vblock.shape[0])

def run_deep_oracle():
    # Setup
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64, 80]
    
    final_results = {}

    for bsize in block_sizes:
        print(f"\n{'='*40}\nDEEP ORACLE SEARCH B={bsize}\n{'='*40}")
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        sampler = Sampler(ratio=0.005 if bsize==8 else 0.01, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        rates_s, dists_s, rates_o, dists_o = [], [], [], []
        
        for q in q_steps:
            print(f"Processing Q={q}...")
            # Prepare args for pool
            tasks = []
            for b in sampled_blocks:
                b.init_data(pc.V, pc.A)
                tasks.append((b.Vblock.copy(), b.Ablock.copy(), b.metadata, q, params.sequential_params.lagrange_proportional))
                b.clear_data()
            
            # Use all cores
            with Pool(cpu_count()) as pool:
                results = pool.map(optimize_block, tasks)
            
            # Aggregate
            total_v = sum(r[4] for r in results)
            total_r_s = sum(r[0] for r in results)
            total_d_s = sum(r[1] for r in results)
            total_r_o = sum(r[2] for r in results)
            total_d_o = sum(r[3] for r in results)
            
            rates_s.append(total_r_s / total_v)
            dists_s.append(20 * np.log10(255 / (total_d_s / total_v)))
            rates_o.append(total_r_o / total_v)
            dists_o.append(20 * np.log10(255 / (total_d_o / total_v)))

        bd_rate = bjontegaard_metric(np.array(rates_s), np.array(dists_s), np.array(rates_o), np.array(dists_o))
        print(f"\n>>> FINAL UPPER BOUND BD-RATE (B={bsize}): {bd_rate:.4f}%")
        
        final_results[bsize] = (rates_s, dists_s, rates_o, dists_o, bd_rate)

    # Final Summary Plot
    plt.figure(figsize=(15, 5))
    for i, bsize in enumerate(block_sizes):
        rs, ds, ro, do, bdr = final_results[bsize]
        plt.subplot(1, 3, i+1)
        plt.plot(rs, ds, 'o-', label='Structural')
        plt.plot(ro, do, 'd--', label='Deep Oracle', color='red')
        for j, q in enumerate(q_steps):
            plt.text(ro[j], do[j], f"Q={q}", fontsize=8)
        plt.title(f"B={bsize} | BD-Rate: {bdr:.2f}%")
        plt.xlabel("Rate")
        plt.ylabel("PSNR")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("logs/deep_oracle_upper_bound.png")
    print("\nResults saved to logs/deep_oracle_upper_bound.png")

if __name__ == "__main__":
    # Fix for gft_computer manual call: GFTStrategyWraper needs to be compatible
    # Overriding computer for manual use in optimize_block
    from pcadc.transforms import ConnectedGFTProcessor
    def manual_gft(self, block, graph, V, A):
        # Dummy block with data
        class Dummy: pass
        d = Dummy()
        d.get_data = lambda: (V, A)
        return ConnectedGFTProcessor().compute(d, graph)
    GFTStrategyWraper.__call__ = manual_gft
    
    run_deep_oracle()
