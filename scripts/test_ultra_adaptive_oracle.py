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

def run_ultra_adaptive_experiment():
    # 1. Setup
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64, 80]
    thresholds = np.linspace(0.1, 0.9, 5) # Coarse sweep for speed
    weights = np.linspace(0.5, 2.5, 5)
    
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
        
        rates_struct = []
        psnr_struct = []
        rates_oracle_simple = []
        psnr_oracle_simple = []
        rates_oracle_local = []
        psnr_oracle_local = []

        for q in q_steps:
            decider._set_vars(q)
            print(f"  Testing Q={q}...")
            
            # Data storage for this Q
            n_blocks = len(sampled_blocks)
            block_costs_struct = np.zeros(n_blocks)
            block_rates_struct = np.zeros(n_blocks)
            block_dist_struct = np.zeros(n_blocks)
            
            # sweep_results[block_idx][t_idx, w_idx] = (cost, rate, dist)
            sweep_results = np.zeros((n_blocks, len(thresholds), len(weights), 3))
            
            total_v = 0
            
            for b_idx, block in enumerate(tqdm(sampled_blocks, desc=f"Sweeping Blocks (Q={q})", leave=False)):
                block.init_data(pc.V, pc.A)
                total_v += block.Vblock.shape[0]
                
                # 1. Structural Baseline
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                _, coeffs_s = gft_computer(block, s_graph)
                cost_s, rate_s, dist_s = decider._RDcost(coeffs_s)
                block_costs_struct[b_idx] = cost_s
                block_rates_struct[b_idx] = rate_s
                block_dist_struct[b_idx] = dist_s
                
                # 2. Hybrid Sweep
                for ti, t in enumerate(thresholds):
                    for wi, w in enumerate(weights):
                        attr_graph = AttributeGraph(s_graph, np.zeros(3), 1, t, w)
                        attr_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_a = gft_computer(block, attr_graph)
                        cost_a, rate_a, dist_a = decider._RDcost(coeffs_a)
                        sweep_results[b_idx, ti, wi] = [cost_a, rate_a, dist_a]
                
                block.clear_data()

            # --- CALCULATE RD POINTS ---
            
            # 1. Structural
            rates_struct.append(np.sum(block_rates_struct) / total_v)
            psnr_struct.append(20 * np.log10(255 / (np.sum(block_dist_struct) / total_v)))

            # 2. Oracle Local (Best config per block)
            total_r_local = 0
            total_d_local = 0
            for b_idx in range(n_blocks):
                best_hybrid_idx = np.unravel_index(np.argmin(sweep_results[b_idx, :, :, 0]), (len(thresholds), len(weights)))
                cost_h, rate_h, dist_h = sweep_results[b_idx][best_hybrid_idx]
                
                if cost_h < block_costs_struct[b_idx]:
                    total_r_local += rate_h
                    total_d_local += dist_h
                else:
                    total_r_local += block_rates_struct[b_idx]
                    total_d_local += block_dist_struct[b_idx]
            
            rates_oracle_local.append(total_r_local / total_v)
            psnr_oracle_local.append(20 * np.log10(255 / (total_d_local / total_v)))

            # 3. Oracle Simple (Best GLOBAL config)
            # Find ti, wi that minimizes sum(min(cost_s, cost_h(ti, wi)))
            global_costs = np.zeros((len(thresholds), len(weights)))
            for ti in range(len(thresholds)):
                for wi in range(len(weights)):
                    total_cost_tw = 0
                    for b_idx in range(n_blocks):
                        total_cost_tw += min(block_costs_struct[b_idx], sweep_results[b_idx, ti, wi, 0])
                    global_costs[ti, wi] = total_cost_tw
            
            best_global_idx = np.unravel_index(np.argmin(global_costs), global_costs.shape)
            ti_g, wi_g = best_global_idx
            
            total_r_simple = 0
            total_d_simple = 0
            for b_idx in range(n_blocks):
                cost_h, rate_h, dist_h = sweep_results[b_idx, ti_g, wi_g]
                if cost_h < block_costs_struct[b_idx]:
                    total_r_simple += rate_h
                    total_d_simple += dist_h
                else:
                    total_r_simple += block_rates_struct[b_idx]
                    total_d_simple += block_dist_struct[b_idx]
            
            rates_oracle_simple.append(total_r_simple / total_v)
            psnr_oracle_simple.append(20 * np.log10(255 / (total_d_simple / total_v)))

        # Plotting for this block size
        plt.subplot(1, 3, i+1)
        plt.plot(rates_struct, psnr_struct, 'o-', label='Structural Baseline', alpha=0.8)
        plt.plot(rates_oracle_simple, psnr_oracle_simple, 's--', label='Oracle Simple (Global Best T,W)', alpha=0.8)
        plt.plot(rates_oracle_local, psnr_oracle_local, 'd:', label='Oracle Ultra (Best T,W per Block)', color='red', linewidth=2)
        plt.title(f"RD Curve B={bsize}")
        plt.xlabel("Rate (Non-zero / Vertex)")
        plt.ylabel("PSNR (Approximated)")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig("logs/rd_ultra_adaptive.png")
    print("\nUltra-adaptive comparison plot saved to logs/rd_ultra_adaptive.png")

if __name__ == "__main__":
    run_ultra_adaptive_experiment()
