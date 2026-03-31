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

def get_rd_points(bsize, q_steps, best_params):
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # Sample 1% for speed
    sampler = Sampler(ratio=0.01, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    t_sl, w_sl = best_params
    
    rates_s = []
    dist_s = []
    rates_h = []
    dist_h = []

    decider = Decider(mode="0", lagrange_proportional=0.8)

    for q in q_steps:
        decider._set_vars(q)
        total_r_s = 0
        total_d_s = 0
        total_r_h = 0
        total_d_h = 0
        num_v = 0

        for block in sampled_blocks:
            block.init_data(pc.V, pc.A)
            num_v += block.Vblock.shape[0]
            
            # Structural
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            _, coeffs_s = gft_computer(block, s_graph)
            _, r_s, d_s = decider._RDcost(coeffs_s)
            
            # Hybrid (Oracle logic)
            attr_graph = AttributeGraph(s_graph, np.zeros(3), 1, t_sl, w_sl)
            attr_graph.set_data(block.Vblock, block.Ablock)
            _, coeffs_a = gft_computer(block, attr_graph)
            cost_a, r_a, d_a = decider._RDcost(coeffs_a)
            cost_s, _, _ = decider._RDcost(coeffs_s)
            
            total_r_s += r_s
            total_d_s += d_s
            
            if cost_a < cost_s:
                total_r_h += r_a
                total_d_h += d_a
            else:
                total_r_h += r_s
                total_d_h += d_s
                
            block.clear_data()
            
        rates_s.append(total_r_s / num_v)
        dist_s.append(20 * np.log10(255 / (total_d_s / num_v))) # PSNR-like
        rates_h.append(total_r_h / num_v)
        dist_h.append(20 * np.log10(255 / (total_d_h / num_v)))

    return rates_s, dist_s, rates_h, dist_h

def main():
    q_steps = [24, 32, 48, 64, 80]
    # Best params from previous oracle run
    # B4: T=0.29, W=0.42 | B8: T=0.86, W=1.39 | B16: T=0.95, W=1.39
    b_configs = {
        4: (0.29, 0.42),
        8: (0.86, 1.39),
        16: (0.95, 1.39)
    }

    plt.figure(figsize=(15, 5))

    for i, bsize in enumerate([4, 8, 16]):
        print(f"Processing B={bsize}...")
        rs, ds, rh, dh = get_rd_points(bsize, q_steps, b_configs[bsize])
        
        plt.subplot(1, 3, i+1)
        plt.plot(rs, ds, 'o-', label='Structural')
        plt.plot(rh, dh, 's--', label='Hybrid (Oracle)')
        plt.title(f"RD Curve B={bsize}")
        plt.xlabel("Rate (Non-zero coeffs / Vertex)")
        plt.ylabel("PSNR (Approximated)")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig("logs/rd_curves_sampled.png")
    print("Plot saved to logs/rd_curves_sampled.png")

if __name__ == "__main__":
    main()
