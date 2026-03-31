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

def run_debug_experiment():
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
    
    sampler = Sampler(ratio=0.01, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    decider = Decider(mode="0", lagrange_proportional=0.8)
    
    q_steps = [24, 32, 48, 64]
    
    results = []
    
    for q in q_steps:
        decider._set_vars(q)
        total_rd_s = 0
        total_rd_a = 0
        total_rd_a_sse = 0
        
        # Test for a fixed T and W that seemed good
        T, W = 0.8, 1.6
        
        for block in tqdm(sampled_blocks, desc=f"Testing Q={q}"):
            block.init_data(pc.V, pc.A)
            
            # Structural
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            _, coeffs_s = gft_computer(block, s_graph)
            
            # Attribute
            attr_graph = AttributeGraph(s_graph, np.zeros(3), 1, T, W)
            attr_graph.set_data(block.Vblock, block.Ablock)
            _, coeffs_a = gft_computer(block, attr_graph)
            
            # Standard RD cost (current implementation)
            rd_s, _, _ = decider._RDcost(coeffs_s)
            rd_a, _, _ = decider._RDcost(coeffs_a)
            
            # Energy Compaction (Energy in first 10% of coeffs)
            def energy_compaction(c):
                c_lum = c[:, 0]
                energy = c_lum**2
                sorted_energy = np.sort(energy)[::-1]
                k = max(1, int(len(sorted_energy) * 0.1))
                return np.sum(sorted_energy[:k]) / np.sum(energy) if np.sum(energy) > 0 else 1.0

            # Oracle choice for standard RD
            total_rd_s += rd_s
            total_rd_a += min(rd_s, rd_a)
            
            block.clear_data()
            
        gain = (total_rd_s - total_rd_a) / total_rd_s * 100
        print(f"Q={q} | Gain: {gain:.4f}%")
        results.append(gain)

    print("\nFinal Results for T=0.8, W=1.6:")
    for i, q in enumerate(q_steps):
        print(f"  Q={q}: Gain={results[i]:.4f}%")

if __name__ == "__main__":
    run_debug_experiment()
