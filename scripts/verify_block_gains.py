import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

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

def get_numeric_gains():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [32, 48, 64] # Focused range
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    print(f"\n{'='*60}")
    print(f"{'Block Size':<12} | {'Q':<5} | {'Ultra-Adaptive Gain (%)':<25}")
    print(f"{'-'*60}")

    for bsize in block_sizes:
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        sample_ratio = 0.005 if bsize == 4 else 0.01 if bsize == 8 else 0.02
        sampler = Sampler(ratio=sample_ratio, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        for q in q_steps:
            decider._set_vars(q)
            total_cost_s = 0
            total_cost_ultra = 0
            
            for block in tqdm(sampled_blocks, desc=f"B={bsize}, Q={q}", leave=False):
                block.init_data(pc.V, pc.A)
                
                # Structural
                s_graph = StructuralGraph(block.metadata)
                s_graph.set_data(block.Vblock)
                _, coeffs_s = gft_computer(block, s_graph)
                cost_s, _, _ = decider._RDcost(coeffs_s)
                total_cost_s += cost_s
                
                # Local Adaptive Oracle
                min_cost_h = float('inf')
                for t in thresholds:
                    for w in weights:
                        attr_graph = AttributeGraph(s_graph, np.zeros(3), 1, t, w)
                        attr_graph.set_data(block.Vblock, block.Ablock)
                        _, coeffs_a = gft_computer(block, attr_graph)
                        cost_a, _, _ = decider._RDcost(coeffs_a)
                        if cost_a < min_cost_h:
                            min_cost_h = cost_a
                
                total_cost_ultra += min(cost_s, min_cost_h)
                block.clear_data()
            
            gain = (total_cost_s - total_cost_ultra) / total_cost_s * 100
            print(f"{bsize:<12} | {q:<5} | {gain:24.4f}%")

if __name__ == "__main__":
    get_numeric_gains()
