import sys
import os
import json
import numpy as np
import zlib
import lzma
import bz2
from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper

def _get_assignment(metadata, Vblock, Ablock, slopes, slw_vals, slp_vals, q_step, lagrange_proportional, structural_coeffs, decider_mode):
    os.environ["OMP_NUM_THREADS"] = "1"
    from pcadc.blocks import Block
    from pcadc.decider import Decider
    from pcadc.transforms import GFTStrategyWraper
    
    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    block = Block(metadata)
    block.Vblock, block.Ablock = Vblock, Ablock
    
    cost_s, _, _ = decider._RDcost(structural_coeffs)
    costs = [cost_s]
    
    for k, (slope, slw, slp) in enumerate(zip(slopes, slw_vals, slp_vals)):
        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, slope, k+1, slp, slw)
        
        V_rot = Approximator()._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ slope.T
        a_graph.set_data(Vblock, A_app)
        
        _, coeffs = gft_computer(block, a_graph)
        c, _, _ = decider._RDcost(coeffs)
        costs.append(c)
        
    return int(np.argmin(costs))

def main():
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--q", type=int, default=24)
    parser.add_argument("--sweep_folder", type=str, required=True)
    args = parser.parse_args()

    sweep_path = Path(args.sweep_folder)
    res_file = sweep_path / f"result_B{args.block_size}_C{args.clusters}_Q{args.q}.json"
    
    with open(res_file, 'r') as f:
        cb = json.load(f)
    
    slopes = np.array(cb['final_slopes'])[1:]
    slw = np.array(cb['final_slw'])[1:]
    slp = np.array(cb['final_slp'])[1:]

    params = load_experiment_config(Path("config/base_config.yaml"))
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=args.block_size)
    
    # Precompute structural
    gft_computer = GFTStrategyWraper()
    print("[*] Precomputing structural coeffs...")
    s_coeffs = {}
    for b in tqdm(all_blocks):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata)
        sg.set_data(b.Vblock)
        # Mocking block for GFT wrapper
        from pcadc.blocks import Block
        dummy = Block(b.metadata)
        dummy.Vblock, dummy.Ablock = b.Vblock, b.Ablock
        _, c = gft_computer(dummy, sg)
        s_coeffs[b.block_id] = c
        b.clear_data()

    print(f"[*] Extracting labels for {len(all_blocks)} blocks...")
    tasks = []
    for b in all_blocks:
        b.init_data(pc.V, pc.A)
        tasks.append((b.metadata, b.Vblock.copy(), b.Ablock.copy(), slopes, slw, slp, args.q, 
                      params.sequential_params.lagrange_proportional, s_coeffs[b.block_id], 
                      params.sequential_params.decider_mode))
        b.clear_data()

    labels = Parallel(n_jobs=-1)(delayed(_get_assignment)(*t) for t in tqdm(tasks))
    label_bytes = bytes(labels)
    
    # Benchmarking
    total_points = pc.V.shape[0]
    
    # 1. Entropy (Ideal)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    entropy_bits = entropy * len(labels)
    
    # 2. ZLIB (Deflate)
    z_data = zlib.compress(label_bytes, level=9)
    z_bits = len(z_data) * 8
    
    # 3. LZMA
    l_data = lzma.compress(label_bytes)
    l_bits = len(l_data) * 8
    
    # 4. BZ2
    b_data = bz2.compress(label_bytes)
    b_bits = len(b_data) * 8

    print(f"\n### ENCODER BENCHMARK (B={args.block_size}, C={args.clusters}, Q={args.q}) ###")
    print(f"Total Blocks: {len(labels)}")
    print(f"Total Points: {total_points}")
    print("-" * 50)
    print(f"| Method   | Size (Bytes) | bpv Overhead | bits/block |")
    print(f"|----------|--------------|--------------|------------|")
    print(f"| Entropy  | {entropy_bits/8:12.1f} | {entropy_bits/total_points:12.6f} | {entropy:10.4f} |")
    print(f"| Zlib     | {len(z_data):12d} | {z_bits/total_points:12.6f} | {z_bits/len(labels):10.4f} |")
    print(f"| BZ2      | {len(b_data):12d} | {b_bits/total_points:12.6f} | {b_bits/len(labels):10.4f} |")
    print(f"| LZMA     | {len(l_data):12d} | {l_bits/total_points:12.6f} | {l_bits/len(labels):10.4f} |")

if __name__ == "__main__":
    main()
