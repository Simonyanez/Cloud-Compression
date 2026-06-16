import sys
import os
import json
import numpy as np
import zlib
import lzma
from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block

def _compute_all_costs_worker(metadata, Vblock, Ablock, slopes, slw_vals, slp_vals, q_step, lagrange_proportional, structural_coeffs, decider_mode):
    os.environ["OMP_NUM_THREADS"] = "1"
    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    block = Block(metadata)
    block.Vblock, block.Ablock = Vblock, Ablock
    cost_s, r_s, d_s = decider._RDcost(structural_coeffs)
    costs = [cost_s]
    rates = [r_s]
    distortions = [d_s]
    for k, (slope, slw, slp) in enumerate(zip(slopes, slw_vals, slp_vals)):
        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, slope, k+1, slp, slw)
        V_norm = Approximator()._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_norm @ slope.T
        a_graph.set_data(Vblock, A_app)
        _, coeffs = gft_computer(block, a_graph)
        c, r, d = decider._RDcost(coeffs)
        costs.append(c)
        rates.append(r)
        distortions.append(d)
    return costs, rates, distortions

def main():
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--q", type=int, default=24)
    parser.add_argument("--codebook_q", type=int, default=None, help="Q step of the codebook to load")
    parser.add_argument("--betas", type=str, default="0,5,20,100,500")
    parser.add_argument("--sweep_folder", type=str, required=True)
    args = parser.parse_args()

    cb_q = args.codebook_q if args.codebook_q is not None else args.q
    sweep_path = Path(args.sweep_folder)
    res_file = sweep_path / f"result_B{args.block_size}_C{args.clusters}_Q{cb_q}.json"
    
    with open(res_file, 'r') as f:
        cb = json.load(f)
    
    full_slopes = np.array(cb['final_slopes'])
    full_slw = np.array(cb['final_slw'])
    full_slp = np.array(cb['final_slp'])
    num_adaptive = len(full_slopes) - 1
    slopes, slw, slp = full_slopes[1:], full_slw[1:], full_slp[1:]

    params = load_experiment_config(Path("config/base_config.yaml"))
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    total_points = pc.V.shape[0]

    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=args.block_size)
    
    gft_computer = GFTStrategyWraper()
    s_coeffs = {}
    for b in tqdm(all_blocks, desc="Structural Precompute", leave=False):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata)
        sg.set_data(b.Vblock)
        dummy = Block(b.metadata); dummy.Vblock, dummy.Ablock = b.Vblock, b.Ablock
        _, c = gft_computer(dummy, sg)
        s_coeffs[b.block_id] = c
        b.clear_data()

    tasks = []
    for b in all_blocks:
        b.init_data(pc.V, pc.A)
        tasks.append((b.metadata, b.Vblock.copy(), b.Ablock.copy(), slopes, slw, slp, args.q, 
                      params.sequential_params.lagrange_proportional, s_coeffs[b.block_id], 
                      params.sequential_params.decider_mode))
        b.clear_data()

    results = Parallel(n_jobs=-1)(delayed(_compute_all_costs_worker)(*t) for t in tqdm(tasks, desc="Parallel Costs", leave=False))
    
    num_blocks = len(all_blocks)
    num_c = num_adaptive + 1
    M_costs = np.zeros((num_blocks, num_c))
    M_rates = np.zeros((num_blocks, num_c))
    M_dists = np.zeros((num_blocks, num_c))
    for i, (c, r, d) in enumerate(results):
        M_costs[i, :] = c; M_rates[i, :] = r; M_dists[i, :] = d

    base_bpv = sum(M_rates[:, 0]) / total_points
    base_psnr = 20 * np.log10(255 / np.sqrt(sum(M_dists[:, 0]) / total_points))

    print(f"\n### MULTI-BETA SWEEP: B{args.block_size} C{args.clusters} Q{args.q} ###")
    print(f"Baseline: {base_psnr:.2f} dB @ {base_bpv:.6f} bpv")
    print("-" * 100)
    print(f"| Beta | PSNR (dB) | Data bpv | LZMA bpv | Total bpv | Net Gain % | bits/block |")
    print(f"|------|-----------|----------|----------|-----------|------------|------------|")

    for beta in map(float, args.betas.split(',')):
        labels = np.zeros(num_blocks, dtype=np.uint8)
        labels[0] = np.argmin(M_costs[0, :])
        for i in range(1, num_blocks):
            adj = M_costs[i, :].copy()
            mask = np.ones(num_c, dtype=bool)
            mask[labels[i-1]] = False
            adj[mask] += beta
            labels[i] = np.argmin(adj)

        total_data_bits = sum(M_rates[i, labels[i]] for i in range(num_blocks))
        total_dist = sum(M_dists[i, labels[i]] for i in range(num_blocks))
        psnr = 20 * np.log10(255 / np.sqrt(total_dist / total_points))
        l_bits = len(lzma.compress(labels.tobytes())) * 8
        total_bpv = (total_data_bits + l_bits) / total_points
        gain = (total_bpv - base_bpv) / base_bpv * 100
        print(f"| {beta:4.0f} | {psnr:9.2f} | {total_data_bits/total_points:8.6f} | {l_bits/total_points:8.6f} | {total_bpv:9.6f} | {gain:+10.2f}% | {l_bits/num_blocks:10.4f} |")

if __name__ == "__main__":
    main()
