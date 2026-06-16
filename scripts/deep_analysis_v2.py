import sys
import os
import json
import numpy as np
import zlib
import lzma
from pathlib import Path
import matplotlib.pyplot as plt
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

def _eval_block(metadata, Vblock, Ablock, slopes, slw_vals, slp_vals, q_step, lagrange_proportional, structural_coeffs, decider_mode, beta):
    os.environ["OMP_NUM_THREADS"] = "1"
    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    block = Block(metadata)
    block.Vblock, block.Ablock = Vblock, Ablock
    
    costs, rates, dists = [], [], []
    # Structural (Index 0)
    c_s, r_s, d_s = decider._RDcost(structural_coeffs)
    costs.append(c_s); rates.append(r_s); dists.append(d_s)
    
    # Adaptive
    for k, slope in enumerate(slopes[1:]):
        slw, slp = slw_vals[k+1], slp_vals[k+1]
        s_graph = StructuralGraph(metadata); s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, slope, k+1, slp, slw)
        V_rot = Approximator()._spatial_norm(Vblock)
        A_app = Ablock.copy(); A_app[:, 0] = V_rot @ slope.T
        a_graph.set_data(Vblock, A_app)
        _, coeffs = gft_computer(block, a_graph)
        c, r, d = decider._RDcost(coeffs)
        costs.append(c); rates.append(r); dists.append(d)
        
    return costs, rates, dists

def analyze_sweep(folder):
    results_dir = Path(folder)
    json_files = list(results_dir.glob("result_*.json"))
    
    params = load_experiment_config(Path("config/base_config.yaml"))
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    total_points = pc.V.shape[0]

    report = []

    # Cache for structural baseline per (B, L)
    baseline_cache = {}

    for jf in sorted(json_files):
        with open(jf, 'r') as f:
            d = json.load(f)
        
        code = d['experiment_code']
        # B16_C2_L5.0_Beta2000.0
        parts = code.split('_')
        b_val = int(parts[0][1:])
        l_val = float(parts[2][1:])
        beta = float(parts[3][4:])
        
        print(f"\n[*] Deep Analysis of {code}...")
        
        # Partition
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=b_val)
        
        # Precompute structural
        gft_computer = GFTStrategyWraper()
        s_coeffs = {}
        for b in tqdm(all_blocks, desc=f"Structural Precompute B{b_val}", leave=False):
            b.init_data(pc.V, pc.A)
            sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
            dummy = Block(b.metadata); dummy.Vblock, dummy.Ablock = b.Vblock, b.Ablock
            _, c = gft_computer(dummy, sg)
            s_coeffs[b.block_id] = c
            b.clear_data()

        # Full Assignment with Beta
        tasks = []
        for b in all_blocks:
            b.init_data(pc.V, pc.A)
            tasks.append((b.metadata, b.Vblock.copy(), b.Ablock.copy(), 
                          np.array(d['final_slopes']), np.array(d['final_slw']), np.array(d['final_slp']),
                          24, l_val, s_coeffs[b.block_id], "0", beta))
            b.clear_data()

        res = Parallel(n_jobs=-1)(delayed(_eval_block)(*t) for t in tqdm(tasks, desc="Evaluating Blocks", leave=False))
        
        num_blocks = len(all_blocks)
        num_c = len(d['final_slopes'])
        M_costs = np.zeros((num_blocks, num_c))
        M_rates = np.zeros((num_blocks, num_c))
        M_dists = np.zeros((num_blocks, num_c))
        for i, (cs, rs, ds) in enumerate(res):
            M_costs[i, :] = cs; M_rates[i, :] = rs; M_dists[i, :] = ds

        # Sequential greedy
        labels = np.zeros(num_blocks, dtype=np.uint8)
        labels[0] = np.argmin(M_costs[0, :])
        for i in range(1, num_blocks):
            adj = M_costs[i, :].copy()
            mask = np.ones(num_c, dtype=bool)
            mask[labels[i-1]] = False
            adj[mask] += beta
            labels[i] = np.argmin(adj)

        data_bits = sum(M_rates[i, labels[i]] for i in range(num_blocks))
        total_dist = sum(M_dists[i, labels[i]] for i in range(num_blocks))
        psnr = 20 * np.log10(255 / np.sqrt(total_dist / total_points))
        l_bits = len(lzma.compress(labels.tobytes())) * 8
        
        # Baseline
        base_bits = sum(M_rates[:, 0])
        base_dist = sum(M_dists[:, 0])
        base_psnr = 20 * np.log10(255 / np.sqrt(base_dist / total_points))
        base_bpv = base_bits / total_points
        
        total_bpv = (data_bits + l_bits) / total_points
        gain = (total_bpv - base_bpv) / base_bpv * 100
        
        report.append({
            "code": code,
            "psnr": psnr,
            "data_bpv": data_bits/total_points,
            "overhead_bpv": l_bits/total_points,
            "total_bpv": total_bpv,
            "gain": gain,
            "base_psnr": base_psnr,
            "base_bpv": base_bpv
        })

    # Print Final Table
    print("\n" + "="*80)
    print("   DEEP ANALYSIS: COMPRESSION-FOCUSED SWEEP (FULL POINT CLOUD)")
    print("="*80)
    print(f"| {'Config':30} | PSNR | bpv | Net Gain | LZMA/Total |")
    print(f"|{'-'*32}|{'-'*6}|{'-'*7}|{'-'*10}|{'-'*12}|")
    for r in report:
        lzma_ratio = (r['overhead_bpv'] / r['total_bpv']) * 100
        print(f"| {r['code']:30} | {r['psnr']:5.2f} | {r['total_bpv']:.4f} | {r['gain']:+8.2f}% | {lzma_ratio:9.1f}% |")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_sweep(sys.argv[1])
    else:
        print("Usage: python scripts/deep_analysis_v2.py <folder>")
