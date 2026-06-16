import sys
import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict, Tuple
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

class DummyBlock:
    """Minimal wrapper for GFT strategy that satisfies Block protocol."""
    def __init__(self, Vblock, Ablock, metadata=None):
        self.Vblock = Vblock
        self.Ablock = Ablock
        self.metadata = metadata
    def get_data(self):
        return self.Vblock, self.Ablock
    def get_absolute_idx(self, sub_idx):
        return self.metadata.get_absolute_idx(sub_idx)

def _evaluate_block_fixed_codebook(block_id, Vblock, Ablock, metadata, slopes, 
                                   slw_vals, slp_vals, q_step, lagrange_proportional,
                                   structural_coeffs, decider_mode):
    """Worker for fixed codebook assignment."""
    os.environ["OMP_NUM_THREADS"] = "1"
    
    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    
    # We use a real Block object or equivalent that supports metadata
    from pcadc.blocks import Block
    block = Block(metadata)
    block.Vblock = Vblock
    block.Ablock = Ablock
    
    # 0. Structural Cost (Baseline)
    cost_s, r_s, d_s = decider._RDcost(structural_coeffs)
    
    costs = [cost_s]
    rates = [r_s]
    distortions = [d_s]
    
    # 1. Test every cluster in codebook
    for k, (slope, slw, slp) in enumerate(zip(slopes, slw_vals, slp_vals)):
        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        
        a_graph = AttributeGraph(s_graph, slope, k+1, slp, slw)
        
        V_rot = Approximator()._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ slope.T
        a_graph.set_data(Vblock, A_app)
        
        _, coeffs = gft_computer(block, a_graph)
        
        c, r, d = decider._RDcost(coeffs)
        costs.append(c)
        rates.append(r)
        distortions.append(d)
        
    best_idx = np.argmin(costs)
    return best_idx, costs[best_idx], rates[best_idx], distortions[best_idx]

def load_codebook(result_folder: Path, b: int, c: int, q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    res_file = result_folder / f"result_B{b}_C{c}_Q{q}.json"
    if not res_file.exists():
        raise FileNotFoundError(f"Result file {res_file} not found.")
    with open(res_file, 'r') as f:
        data = json.load(f)
    return (np.array(data['final_slopes']), 
            np.array(data['final_slw']), 
            np.array(data['final_slp']))

def main():
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--codebook_q", type=int, default=24)
    parser.add_argument("--sweep_folder", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="fixed_codebook_full_results")
    args = parser.parse_args()

    sweep_path = Path(args.sweep_folder)
    base_out_dir = Path(args.out_dir)
    exp_out_dir = base_out_dir / f"B{args.block_size}_C{args.clusters}"
    exp_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Persistence: Check if all results are already computed
    results_file = exp_out_dir / "results.json"
    existing_results = []
    if results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
    
    existing_qs = [r['q'] for r in existing_results]
    q_steps = [12, 24, 36, 44, 48, 64]
    remaining_qs = [q for q in q_steps if q not in existing_qs]

    if not remaining_qs:
        print(f"[SKIP] B{args.block_size} C{args.clusters} already completed.")
        return

    # Load parameters
    params = load_experiment_config(Path("config/base_config.yaml"))
    
    # Load Codebook
    full_slopes, full_slw, full_slp = load_codebook(sweep_path, args.block_size, args.clusters, args.codebook_q)
    adaptive_slopes = full_slopes[1:]
    adaptive_slw = full_slw[1:]
    adaptive_slp = full_slp[1:]
    
    # Load Point Cloud (FULL)
    print(f"[*] Loading point cloud for B={args.block_size} C={args.clusters}...")
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=args.block_size)
    
    # Precompute vertices
    total_v = pc.V.shape[0] 
    
    # Precompute structural coeffs
    gft_computer = GFTStrategyWraper()
    print("[*] Precomputing structural coefficients for all blocks...")
    structural_coeffs_map = {}
    for b in tqdm(all_blocks, desc="Structural Precompute"):
        b.init_data(pc.V, pc.A)
        s_graph = StructuralGraph(b.metadata)
        s_graph.set_data(b.Vblock)
        _, coeffs = gft_computer(b, s_graph)
        structural_coeffs_map[b.block_id] = coeffs
        b.clear_data()

    results_all_q = existing_results
    
    for q in remaining_qs:
        print(f"\n[*] Evaluating Q={q} with fixed codebook (FULL PC)...")
        
        tasks = []
        for b in all_blocks:
            b.init_data(pc.V, pc.A)
            tasks.append((
                b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                adaptive_slopes, adaptive_slw, adaptive_slp,
                q, params.sequential_params.lagrange_proportional,
                structural_coeffs_map[b.block_id], params.sequential_params.decider_mode
            ))
            b.clear_data()
            
        eval_results = Parallel(n_jobs=-1)(
            delayed(_evaluate_block_fixed_codebook)(*t) for t in tqdm(tasks, desc=f"Q={q} Full")
        )
        
        labels = np.array([r[0] for r in eval_results])
        rates = np.array([r[2] for r in eval_results])
        distortions = np.array([r[3] for r in eval_results])
        
        unique, counts = np.unique(labels, return_counts=True)
        dist_dict = {int(k): int(v) for k, v in zip(unique, counts)}
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        
        overhead_bpv = (entropy * len(all_blocks)) / total_v
        total_bits = np.sum(rates)
        total_dist = np.sum(distortions)
        
        final_bpv = (total_bits + (entropy * len(all_blocks))) / total_v
        final_psnr = 20 * np.log10(255 / np.sqrt(total_dist / total_v)) if total_dist > 0 else 0
        
        results_all_q.append({
            "q": q,
            "entropy": float(entropy),
            "overhead_bpv": float(overhead_bpv),
            "avg_bpv_no_overhead": float(total_bits / total_v),
            "total_bpv": float(final_bpv),
            "psnr": float(final_psnr),
            "distribution": dist_dict
        })
        
        # Save after every Q step for safety
        with open(results_file, 'w') as f:
            json.dump(results_all_q, f, indent=4)

    print(f"\n[+] Completed B{args.block_size}_C{args.clusters}. Results in {exp_out_dir}")

if __name__ == "__main__":
    main()
