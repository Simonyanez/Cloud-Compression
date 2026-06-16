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

def _get_labels_worker(metadata, Vblock, Ablock, slopes, slw_vals, slp_vals, q_step, lagrange_proportional, structural_coeffs, decider_mode):
    os.environ["OMP_NUM_THREADS"] = "1"
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
    parser.add_argument("--configs", type=str, help="Comma separated B_C pairs, e.g. 4_12,8_8,16_8")
    parser.add_argument("--q", type=int, default=24)
    parser.add_argument("--sweep_folder", type=str, required=True)
    args = parser.parse_args()

    sweep_path = Path(args.sweep_folder)
    config_list = args.configs.split(',')
    
    params = load_experiment_config(Path("config/base_config.yaml"))
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    total_points = pc.V.shape[0]

    print(f"| Config | Entropy bpv | Zlib bpv | LZMA bpv | Savings vs Entropy |")
    print(f"|--------|-------------|----------|----------|--------------------|")

    for cfg in config_list:
        b_val, c_val = map(int, cfg.split('_'))
        
        # Load Codebook
        res_file = sweep_path / f"result_B{b_val}_C{c_val}_Q{args.q}.json"
        with open(res_file, 'r') as f:
            cb = json.load(f)
        
        slopes = np.array(cb['final_slopes'])[1:]
        slw = np.array(cb['final_slw'])[1:]
        slp = np.array(cb['final_slp'])[1:]

        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=b_val)
        
        # Precompute structural (required for comparison)
        gft_computer = GFTStrategyWraper()
        s_coeffs = {}
        for b in all_blocks:
            b.init_data(pc.V, pc.A)
            sg = StructuralGraph(b.metadata)
            sg.set_data(b.Vblock)
            dummy = Block(b.metadata); dummy.Vblock, dummy.Ablock = b.Vblock, b.Ablock
            _, c = gft_computer(dummy, sg)
            s_coeffs[b.block_id] = c
            b.clear_data()

        # Assignment
        tasks = []
        for b in all_blocks:
            b.init_data(pc.V, pc.A)
            tasks.append((b.metadata, b.Vblock.copy(), b.Ablock.copy(), slopes, slw, slp, args.q, 
                          params.sequential_params.lagrange_proportional, s_coeffs[b.block_id], 
                          params.sequential_params.decider_mode))
            b.clear_data()

        labels = Parallel(n_jobs=-1)(delayed(_get_labels_worker)(*t) for t in tqdm(tasks, desc=f"B{b_val} C{c_val}", leave=False))
        labels = np.array(labels, dtype=np.uint8)
        
        # 1. Entropy
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        ent_bpv = (entropy * len(labels)) / total_points
        
        # 2. Zlib
        z_bits = len(zlib.compress(labels.tobytes(), level=9)) * 8
        z_bpv = z_bits / total_points
        
        # 3. LZMA
        l_bits = len(lzma.compress(labels.tobytes())) * 8
        l_bpv = l_bits / total_points
        
        savings = (ent_bpv - l_bpv) / ent_bpv * 100
        print(f"| B{b_val} C{c_val} | {ent_bpv:11.6f} | {z_bpv:8.6f} | {l_bpv:8.6f} | {savings:+.1f}% |")

if __name__ == "__main__":
    main()
