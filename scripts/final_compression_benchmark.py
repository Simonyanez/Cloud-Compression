import sys
import os
import json
import sqlite3
import numpy as np
import lzma
from pathlib import Path
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

# --- DB SETUP ---
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dict_code TEXT,
        block_size INTEGER,
        clusters INTEGER,
        q INTEGER,
        lambda_prop REAL,
        beta REAL,
        base_psnr REAL,
        base_bpv REAL,
        data_psnr REAL,
        data_bpv REAL,
        lzma_overhead_bpv REAL,
        cabac_overhead_bpv REAL,
        total_bpv_lzma REAL,
        total_bpv_cabac REAL,
        distribution TEXT
    )""")
    conn.commit()
    return conn

# --- CABAC PROXY ---
def estimate_cabac_bits(labels):
    if len(labels) < 2: return 0
    num_c = int(np.max(labels) + 1)
    transitions = np.zeros((num_c, num_c))
    for i in range(1, len(labels)):
        transitions[labels[i-1], labels[i]] += 1
    
    row_sums = transitions.sum(axis=1, keepdims=True)
    p_y = row_sums.flatten() / (len(labels) - 1)
    
    cond_entropy = 0
    for y in range(num_c):
        if row_sums[y] > 0:
            p_x_given_y = transitions[y, :] / row_sums[y]
            nz = p_x_given_y[p_x_given_y > 0]
            h_x_given_y = -np.sum(nz * np.log2(nz))
            cond_entropy += p_y[y] * h_x_given_y
            
    return cond_entropy * len(labels)

# --- OPTIMIZED WORKER ---
def compute_all_coeffs_for_block(metadata, Vb, Ab, slopes, slw_vals, slp_vals, structural_coeffs):
    os.environ["OMP_NUM_THREADS"] = "1"
    gft_computer = GFTStrategyWraper()
    block = Block(metadata); block.Vblock, block.Ablock = Vb, Ab
    all_coeffs = [structural_coeffs]
    for k, slope in enumerate(slopes[1:]):
        slw, slp = slw_vals[k+1], slp_vals[k+1]
        s_graph = StructuralGraph(metadata); s_graph.set_data(Vb)
        a_graph = AttributeGraph(s_graph, slope, k+1, slp, slw)
        V_rot = Approximator()._spatial_norm(Vb)
        A_app = Ab.copy(); A_app[:, 0] = V_rot @ slope.T
        a_graph.set_data(Vb, A_app)
        _, coeffs = gft_computer(block, a_graph)
        all_coeffs.append(coeffs)
    return all_coeffs

def main():
    Q_STEPS = [24, 32, 48, 64]
    DB_PATH = "final_full_benchmark_v2.db"
    conn = init_db(DB_PATH)
    
    with open("selected_dicts.txt", "r") as f:
        dict_files = [line.strip() for line in f if line.strip()]

    params = load_experiment_config(Path("config/base_config.yaml"))
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    total_points = pc.V.shape[0]

    partition_cache = {}
    structural_coeffs_cache = {}

    for df_path in tqdm(dict_files, desc="Dictionaries"):
        with open(df_path, 'r') as f:
            d = json.load(f)
        
        code = d['experiment_code']
        b_val = 16 if "B16" in code else 32 if "B32" in code else 8 if "B8" in code else 4
        lambda_p = d.get('lambda_prop', 1.0)
        beta = d.get('beta', 1000.0)
        
        if b_val not in partition_cache:
            _, blocks = MortonBlockPartition().partition(pc, bsize=b_val)
            partition_cache[b_val] = blocks
            gft = GFTStrategyWraper()
            s_map = {}
            for b in tqdm(blocks, leave=False, desc=f"Struct B{b_val}"):
                b.init_data(pc.V, pc.A)
                sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
                dummy = Block(b.metadata); dummy.Vblock, dummy.Ablock = b.Vblock, b.Ablock
                _, c = gft(dummy, sg)
                s_map[b.block_id] = c
                b.clear_data()
            structural_coeffs_cache[b_val] = s_map

        blocks = partition_cache[b_val]
        s_coeffs_map = structural_coeffs_cache[b_val]
        num_blocks = len(blocks)

        slopes = np.array(d['final_slopes'])
        slw, slp = np.array(d['final_slw']), np.array(d['final_slp'])
        tasks = []
        for b in blocks:
            b.init_data(pc.V, pc.A)
            tasks.append((b.metadata, b.Vblock.copy(), b.Ablock.copy(), slopes, slw, slp, s_coeffs_map[b.block_id]))
            b.clear_data()
        
        all_coeffs_matrix = Parallel(n_jobs=-1)(delayed(compute_all_coeffs_for_block)(*t) for t in tqdm(tasks, leave=False, desc="GFTs"))

        for q in Q_STEPS:
            decider = Decider("0", lambda_p)
            decider._set_vars(q)
            num_c = len(slopes)
            cost_mat = np.zeros((num_blocks, num_c))
            rate_mat = np.zeros((num_blocks, num_c))
            dist_mat = np.zeros((num_blocks, num_c))
            
            for i in range(num_blocks):
                for k in range(num_c):
                    c, r, d = decider._RDcost(all_coeffs_matrix[i][k])
                    cost_mat[i, k] = c; rate_mat[i, k] = r; dist_mat[i, k] = d
            
            labels = np.zeros(num_blocks, dtype=np.uint8)
            labels[0] = np.argmin(cost_mat[0, :])
            for i in range(1, num_blocks):
                adj = cost_mat[i, :].copy(); mask = np.ones(num_c, dtype=bool); mask[labels[i-1]] = False; adj[mask] += beta
                labels[i] = np.argmin(adj)
            
            data_bits = sum(rate_mat[i, labels[i]] for i in range(num_blocks))
            total_dist = sum(dist_mat[i, labels[i]] for i in range(num_blocks))
            psnr = 20 * np.log10(255 / np.sqrt(total_dist / total_points))
            
            lzma_bits = len(lzma.compress(labels.tobytes())) * 8
            cabac_bits = estimate_cabac_bits(labels)
            
            base_bits = sum(rate_mat[:, 0])
            base_psnr = 20 * np.log10(255 / np.sqrt(sum(dist_mat[:, 0]) / total_points))
            
            unique, counts = np.unique(labels, return_counts=True)
            dist_dict = {int(k): int(v) for k, v in zip(unique, counts)}
            
            conn.execute("INSERT INTO results VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                code, b_val, num_c-1, q, lambda_p, beta,
                base_psnr, base_bits/total_points,
                psnr, data_bits/total_points,
                lzma_bits/total_points,
                cabac_bits/total_points,
                (data_bits + lzma_bits)/total_points,
                (data_bits + cabac_bits)/total_points,
                json.dumps(dist_dict)
            ))
            conn.commit()

    print(f"\n[+] BENCHMARK COMPLETE. Results in {DB_PATH}")
    conn.close()

if __name__ == "__main__":
    main()
