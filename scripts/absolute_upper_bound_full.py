import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import sqlite3
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
import time

# Path setup
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.decider import Decider

# Database setup
DB_PATH = project_root / "results" / "oracle_upper_bound.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            block_id TEXT,
            bsize INTEGER,
            q_step INTEGER,
            point_count INTEGER,
            y_var REAL,
            u_var REAL,
            v_var REAL,
            y_mean REAL,
            geom_range_max REAL,
            struct_rate INTEGER,
            struct_dist REAL,
            struct_cost REAL,
            oracle_rate INTEGER,
            oracle_dist REAL,
            oracle_cost REAL,
            best_p REAL,
            best_w REAL,
            gain REAL,
            PRIMARY KEY (block_id, bsize, q_step)
        )
    """)
    conn.commit()
    conn.close()

def compute_sink_vector(W, A):
    Y = A[:, 0]
    M = W * np.subtract.outer(Y, Y)
    M_masked = np.copy(M); M_masked[W == 0] = np.inf; np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    S = np.zeros(W.shape[0]); S[unique] = counts
    max_val = np.max(S)
    return S / max_val if max_val > 0 else S

def oracle_worker(args):
    V, A, b_id, bsize, q_step, lp, percentages, weights = args
    decider = Decider(mode="0", lagrange_proportional=lp)
    decider._set_vars(q_step)
    
    N = V.shape[0]
    # Basic Features
    y_var, u_var, v_var = np.var(A, axis=0) if N > 1 else (0,0,0)
    y_mean = np.mean(A[:, 0])
    geom_range = np.max(np.max(V, axis=0) - np.min(V, axis=0)) if N > 1 else 0

    if N <= 1:
        return (b_id, bsize, q_step, N, y_var, u_var, v_var, y_mean, geom_range, 1, 0, 0, 1, 0, 0, 0, 0, 0)

    # 1. Structural
    D = cdist(V, V)
    W_s = np.zeros_like(D)
    mask = (D > 0) & (D <= np.sqrt(3) + 1e-5)
    W_s[mask] = 1.0 / D[mask]; W_s = W_s + W_s.T
    
    L_s = np.diag(np.sum(W_s, axis=1)) - W_s
    evals_s, evecs_s = eigh(L_s)
    for i in range(evecs_s.shape[1]):
        if evecs_s[0, i] < 0: evecs_s[:, i] *= -1
    coeffs_s = evecs_s.T @ A
    cost_s, r_s, d_s = decider._RDcost(coeffs_s)
    
    best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0.0, 0.0
    
    # 2. Hybrid Search
    S = compute_sink_vector(W_s, A)
    sorted_nodes = np.argsort(S)[::-1]
    unique_counts = sorted(list(set([max(1, int(round(N * p))) for p in percentages])))
    
    for n_sl in unique_counts:
        sel = sorted_nodes[:n_sl]
        for w in weights:
            W_h = W_s.copy(); W_h[sel, sel] = w
            L_h = np.diag(np.sum(W_h, axis=1)) - W_h + np.diag(np.diag(W_h))
            evals_h, evecs_h = eigh(L_h)
            for i in range(evecs_h.shape[1]):
                if evecs_h[0, i] < 0: evecs_h[:, i] *= -1
            coeffs_h = evecs_h.T @ A
            c, r, d = decider._RDcost(coeffs_h)
            if c < best_c:
                best_c, best_r, best_d, best_p, best_w = c, r, d, n_sl/N, w

    gain = cost_s - best_c
    return (b_id, bsize, q_step, N, float(y_var), float(u_var), float(v_var), float(y_mean), float(geom_range), 
            int(r_s), float(d_s), float(cost_s), int(best_r), float(best_d), float(best_c), 
            float(best_p), float(best_w), float(gain))

def main():
    init_db()
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.A = Colourist()._RGBtoYUV(pc.A)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64]
    percentages = np.arange(0.01, 1.01, 0.02)
    weights = np.arange(0.2, 10.2, 0.2)
    
    print(f"\n[ORACLE] Starting FULL POINT CLOUD execution.")
    print(f"[ORACLE] Storing results in: {DB_PATH}")

    for bsize in block_sizes:
        print(f"\n--- Block Size {bsize} ---")
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        # Check existing in DB to allow resume
        conn = sqlite3.connect(DB_PATH)
        existing = pd.read_sql(f"SELECT block_id, q_step FROM results WHERE bsize={bsize}", conn)
        conn.close()
        existing_set = set(zip(existing.block_id, existing.q_step))

        for q in q_steps:
            tasks = []
            for b in all_blocks:
                if (b.metadata.block_id, q) in existing_set: continue
                b.init_data(pc.V, pc.A)
                tasks.append((b.Vblock.copy(), b.Ablock.copy(), b.metadata.block_id, bsize, q, 
                              params.sequential_params.lagrange_proportional, percentages, weights))
                b.clear_data()
            
            if not tasks:
                print(f"  Q={q} already completed.")
                continue

            print(f"  Processing Q={q} ({len(tasks)} blocks remaining)...")
            with Pool(cpu_count()) as pool:
                # Use a buffer to insert in batches
                batch = []
                for res in tqdm(pool.imap_unordered(oracle_worker, tasks), total=len(tasks), desc=f"B={bsize} Q={q}"):
                    batch.append(res)
                    if len(batch) >= 100:
                        conn = sqlite3.connect(DB_PATH)
                        conn.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                        conn.commit()
                        conn.close()
                        batch = []
                
                if batch:
                    conn = sqlite3.connect(DB_PATH)
                    conn.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    conn.commit()
                    conn.close()

    print("\n[COMPLETE] Full point cloud exhaustive search finished.")

if __name__ == "__main__":
    import pandas as pd
    main()
