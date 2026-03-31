import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.linalg import eigh

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.decider import Decider

def bjontegaard_metric(R1, PSNR1, R2, PSNR2):
    """Calculates BD-Rate."""
    lR1, lR2 = np.log10(R1), np.log10(R2)
    idx1, idx2 = np.argsort(PSNR1), np.argsort(PSNR2)
    f1, f2 = interp1d(PSNR1[idx1], lR1[idx1], kind='cubic'), interp1d(PSNR2[idx2], lR2[idx2], kind='cubic')
    min_p, max_p = max(min(PSNR1), min(PSNR2)), min(max(PSNR1), max(PSNR2))
    return (10**((quad(f2, min_p, max_p)[0] - quad(f1, min_p, max_p)[0]) / (max_p - min_p)) - 1) * 100

def compute_sink_vector(W, A):
    """Optimized sink vector calculation."""
    Y = A[:, 0]
    M = W * np.subtract.outer(Y, Y)
    # Masking self and non-neighbors
    M_masked = np.copy(M)
    M_masked[W == 0] = np.inf
    np.fill_diagonal(M_masked, np.inf)
    
    # Get nodes that are pointed to by the most decreased neighbors
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    
    S = np.zeros(W.shape[0])
    S[unique] = counts
    
    # Min-Max Normalization
    min_val, max_val = np.min(S), np.max(S)
    if max_val > min_val:
        S = (S - min_val) / (max_val - min_val)
    else:
        S[:] = 0
    return S

def brute_force_block_worker(args):
    """
    Extremely optimized brute force worker.
    Uses direct matrix operations to minimize overhead.
    """
    Vblock, Ablock, q_step, lp, percentages, weights, dist_th = args
    
    decider = Decider(mode="0", lagrange_proportional=lp)
    decider._set_vars(q_step)
    
    N = Vblock.shape[0]
    if N == 1:
        return (1, 0, 1, 0, 1, 0, 0) # Rate, Dist, Rate_O, Dist_O, N, P, W

    # 1. Structural Adjacency
    from scipy.spatial.distance import cdist
    D = cdist(Vblock, Vblock)
    W_s = np.zeros_like(D)
    mask = (D > 0) & (D <= dist_th + 1e-5)
    W_s[mask] = 1.0 / D[mask]
    W_s = W_s + W_s.T
    
    # 2. Structural Baseline
    Deg_s = np.diag(np.sum(W_s, axis=1))
    L_s = Deg_s - W_s
    evals_s, evecs_s = eigh(L_s)
    # DC sign convention
    for i in range(evecs_s.shape[1]):
        if evecs_s[0, i] < 0: evecs_s[:, i] *= -1
    coeffs_s = evecs_s.T @ Ablock
    cost_s, r_s, d_s = decider._RDcost(coeffs_s)
    
    best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0.0, 0.0
    
    # Pre-calculate sink vector (only depends on Structural W and A)
    S = compute_sink_vector(W_s, Ablock)
    sorted_nodes = np.argsort(S)[::-1]
    
    # 3. Brute Force Attributes
    # We iterate over unique SL counts to avoid redundant GFTs
    unique_sl_counts = sorted(list(set([int(round(N * p)) for p in percentages if p > 0])))
    if not unique_sl_counts: unique_sl_counts = [1]
    
    for n_sl in unique_sl_counts:
        if n_sl == 0: continue
        selected_nodes = sorted_nodes[:n_sl]
        
        for w in weights:
            W_h = W_s.copy()
            W_h[selected_nodes, selected_nodes] = w
            
            Deg_h = np.diag(np.sum(W_h, axis=1))
            # Laplacian with self-loops: L = D - A + C where C is diag(A)
            L_h = Deg_h - W_h + np.diag(np.diag(W_h))
            
            evals_h, evecs_h = eigh(L_h)
            for i in range(evecs_h.shape[1]):
                if evecs_h[0, i] < 0: evecs_h[:, i] *= -1
            
            coeffs_h = evecs_h.T @ Ablock
            cost_h, r_h, d_h = decider._RDcost(coeffs_h)
            
            if cost_h < best_c:
                best_c, best_r, best_d, best_p, best_w = cost_h, r_h, d_h, n_sl/N, w

    return (r_s, d_s, best_r, best_d, N, best_p, best_w)

def run_absolute_brute_force():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    block_sizes = [4, 8, 16]
    q_steps = [24, 32, 48, 64]
    
    # EXTREMELY GRANULAR FULL RANGE
    percentages = np.arange(0.01, 1.01, 0.02) # 1% to 100%
    weights = np.arange(0.2, 10.2, 0.2) # 0.2 to 10.0
    
    dist_threshold = np.sqrt(3)
    final_results = {}

    for bsize in block_sizes:
        print(f"\n{'='*60}\nABSOLUTE BRUTE FORCE ORACLE B={bsize}\n{'='*60}")
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        
        # High sample count for stability
        sample_ratio = 0.01
        sampler = Sampler(ratio=sample_ratio, n_strata=5)
        sampled_blocks = sampler(pc.V, pc.A, all_blocks)
        
        rates_s, dists_s, rates_o, dists_o = [], [], [], []
        chosen_p, chosen_w = [], []
        
        for q in q_steps:
            print(f"Brute Forcing Q={q} with {len(sampled_blocks)} blocks...")
            tasks = []
            for b in sampled_blocks:
                b.init_data(pc.V, pc.A)
                tasks.append((b.Vblock.copy(), b.Ablock.copy(), q, 
                              params.sequential_params.lagrange_proportional,
                              percentages, weights, dist_threshold))
                b.clear_data()
            
            with Pool(cpu_count()) as pool:
                results = pool.map(brute_force_block_worker, tasks)
            
            total_v = sum(r[4] for r in results)
            rates_s.append(sum(r[0] for r in results) / total_v)
            dists_s.append(20 * np.log10(255 / (sum(r[1] for r in results) / total_v)))
            rates_o.append(sum(r[2] for r in results) / total_v)
            dists_o.append(20 * np.log10(255 / (sum(r[3] for r in results) / total_v)))
            
            for r in results:
                if r[5] > 0:
                    chosen_p.append(r[5])
                    chosen_w.append(r[6])

        bd_rate = bjontegaard_metric(np.array(rates_s), np.array(dists_s), np.array(rates_o), np.array(dists_o))
        print(f"\n>>> ABSOLUTE BRUTE FORCE BD-RATE (B={bsize}): {bd_rate:.4f}%")
        final_results[bsize] = (rates_s, dists_s, rates_o, dists_o, bd_rate, chosen_p, chosen_w)

        # Plot Joint Distribution
        plt.figure(figsize=(12, 10))
        hist, xedges, yedges = np.histogram2d(chosen_p, chosen_w, 
                                             bins=[np.unique(percentages), weights])
        sns.heatmap(hist.T, annot=False, cmap="viridis",
                    xticklabels=[f"{p*100:.0f}%" for p in np.unique(percentages)], 
                    yticklabels=[f"{w:.1f}" for w in weights])
        plt.title(f"Absolute Oracle Distribution (B={bsize}, Q=24-64)")
        plt.xlabel("Percentage (P)")
        plt.ylabel("Weight (W)")
        plt.savefig(f"logs/absolute_brute_force_B{bsize}_dist.png")
        plt.close()

    # Final RD Plots
    plt.figure(figsize=(15, 5))
    for i, bsize in enumerate(block_sizes):
        rs, ds, ro, do, bdr, _, _ = final_results[bsize]
        plt.subplot(1, 3, i+1)
        plt.plot(rs, ds, 'o-', label='Structural')
        plt.plot(ro, do, 'd--', label='Absolute Oracle', color='red')
        for j, q in enumerate(q_steps):
            plt.text(ro[j], do[j], f"Q={q}", fontsize=8)
        plt.title(f"B={bsize} | BD-Rate: {bdr:.2f}%")
        plt.xlabel("Rate")
        plt.ylabel("PSNR")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("logs/absolute_brute_force_upper_bound.png")
    print("\nResults and distributions saved to logs/")

if __name__ == "__main__":
    run_absolute_brute_force()
