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
from scipy.spatial.distance import cdist

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.decider import Decider

def bjontegaard_metric(R1, PSNR1, R2, PSNR2):
    lR1, lR2 = np.log10(R1), np.log10(R2)
    idx1, idx2 = np.argsort(PSNR1), np.argsort(PSNR2)
    f1 = interp1d(PSNR1[idx1], lR1[idx1], kind='cubic')
    f2 = interp1d(PSNR2[idx2], lR2[idx2], kind='cubic')
    min_p, max_p = max(min(PSNR1), min(PSNR2)), min(max(PSNR1), max(PSNR2))
    return (10**((quad(f2, min_p, max_p)[0] - quad(f1, min_p, max_p)[0]) / (max_p - min_p)) - 1) * 100

def compute_sink_vector(W, A):
    Y = A[:, 0]
    M = W * np.subtract.outer(Y, Y)
    M_masked = np.copy(M)
    M_masked[W == 0] = np.inf
    np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    S = np.zeros(W.shape[0])
    S[unique] = counts
    max_val = np.max(S)
    if max_val > 0: S = S / max_val
    return S

def b16_worker(args):
    Vblock, Ablock, q_step, lp, percentages, weights = args
    decider = Decider(mode="0", lagrange_proportional=lp)
    decider._set_vars(q_step)
    
    N = Vblock.shape[0]
    if N <= 1: return (1, 0, 1, 0, N, 0, 0)

    # 1. Structural
    D = cdist(Vblock, Vblock)
    W_s = np.zeros_like(D)
    mask = (D > 0) & (D <= np.sqrt(3) + 1e-5)
    W_s[mask] = 1.0 / D[mask]
    W_s = W_s + W_s.T
    
    L_s = np.diag(np.sum(W_s, axis=1)) - W_s
    evals_s, evecs_s = eigh(L_s)
    coeffs_s = evecs_s.T @ Ablock
    cost_s, r_s, d_s = decider._RDcost(coeffs_s)
    
    best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0.0, 0.0
    
    # 2. Hybrid Hunt
    S = compute_sink_vector(W_s, Ablock)
    sorted_nodes = np.argsort(S)[::-1]
    unique_counts = sorted(list(set([max(1, int(round(N * p))) for p in percentages])))
    
    for n_sl in unique_counts:
        sel = sorted_nodes[:n_sl]
        for w in weights:
            W_h = W_s.copy()
            W_h[sel, sel] = w
            L_h = np.diag(np.sum(W_h, axis=1)) - W_h + np.diag(np.diag(W_h))
            evals_h, evecs_h = eigh(L_h)
            coeffs_h = evecs_h.T @ Ablock
            c, r, d = decider._RDcost(coeffs_h)
            if c < best_c:
                best_c, best_r, best_d, best_p, best_w = c, r, d, n_sl/N, w

    return (r_s, d_s, best_r, best_d, N, best_p, best_w)

def main():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    Colourist()._RGBtoYUV_inplace(pc)
    
    print("\n[HUNT] Starting high-precision B=16 search (100 blocks)...")
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=16)
    sampled_blocks = Sampler(ratio=300/len(all_blocks), n_strata=5)(pc.V, pc.A, all_blocks)
    
    q_steps = [24, 32, 48, 64]
    # Truly exhaustive granular range for absolute upper bound
    percentages = np.arange(0.01, 1.01, 0.02) 
    weights = np.arange(0.2, 10.2, 0.2)
    
    rates_s, dists_s, rates_o, dists_o = [], [], [], []
    chosen_p, chosen_w = [], []

    for q in q_steps:
        print(f"[HUNT] Processing Q={q}...")
        tasks = []
        for b in sampled_blocks:
            b.init_data(pc.V, pc.A)
            tasks.append((b.Vblock.copy(), b.Ablock.copy(), q, 
                          params.sequential_params.lagrange_proportional,
                          percentages, weights))
            b.clear_data()
        
        # Parallel processing
        with Pool(cpu_count()) as pool:
            results = []
            # We use imap to get results as they finish and print progress to prevent timeout
            for i, res in enumerate(pool.imap(b16_worker, tasks)):
                results.append(res)
                if (i+1) % 10 == 0:
                    print(f"  Progress (Q={q}): {i+1}/{len(tasks)} blocks optimized")
        
        total_v = sum(r[4] for r in results)
        rates_s.append(sum(r[0] for r in results) / total_v)
        dists_s.append(20 * np.log10(255 / (sum(r[1] for r in results) / total_v)))
        rates_o.append(sum(r[2] for r in results) / total_v)
        dists_o.append(20 * np.log10(255 / (sum(r[3] for r in results) / total_v)))
        for r in results:
            if r[5] > 0:
                chosen_p.append(r[5]); chosen_w.append(r[6])

    bd_rate = bjontegaard_metric(np.array(rates_s), np.array(dists_s), np.array(rates_o), np.array(dists_o))
    print(f"\n[FINAL] B=16 BD-RATE UPPER BOUND: {bd_rate:.4f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rates_s, dists_s, 'o-', label='Structural')
    plt.plot(rates_o, dists_o, 'd--', label='Hunt Oracle', color='red')
    plt.title(f"B=16 Final Hunt | BD-Rate: {bd_rate:.2f}%")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    hist, xedges, yedges = np.histogram2d(chosen_p, chosen_w, bins=[len(percentages), len(weights)])
    sns.heatmap(hist.T, cmap="rocket_r", xticklabels=False, yticklabels=False)
    plt.title("B=16 Parameter Distribution")
    plt.xlabel("Percentage (P)"); plt.ylabel("Weight (W)")
    
    plt.tight_layout()
    plt.savefig("logs/b16_definitive_hunt.png")
    print("[FINAL] Results saved to logs/b16_definitive_hunt.png")

if __name__ == "__main__":
    from pcadc.color import Colourist
    def _RGBtoYUV_inplace(self, pc):
        pc.A = self._RGBtoYUV(pc.A)
    Colourist._RGBtoYUV_inplace = _RGBtoYUV_inplace
    main()
