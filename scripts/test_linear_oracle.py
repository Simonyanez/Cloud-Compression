import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Path setup
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.parameters import load_experiment_config
from pcadc.decider import Decider

def compute_sink_vector(W, Y):
    """Fitted Y is used here instead of raw Ablock."""
    M = W * np.subtract.outer(Y, Y)
    M_masked = np.copy(M); M_masked[W == 0] = np.inf; np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    S = np.zeros(W.shape[0]); S[unique] = counts
    max_val = np.max(S)
    return S / max_val if max_val > 0 else S

def linear_oracle_worker(args):
    V, A, q_step, lp, percentages, weights = args
    decider = Decider(mode="0", lagrange_proportional=lp)
    decider._set_vars(q_step)
    N = V.shape[0]
    if N <= 1: return (1, 0, 1, 0, N, 0, 0, 0)

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
    
    # 2. Linear Fit
    poly = PolynomialFeatures(degree=1, include_bias=True)
    V_poly = poly.fit_transform(V)
    model = LinearRegression(fit_intercept=False).fit(V_poly, A[:, 0])
    Y_fitted = model.predict(V_poly)
    rmse_fit = np.sqrt(np.mean((A[:, 0] - Y_fitted)**2))

    # 3. Hybrid Search using Fitted Y
    S = compute_sink_vector(W_s, Y_fitted)
    sorted_nodes = np.argsort(S)[::-1]
    unique_counts = sorted(list(set([max(1, int(round(N * p))) for p in percentages])))
    
    best_c, best_r, best_d, best_p, best_w = cost_s, r_s, d_s, 0.0, 0.0
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

    return (r_s, d_s, best_r, best_d, N, best_p, best_w, rmse_fit)

def run_test():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.A = Colourist()._RGBtoYUV(pc.A)
    
    bsize = 8
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # 1% Sample for testing
    sampler = Sampler(ratio=0.01, n_strata=5)
    sampled = sampler(pc.V, pc.A, all_blocks)
    
    q_steps = [32, 48, 64]
    percentages = np.arange(0.01, 1.01, 0.05)
    weights = np.arange(0.5, 5.5, 0.5)
    
    print(f"\n[LINEAR ORACLE] Testing B={bsize} with {len(sampled)} blocks...")
    
    results_all = {}
    for q in q_steps:
        tasks = [(b.init_data(pc.V, pc.A) or b.Vblock.copy(), b.Ablock.copy(), q, 
                  params.sequential_params.lagrange_proportional, percentages, weights) for b in sampled]
        # Clear data immediately to save memory
        for b in sampled: b.clear_data()

        with Pool(cpu_count()) as pool:
            res = list(tqdm(pool.imap(linear_oracle_worker, tasks), total=len(tasks), desc=f"Q={q}"))
        
        results_all[q] = res

    # Summary
    print("\n--- Linear Oracle Summary (Sampled) ---")
    for q in q_steps:
        res = results_all[q]
        total_v = sum(r[4] for r in res)
        gain_rate = (sum(r[0] for r in res) - sum(r[2] for r in res)) / sum(r[0] for r in res) * 100
        hyb_perc = len([r for r in res if r[5] > 0]) / len(res) * 100
        print(f"Q={q}: Rate Reduction = {gain_rate:.2f}% | Hybrid Selection = {hyb_perc:.1f}%")

if __name__ == "__main__":
    run_test()
