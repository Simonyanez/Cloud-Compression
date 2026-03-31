import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Path setup
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.decider import Decider

def compute_graph_features(V, A):
    """Extract advanced math properties from the Attribute Graph logic."""
    N = V.shape[0]
    # 1. Structural Adjacency
    D = cdist(V, V)
    W_s = np.zeros_like(D)
    mask = (D > 0) & (D <= np.sqrt(3) + 1e-5)
    W_s[mask] = 1.0 / D[mask]
    W_s = W_s + W_s.T
    
    # 2. Sink Vector S
    Y = A[:, 0]
    M = W_s * np.subtract.outer(Y, Y)
    M_masked = np.copy(M); M_masked[W_s == 0] = np.inf; np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    S = np.zeros(N); S[unique] = counts
    
    # 3. Features
    s_mean = np.mean(S)
    s_var = np.var(S)
    s_max = np.max(S)
    s_sparsity = np.count_nonzero(S) / N
    s_entropy = entropy(S + 1e-9) # Distribution of 'influence'
    
    # Gradient intensity (only on edges)
    grad_vals = M[W_s > 0]
    avg_grad = np.mean(np.abs(grad_vals)) if grad_vals.size > 0 else 0
    
    # Graph Topology
    edge_density = np.count_nonzero(W_s) / (N * (N-1)) if N > 1 else 0
    
    return {
        "s_mean": s_mean, "s_var": s_var, "s_max": s_max, 
        "s_sparsity": s_sparsity, "s_entropy": s_entropy,
        "avg_grad": avg_grad, "edge_density": edge_density
    }

def exhaustive_worker(args):
    V, A, q_step, lp = args
    decider = Decider(mode="0", lagrange_proportional=lp)
    decider._set_vars(q_step)
    N = V.shape[0]
    
    # Advanced Features
    adv_feats = compute_graph_features(V, A)
    # Raw Features
    raw_feats = {
        "point_count": N,
        "y_var": np.var(A[:, 0]),
        "y_mean": np.mean(A[:, 0]),
        "geom_range": np.max(np.max(V, axis=0) - np.min(V, axis=0))
    }

    # Optimization
    D = cdist(V, V)
    W_s = np.zeros_like(D); mask = (D > 0) & (D <= np.sqrt(3) + 1e-5); W_s[mask] = 1.0/D[mask]; W_s += W_s.T
    L_s = np.diag(np.sum(W_s, axis=1)) - W_s
    evals_s, evecs_s = eigh(L_s)
    coeffs_s = evecs_s.T @ A
    cost_s, _, _ = decider._RDcost(coeffs_s)
    
    best_c, best_p, best_w = cost_s, 0.0, 0.0
    
    # Granular sweep
    percentages = np.arange(0.02, 0.52, 0.04)
    weights = np.arange(0.5, 5.5, 0.5)
    
    # Simplified sink selection for speed
    S = compute_graph_features(V, A)["s_max"] # Need to sort nodes
    # Recalculate S vector properly for selection
    Y = A[:, 0]; M = W_s * np.subtract.outer(Y, Y); M_masked = np.copy(M)
    M_masked[W_s == 0] = np.inf; np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1); unique, counts = np.unique(dec_i, return_counts=True)
    S_vec = np.zeros(N); S_vec[unique] = counts
    sorted_nodes = np.argsort(S_vec)[::-1]

    for p in percentages:
        n_sl = max(1, int(round(N * p)))
        sel = sorted_nodes[:n_sl]
        for w in weights:
            W_h = W_s.copy(); W_h[sel, sel] = w
            L_h = np.diag(np.sum(W_h, axis=1)) - W_h + np.diag(np.diag(W_h))
            evals_h, evecs_h = eigh(L_h)
            coeffs_h = evecs_h.T @ A
            cost_h, _, _ = decider._RDcost(coeffs_h)
            if cost_h < best_c:
                best_c, best_p, best_w = cost_h, p, w

    return {**raw_feats, **adv_feats, "best_p": best_p, "best_w": best_w}

def main():
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.A = Colourist()._RGBtoYUV(pc.A)
    
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=8)
    
    n_sample = 500
    sampler = Sampler(ratio=n_sample/len(all_blocks), n_strata=5)
    sampled = sampler(pc.V, pc.A, all_blocks)
    
    tasks = []
    for b in sampled:
        b.init_data(pc.V, pc.A)
        tasks.append((b.Vblock.copy(), b.Ablock.copy(), 48, params.sequential_params.lagrange_proportional))
        b.clear_data()
    
    print(f"Extracting features and optimal parameters for {len(tasks)} blocks...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(exhaustive_worker, tasks), total=len(tasks)))
    
    df = pd.DataFrame(results)
    
    raw_cols = ["point_count", "y_var", "y_mean", "geom_range"]
    adv_cols = ["s_mean", "s_var", "s_max", "s_sparsity", "s_entropy", "avg_grad", "edge_density"]
    targets = ["best_p", "best_w"]

    print("\n--- Non-Linear Predictive Power (R^2 Score) ---")
    for target in targets:
        print(f"\nPredicting {target}:")
        
        # Raw Set
        X_train, X_test, y_train, y_test = train_test_split(df[raw_cols], df[target], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        score_raw = r2_score(y_test, model.predict(X_test))
        
        # Advanced Set
        X_train, X_test, y_train, y_test = train_test_split(df[adv_cols], df[target], test_size=0.2, random_state=42)
        model_adv = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        score_adv = r2_score(y_test, model_adv.predict(X_test))
        
        print(f"  Raw Features R^2:      {score_raw:.4f}")
        print(f"  Advanced Features R^2: {score_adv:.4f}")
        
        # Feature Importance for Advanced
        importances = pd.Series(model_adv.feature_importances_, index=adv_cols).sort_values(ascending=False)
        print(f"  Top Advanced Importance:\n{importances.head(3)}")

    # Visualization of importance
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='teal')
    plt.title(f"Feature Importance for {targets[0]} (Advanced Set)")
    plt.tight_layout()
    plt.savefig("logs/advanced_feature_importance.png")
    print("\nPlot saved to logs/advanced_feature_importance.png")

if __name__ == "__main__":
    from pcadc.color import Colourist
    main()
