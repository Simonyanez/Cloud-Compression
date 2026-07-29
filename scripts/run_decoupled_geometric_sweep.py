"""
Decoupled Geometric EM and Viterbi Sweep
=========================================
Phase 1: Unconstrained EM Codebook Training (on 3D normals, gamma = 0)
Phase 2: Frozen One-Shot Spatial Viterbi Sweep
Evaluates over K in [2, 4, 6, 8] and gamma0 in [2000.0, 4000.0].
"""

import sys
import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup project path
project_root = Path("/home/simao/Documents/Repositories/Cloud-Compression")
sys.path.insert(0, str(project_root / "src"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block

# 1. LOAD DATA
params = load_experiment_config(project_root / "config/base_config.yaml")
colourist = Colourist()
pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path, "ply", params.pointcloud)
pc.transform_attributes(colourist._RGBtoYUV)

_, blocks = MortonBlockPartition().partition(pc, bsize=16)
N = len(blocks)
base_rate = 0.1480  # for B=16
total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)

# Extract centroids and normal vectors
print("Extracting block centroids and normals...")
coords = []
normals = []
tangent_data = []

for block in blocks:
    block.init_data(pc.V, pc.A)
    V = block.Vblock
    coords.append(np.mean(V, axis=0))
    
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])
    V_centered = np.zeros_like(V)
    if V.shape[0] >= 4:
        V_centered = V - np.mean(V, axis=0)
        cov = np.dot(V_centered.T, V_centered) / (V.shape[0] - 1)
        try:
            _, eigenvectors = np.linalg.eigh(cov)
            v1 = eigenvectors[:, 2]
            v2 = eigenvectors[:, 1]
            v3 = eigenvectors[:, 0]
        except np.linalg.LinAlgError:
            pass
    X_tangent = np.column_stack((V_centered @ v1, V_centered @ v2))
    
    normals.append(v3)
    tangent_data.append({
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "X_tangent": X_tangent
    })
    block.clear_data()

coords = np.array(coords)
normals = np.array(normals)

# Precompute structural GFT coeffs
print("Pre-computing structural GFT coefficients...")
gft_computer = GFTStrategyWraper()
structural_coeffs_map = {}
for block in blocks:
    block.init_data(pc.V, pc.A)
    s_graph = StructuralGraph(block.metadata)
    s_graph.set_data(block.Vblock)
    _, coeffs = gft_computer(block, s_graph)
    structural_coeffs_map[block.block_id] = coeffs
    block.clear_data()

decider_mode = params.sequential_params.decider_mode
lagrange = params.sequential_params.lagrange_proportional
q_step = 24

# Helper to initialize centroids (K-Means++)
def init_centroids_kmeanspp(normals, K):
    np.random.seed(42)  # For reproducibility
    centroids = [normals[np.random.choice(len(normals))]]
    for _ in range(1, K):
        dists = []
        for n in normals:
            min_d = min(1.0 - np.dot(n, mu)**2 for mu in centroids)
            dists.append(min_d)
        dists = np.array(dists)
        probs = dists / dists.sum()
        centroids.append(normals[np.random.choice(len(normals), p=probs)])
    return np.array(centroids)

# Sweeps configuration
Ks = [2, 4, 6, 8]
gammas = [2000.0, 4000.0]

results_table = []
results_table.append("# Decoupled Geometric EM Sweep Report\n")
results_table.append("| $K$ | Baseline $\\gamma_0$ | Converged Iter | Final MSAE | GFT Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches |")
results_table.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for K in Ks:
    print(f"\n============================================================")
    print(f"PHASE 1: TRAINING UNCONSTRAINED EM FOR K = {K}")
    print(f"============================================================")
    
    # K-means++ initialization
    centroids = init_centroids_kmeanspp(normals, K)
    
    labels = np.zeros(N, dtype=int)
    prev_msae = 1e9
    converged_iter = -1
    
    for it in range(1, 21):
        # E-step: cosine similarity
        for i in range(N):
            labels[i] = np.argmax([np.abs(np.dot(normals[i], mu)) for mu in centroids])
            
        # M-step: sign-aligned normal update
        for k in range(K):
            assigned_idx = np.where(labels == k)[0]
            if len(assigned_idx) == 0:
                continue
            
            # Align signs to prevent cancellation
            mu_curr = centroids[k]
            sum_n = np.zeros(3)
            for idx in assigned_idx:
                n = normals[idx]
                sign = np.sign(np.dot(n, mu_curr))
                if sign == 0:
                    sign = 1.0
                sum_n += sign * n
                
            norm_sum = np.linalg.norm(sum_n)
            if norm_sum > 0:
                centroids[k] = sum_n / norm_sum
                
        # Compute MSAE
        msae = 0.0
        for i in range(N):
            k = labels[i]
            msae += 1.0 - np.dot(normals[i], centroids[k])**2
        msae /= N
        
        delta = np.abs(msae - prev_msae)
        print(f"  Iteration {it:2d}: MSAE = {msae:.6f} | Delta = {delta:.8f}")
        
        if delta < 1e-5 and converged_iter == -1:
            converged_iter = it
            print(f"  [Convergence] Converged at iteration {it}!")
            
        prev_msae = msae
        
    if converged_iter == -1:
         converged_iter = 20

    print(f"\n[Phase 1 Done] K={K} converged in {converged_iter} iterations. Final MSAE: {prev_msae:.6f}")
    
    # 2. Train GFT Codebook (slopes, slp, slw) from unconstrained labels
    print(f"Training GFT representation parameters for K={K} clusters...")
    slopes_2d = np.zeros((K, 2))
    slp_vals = np.concatenate([[0.0], np.full(K - 1, 0.65)])
    slw_vals = np.concatenate([[1.0], np.full(K - 1, 1.4)])
    
    # Fit regression for each hybrid cluster k >= 1
    for k in range(1, K):
        mask = (labels == k)
        n_assigned = mask.sum()
        if n_assigned < 2:
            continue
            
        # Fit 2D slope parameters
        X_list = []
        Y_list = []
        s_prev = np.array([0.5, 0.0])  # fallback prior
        
        for idx in np.where(mask)[0]:
            block = blocks[idx]
            block.init_data(pc.V, pc.A)
            if block.Vblock.shape[0] >= 4:
                X = tangent_data[idx]["X_tangent"]
                Y = block.Ablock[:, 0] - np.mean(block.Ablock[:, 0])
                s_local, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                if np.dot(s_local, s_prev) < 0:
                    Y = -Y
                X_list.append(X)
                Y_list.append(Y)
            block.clear_data()
            
        if X_list:
            X_all = np.vstack(X_list)
            Y_all = np.concatenate(Y_list)
            s2d_new, _, _, _ = np.linalg.lstsq(X_all, Y_all, rcond=None)
            slopes_2d[k] = s2d_new
            
            # Grid search slp, slw on a sample
            Vblocks, Ablocks, metadatas, v1s, v2s = [], [], [], [], []
            sample_idx = np.random.choice(np.where(mask)[0], size=min(100, n_assigned), replace=False)
            for idx in sample_idx:
                block = blocks[idx]
                block.init_data(pc.V, pc.A)
                Vblocks.append(block.Vblock.copy())
                Ablocks.append(block.Ablock.copy())
                metadatas.append(block.metadata)
                v1s.append(tangent_data[idx]["v1"])
                v2s.append(tangent_data[idx]["v2"])
                block.clear_data()
                
            # Perform grid search to find best slp, slw
            dec = Decider(decider_mode, lagrange)
            dec._set_vars(q_step)
            gft = GFTStrategyWraper()
            app = Approximator()
            
            best_cost = 1e18
            best_p, best_w = 0.65, 1.4
            
            for p in [0.293, 0.447, 0.720]:
                for w in [0.325, 0.550, 0.872]:
                    # Evaluate cost
                    cost_sum = 0.0
                    for Vb, Ab, meta, v1, v2 in zip(Vblocks, Ablocks, metadatas, v1s, v2s):
                        b = Block(meta)
                        b.Vblock = Vb
                        b.Ablock = Ab
                        s3d = slopes_2d[k,0]*v1 + slopes_2d[k,1]*v2
                        s_graph = StructuralGraph(meta)
                        s_graph.set_data(Vb)
                        a_graph = AttributeGraph(s_graph, s3d, k, p, w)
                        V_rot = app._spatial_norm(Vb)
                        A_app = Ab.copy()
                        A_app[:, 0] = V_rot @ s3d.T
                        a_graph.set_data(Vb, A_app)
                        try:
                            _, coeffs = gft(b, a_graph)
                            c, _, _ = dec._RDcost(coeffs)
                            cost_sum += c
                        except Exception:
                            cost_sum += 1e18
                    if cost_sum < best_cost:
                        best_cost = cost_sum
                        best_p = p
                        best_w = w
                        
            slp_vals[k] = best_p
            slw_vals[k] = best_w

    # Evaluate cost matrix for full GFT
    print("Evaluating GFT cost matrix for Viterbi decoding...")
    gft_cost_matrix = []
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        Vblock = block.Vblock
        Ablock = block.Ablock
        metadata = block.metadata
        
        dec = Decider(decider_mode, lagrange)
        dec._set_vars(q_step)
        gft = GFTStrategyWraper()
        app = Approximator()
        
        b = Block(metadata)
        b.Vblock = Vblock
        b.Ablock = Ablock
        
        costs = np.full(K, np.inf)
        costs[0] = dec._RDcost(structural_coeffs_map[block.block_id])[0]
        
        for k in range(1, K):
            s2d = slopes_2d[k]
            s3d = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
            s_graph = StructuralGraph(metadata)
            s_graph.set_data(Vblock)
            a_graph = AttributeGraph(s_graph, s3d, k, slp_vals[k], slw_vals[k])
            V_rot = app._spatial_norm(Vblock)
            A_app = Ablock.copy()
            A_app[:, 0] = V_rot @ s3d.T
            a_graph.set_data(Vblock, A_app)
            try:
                _, coeffs = gft(b, a_graph)
                c, _, _ = dec._RDcost(coeffs)
                costs[k] = c
            except Exception:
                costs[k] = 1e18
        gft_cost_matrix.append(costs)
        block.clear_data()
        
    gft_cost_matrix = np.array(gft_cost_matrix)
    structural_cost = sum(gft_cost_matrix[i, 0] for i in range(N))

    # Phase 2: Frozen One-Shot Spatial Viterbi Sweep
    for gamma0 in gammas:
        print(f"  Evaluating Viterbi for gamma0 = {gamma0:.1f}...")
        
        # 1. Build normal alignment weights along Morton sequence
        normal_weights = np.ones(N)
        for i in range(1, N):
            normal_weights[i] = np.abs(np.dot(normals[i], normals[i-1]))
            
        # 2. Geometric Viterbi trellis solver
        # Data Cost: D(i, k) = 1 - (n_i . mu_k)^2
        dp = np.zeros((N, K))
        paths = np.zeros((N, K), dtype=int)
        
        # Initialize
        for k in range(K):
            dp[0, k] = 1.0 - np.dot(normals[0], centroids[k])**2
            
        for i in range(1, N):
            w = normal_weights[i]
            for k in range(K):
                data_cost = 1.0 - np.dot(normals[i], centroids[k])**2
                transition_costs = dp[i-1] + gamma0 * w * (np.arange(K) != k)
                best_prev = np.argmin(transition_costs)
                dp[i, k] = data_cost + transition_costs[best_prev]
                paths[i, k] = best_prev
                
        # Backtrack labels
        viterbi_labels = np.zeros(N, dtype=int)
        viterbi_labels[N-1] = np.argmin(dp[N-1])
        for i in range(N-2, -1, -1):
            viterbi_labels[i] = paths[i+1, viterbi_labels[i+1]]
            
        # 3. Evaluate GFT metrics for this label sequence
        total_gft_cost = sum(gft_cost_matrix[i, viterbi_labels[i]] for i in range(N))
        savings_pct = (structural_cost - total_gft_cost) / structural_cost * 100
        
        # Compute spatial switches
        switches = int(np.sum(viterbi_labels[1:] != viterbi_labels[:-1]))
        
        # Compute sequence entropy H(X|X-1)
        entropy = compute_entropy(viterbi_labels)
        overhead = entropy * N / total_v
        saved_bpv = base_rate * (savings_pct / 100.0)
        net_bpv = saved_bpv - overhead
        
        print(f"    GFT Savings: {savings_pct:.3f}% | Entropy: {entropy:.4f} | Net BPV: {net_bpv:+.5f} | Switches: {switches}")
        
        # Record
        results_table.append(f"| {K} | {gamma0:.1f} | {converged_iter} | {prev_msae:.6f} | {savings_pct:.3f}% | {entropy:.4f} | {overhead:.5f} | {net_bpv:+.5f} | {switches} |")

# Write to markdown file
artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "decoupled_geom_sweep_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(results_table) + "\n")

print(f"\nSweep completed successfully! Report written to: {report_file.absolute()}")
