"""
Fair Comparison: Index-Based vs. 3D Gradient-Aligned Decoupled Viterbi
======================================================================
Trains a single unconstrained EM codebook at gamma = 0 (K = 4, B = 16, Q = 24).
Runs both Viterbi solvers on the exact same cost matrix and block sequence,
using identical spatial transition switch counting logic to produce a fair
side-by-side comparison table.
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup project paths
project_root = Path("/home/simao/Documents/Repositories/Cloud-Compression")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist
from oracle_em_tangent import e_step, m_step, extract_block_tangent_planes

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
tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

for i, block in enumerate(blocks):
    block.init_data(pc.V, pc.A)
    coords.append(np.mean(block.Vblock, axis=0))
    normals.append(tangent_data[i]["v3"])
    block.clear_data()

coords = np.array(coords)
normals = np.array(normals)

# Precompute structural GFT coeffs
print("Pre-computing structural GFT coefficients...")
from pcadc.transforms import GFTStrategyWraper
from pcadc.graph import StructuralGraph
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

# Helper to compute conditional entropy
def compute_entropy(labels, K):
    cluster_sizes = np.bincount(labels, minlength=K)
    trans = np.ones((K, K))
    for i in range(1, len(labels)):
        trans[labels[i-1], labels[i]] += 1
    trans_probs = trans / trans.sum(axis=1, keepdims=True)
    h_cond = 0.0
    for prev in range(K):
        p_prev = cluster_sizes[prev] / len(labels)
        for cur in range(K):
            p = trans_probs[prev, cur]
            if p > 0 and p_prev > 0:
                h_cond -= p_prev * p * np.log2(p)
    return h_cond

# Configuration
K = 4
gammas = [1000.0, 2000.0, 4000.0]

print(f"\n============================================================")
print(f"PHASE 1: TRAINING UNCONSTRAINED GFT EM FOR K = {K}")
print(f"============================================================")

# Initialize GFT codebook parameters
n_hybrid = K - 1
slopes_2d = np.zeros((K, 2))
for k in range(1, K):
    angle = 2 * np.pi * (k - 1) / n_hybrid
    slopes_2d[k] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    
slp = np.concatenate([[0.0], np.full(n_hybrid, 0.65)])
slw = np.concatenate([[1.0], np.full(n_hybrid, 1.4)])

spatial_weights = np.ones(N)
prev_labels = None
min_cluster_size = max(5, int(0.01 * N))

# Run unconstrained EM loop for 10 iterations (identical to our main run setup)
for it in range(1, 11):
    print(f"  Iteration {it:2d}/10...")
    
    # E-step (at gamma = 0.0)
    labels, cost_matrix = e_step(
        blocks, structural_coeffs_map,
        slopes_2d, slp, slw,
        tangent_data, q_step, lagrange, decider_mode,
        pc.V, pc.A,
        gamma=0.0,
        prev_labels=prev_labels,
        spatial_weights=spatial_weights
    )
    
    # M-step
    slopes_2d, slp, slw = m_step(
        blocks, labels, slopes_2d, slp, slw,
        tangent_data, pc.V, pc.A,
        q_step, lagrange, decider_mode,
        min_cluster_size=min_cluster_size,
        freeze_slopes=False,
        sample_rate=0.20
    )
    prev_labels = labels.copy()

print("\n[Phase 1 Done] Frozen unconstrained GFT codebook trained.")
for k in range(1, K):
    print(f"  Cluster {k}: s2d = [{slopes_2d[k,0]:+.3f}, {slopes_2d[k,1]:+.3f}] | slp = {slp[k]:.3f} | slw = {slw[k]:.3f}")

# Final cost matrix evaluation for frozen codebook
print("\nEvaluating final cost matrix...")
labels, cost_matrix = e_step(
    blocks, structural_coeffs_map,
    slopes_2d, slp, slw,
    tangent_data, q_step, lagrange, decider_mode,
    pc.V, pc.A,
    gamma=0.0,
    prev_labels=None,
    spatial_weights=spatial_weights
)

structural_cost = sum(cost_matrix[i, 0] for i in range(N))

# Precompute 3D slope vectors and norms for each block/state
s3d_vectors = np.zeros((N, K, 3))
s3d_norms = np.zeros((N, K))
for i in range(N):
    s3d_vectors[i, 0] = np.zeros(3)
    s3d_norms[i, 0] = 0.0
    for k in range(1, K):
        s2d = slopes_2d[k]
        s3d = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        s3d_vectors[i, k] = s3d
        s3d_norms[i, k] = np.linalg.norm(s3d)

results_table = []
results_table.append("# Fair Comparison: Index-Based vs. 3D Gradient-Aligned Decoupled Viterbi\n")
results_table.append("| Method | Baseline $\\gamma_0$ | GFT Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches (Spatial) |")
results_table.append("|:---|---:|---:|---:|---:|---:|---:|")

# Morton normal alignment weights: w_i = |n_i . n_{i-1}|
normal_weights = np.ones(N)
for i in range(1, N):
    normal_weights[i] = np.abs(np.dot(normals[i], normals[i-1]))

# Phase 2: Decoupled sweeps
for gamma0 in gammas:
    print(f"\n============================================================")
    print(f"PHASE 2: EVALUATING FOR GAMMA0 = {gamma0}")
    print(f"============================================================")
    
    # 1. INDEX-BASED VITERBI
    dp_idx = np.zeros((N, K))
    paths_idx = np.zeros((N, K), dtype=int)
    dp_idx[0] = cost_matrix[0]
    
    for i in range(1, N):
        w = normal_weights[i]
        for k in range(K):
            # Penalty is gamma0 * |n_i . n_{i-1}| if cluster indices change
            transition_costs = dp_idx[i-1] + gamma0 * w * (np.arange(K) != k)
            best_prev = np.argmin(transition_costs)
            dp_idx[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
            paths_idx[i, k] = best_prev
            
    labels_idx = np.zeros(N, dtype=int)
    labels_idx[N-1] = np.argmin(dp_idx[N-1])
    for i in range(N-2, -1, -1):
        labels_idx[i] = paths_idx[i+1, labels_idx[i+1]]
        
    gft_cost_idx = sum(cost_matrix[i, labels_idx[i]] for i in range(N))
    savings_idx = (structural_cost - gft_cost_idx) / structural_cost * 100
    entropy_idx = compute_entropy(labels_idx, K)
    overhead_idx = entropy_idx * N / total_v
    net_bpv_idx = base_rate * (savings_idx / 100.0) - overhead_idx
    switches_idx = int(np.sum(labels_idx[1:] != labels_idx[:-1]))
    
    results_table.append(f"| Index-Based | {gamma0:.1f} | {savings_idx:.3f}% | {entropy_idx:.4f} | {overhead_idx:.5f} | {net_bpv_idx:+.5f} | {switches_idx} |")
    
    # 2. 3D GRADIENT-ALIGNED VITERBI
    dp_grad = np.zeros((N, K))
    paths_grad = np.zeros((N, K), dtype=int)
    dp_grad[0] = cost_matrix[0]
    
    for i in range(1, N):
        w = normal_weights[i]
        s3d_prev = s3d_vectors[i-1]
        s3d_curr = s3d_vectors[i]
        norms_prev = s3d_norms[i-1]
        norms_curr = s3d_norms[i]
        
        for k in range(K):
            transition_costs = np.zeros(K)
            for p in range(K):
                if p == k == 0:
                    T = 0.0
                elif p == 0 or k == 0:
                    T = gamma0 * w
                else:
                    dot_val = np.dot(s3d_prev[p], s3d_curr[k])
                    norm_val = norms_prev[p] * norms_curr[k]
                    cos_sim = np.abs(dot_val) / norm_val if norm_val > 0 else 1.0
                    T = gamma0 * w * (1.0 - cos_sim)
                transition_costs[p] = dp_grad[i-1, p] + T
                
            best_prev = np.argmin(transition_costs)
            dp_grad[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
            paths_grad[i, k] = best_prev
            
    labels_grad = np.zeros(N, dtype=int)
    labels_grad[N-1] = np.argmin(dp_grad[N-1])
    for i in range(N-2, -1, -1):
        labels_grad[i] = paths_grad[i+1, labels_grad[i+1]]
        
    gft_cost_grad = sum(cost_matrix[i, labels_grad[i]] for i in range(N))
    savings_grad = (structural_cost - gft_cost_grad) / structural_cost * 100
    entropy_grad = compute_entropy(labels_grad, K)
    overhead_grad = entropy_grad * N / total_v
    net_bpv_grad = base_rate * (savings_grad / 100.0) - overhead_grad
    switches_grad = int(np.sum(labels_grad[1:] != labels_grad[:-1]))
    
    results_table.append(f"| 3D Gradient-Aligned | {gamma0:.1f} | {savings_grad:.3f}% | {entropy_grad:.4f} | {overhead_grad:.5f} | {net_bpv_grad:+.5f} | {switches_grad} |")

# Write report
artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "fair_comparison_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(results_table) + "\n")

print(f"\nFair comparison sweep completed! Report written to: {report_file.absolute()}")
