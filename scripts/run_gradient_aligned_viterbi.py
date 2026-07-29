"""
3D Gradient-Aligned Decoupled Viterbi Sweep
===========================================
Reuses the exact e_step, m_step, and tangent plane extractor from oracle_em_tangent.py
to guarantee 100% parity in unconstrained EM codebook training.
Executes the one-shot 3D Gradient-Aligned Viterbi sequence optimization.
"""

import sys
import os
import json
import sqlite3
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
print("Loading/Pre-computing structural GFT coefficients...")
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
gammas = [10.0, 50.0, 100.0, 300.0, 500.0, 1000.0, 2000.0, 4000.0]

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
results_table.append("# 3D Gradient-Aligned Decoupled Viterbi Sweep Report\n")
results_table.append("| $K$ | Baseline $\\gamma_0$ | GFT Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches |")
results_table.append("|---:|---:|---:|---:|---:|---:|---:|")

# Phase 2: One-shot 3D Gradient-Aligned Viterbi
for gamma0 in gammas:
    print(f"\n============================================================")
    print(f"PHASE 2: RUNNING GRADIENT-ALIGNED VITERBI FOR GAMMA = {gamma0}")
    print(f"============================================================")
    
    # normal weights along Morton path: w_i = |n_i . n_{i-1}|
    normal_weights = np.ones(N)
    for i in range(1, N):
        normal_weights[i] = np.abs(np.dot(normals[i], normals[i-1]))
        
    dp = np.zeros((N, K))
    paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]
    
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
                    # Absolute value accounts for sign-symmetry in GFT attribute weights
                    cos_sim = np.abs(dot_val) / norm_val if norm_val > 0 else 1.0
                    T = gamma0 * w * (1.0 - cos_sim)
                transition_costs[p] = dp[i-1, p] + T
                
            best_prev = np.argmin(transition_costs)
            dp[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
            paths[i, k] = best_prev
            
    # Backtrack optimal sequence
    viterbi_labels = np.zeros(N, dtype=int)
    viterbi_labels[N-1] = np.argmin(dp[N-1])
    for i in range(N-2, -1, -1):
        viterbi_labels[i] = paths[i+1, viterbi_labels[i+1]]
        
    # Evaluate GFT cost of this label sequence
    total_gft_cost = sum(cost_matrix[i, viterbi_labels[i]] for i in range(N))
    savings_pct = (structural_cost - total_gft_cost) / structural_cost * 100
    
    # Compute spatial switches
    switches = int(np.sum(viterbi_labels[1:] != viterbi_labels[:-1]))
    
    # Compute sequence entropy H(X|X-1)
    entropy = compute_entropy(viterbi_labels, K)
    overhead = entropy * N / total_v
    saved_bpv = base_rate * (savings_pct / 100.0)
    net_bpv = saved_bpv - overhead
    
    print(f"  Results:")
    print(f"    GFT Savings: {savings_pct:.3f}%")
    print(f"    H(X|X-1):    {entropy:.4f} bits")
    print(f"    Net BPV:     {net_bpv:+.5f} bpv")
    print(f"    Switches:    {switches}")
    
    results_table.append(f"| {K} | {gamma0:.1f} | {savings_pct:.3f}% | {entropy:.4f} | {overhead:.5f} | {net_bpv:+.5f} | {switches} |")

# Write to markdown file
artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "gradient_aligned_viterbi_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(results_table) + "\n")

print(f"\nSweep completed successfully! Report written to: {report_file.absolute()}")
