"""
Serialization Gap Analysis Tests
=================================
Executes three diagnostics to analyze the spatial serialization gap:
  1. Experiment 1: The "Jump vs. Switch" Correlation Check
  2. Experiment 2: The "Nearest Neighbor" Alignment Gap (using KDTree)
  3. Experiment 3: The "Cheat-Sort" Quick Run (lexicographical sort)
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import KDTree

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
K = 4
q_step = 24

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

# Frozen slopes from Upgrade B
slopes_2d = np.array([
    [0.0, 0.0],
    [0.720, -1.160],
    [-0.790, 1.120],
    [-0.520, -1.170]
])
slp = np.array([0.0, 0.293, 0.447, 0.720])
slw = np.array([1.0, 0.325, 0.325, 0.325])

# Evaluate Cost Matrix
print("Evaluating cost matrix...")
decider_mode = params.sequential_params.decider_mode
lagrange = params.sequential_params.lagrange_proportional

cost_matrix = []
for i, block in enumerate(tqdm(blocks, desc="Cost Matrix")):
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
        s3d = s2d[0] * tangent_data[i]["v1"] + s2d[1] * tangent_data[i]["v2"]
        
        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, s3d, k, slp[k], slw[k])
        
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
    cost_matrix.append(costs)
    block.clear_data()

cost_matrix = np.array(cost_matrix)

# Viterbi Solver Helper
def solve_viterbi(cost_matrix, gamma, weights):
    dp = np.zeros((N, K))
    paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]
    
    for i in range(1, N):
        w = weights[i]
        for k in range(K):
            transition_costs = dp[i-1] + gamma * w * (np.arange(K) != k)
            best_prev = np.argmin(transition_costs)
            dp[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
            paths[i, k] = best_prev
            
    labels = np.zeros(N, dtype=int)
    labels[N-1] = np.argmin(dp[N-1])
    for i in range(N-2, -1, -1):
        labels[i] = paths[i+1, labels[i+1]]
    return labels

# Helper to compute conditional entropy
def compute_entropy(labels):
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

# ===========================================================================
# EXPERIMENT 1: JUMP VS SWITCH CORRELATION
# ===========================================================================
print("\n--- Running Experiment 1: Jump vs. Switch Correlation ---")
gamma0 = 1000.0
# Modulate by normal weights for Morton
normal_weights_morton = np.ones(N)
for i in range(1, N):
    normal_weights_morton[i] = np.abs(np.dot(normals[i], normals[i-1]))

labels_morton = solve_viterbi(cost_matrix, gamma0, normal_weights_morton)

dists = np.zeros(N)
for i in range(1, N):
    dists[i] = np.linalg.norm(coords[i] - coords[i-1])

switches = (labels_morton[1:] != labels_morton[:-1])
dist_no_switch = dists[1:][~switches]
dist_switch = dists[1:][switches]

avg_dist_no_switch = np.mean(dist_no_switch) if len(dist_no_switch) > 0 else 0
avg_dist_switch = np.mean(dist_switch) if len(dist_switch) > 0 else 0

print(f"  Average distance during NO switch: {avg_dist_no_switch:.3f}")
print(f"  Average distance during SWITCH:    {avg_dist_switch:.3f}")

# ===========================================================================
# EXPERIMENT 2: NEAREST NEIGHBOR ALIGNMENT GAP
# ===========================================================================
print("\n--- Running Experiment 2: Nearest Neighbor Alignment Gap ---")
tree = KDTree(coords)
# Find nearest neighbor for each block (excluding itself)
dists_knn, indices_knn = tree.query(coords, k=2)
nn_indices = indices_knn[:, 1] # 2nd closest is the nearest neighbor excluding itself

a_physical = np.mean([np.abs(np.dot(normals[i], normals[nn_indices[i]])) for i in range(N)])
a_sequence = np.mean([np.abs(np.dot(normals[i], normals[i-1])) for i in range(1, N)])

print(f"  True Physical NN Alignment: {a_physical:.4f}")
print(f"  Morton Sequence Alignment:  {a_sequence:.4f}")
print(f"  Alignment Gap:             {a_physical - a_sequence:+.4f}")

# ===========================================================================
# EXPERIMENT 3: CHEAT-SORT QUICK RUN
# ===========================================================================
print("\n--- Running Experiment 3: Cheat-Sort Quick Run ---")
sorted_indices = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

# Reorder cost matrix and normals
cost_matrix_sorted = cost_matrix[sorted_indices]
normals_sorted = normals[sorted_indices]

normal_weights_sorted = np.ones(N)
for i in range(1, N):
    normal_weights_sorted[i] = np.abs(np.dot(normals_sorted[i], normals_sorted[i-1]))

labels_sorted = solve_viterbi(cost_matrix_sorted, gamma0, normal_weights_sorted)
entropy_sorted = compute_entropy(labels_sorted)
entropy_morton = compute_entropy(labels_morton)

print(f"  Morton Sequence Entropy:   {entropy_morton:.4f} bits")
print(f"  Cheat-Sorted Entropy:      {entropy_sorted:.4f} bits")
print(f"  Entropy Reduction:         {entropy_morton - entropy_sorted:+.4f} bits")

# ===========================================================================
# SAVE ARTIFACT REPORT
# ===========================================================================
report_lines = []
report_lines.append("# Serialization Gap Diagnostics Report\n")
report_lines.append("This report summarizes the results of the three diagnostics verifying the spatial serialization gap in Morton ordering.\n")

report_lines.append("## Experiment 1: Jump vs. Switch Correlation Check")
report_lines.append(f"- **Avg 3D Distance (No Cluster Switch):** `{avg_dist_no_switch:.3f}`")
report_lines.append(f"- **Avg 3D Distance (Cluster Switch):** `{avg_dist_switch:.3f}`")
ratio = avg_dist_switch / avg_dist_no_switch if avg_dist_no_switch > 0 else 1.0
report_lines.append(f"- **Distance Ratio (Switch / No Switch):** `{ratio:.2f}x`\n")

report_lines.append("## Experiment 2: Nearest Neighbor Alignment Gap")
report_lines.append(f"- **True Physical Nearest-Neighbor Normal Alignment:** `{a_physical:.4f}`")
report_lines.append(f"- **Morton Sequence Normal Alignment:** `{a_sequence:.4f}`")
report_lines.append(f"- **Geometric Alignment Gap:** `{a_physical - a_sequence:+.4f}`\n")

report_lines.append("## Experiment 3: Cheat-Sort Quick Run")
report_lines.append(f"- **Morton Sequence Entropy $H(X \\mid X-1)$:** `{entropy_morton:.4f} bits`")
report_lines.append(f"- **Cheat-Sorted Sequence Entropy $H(X \\mid X-1)$:** `{entropy_sorted:.4f} bits`")
report_lines.append(f"- **Entropy Reduction:** `{entropy_morton - entropy_sorted:+.4f} bits`")

artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "serialization_gap_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\nDiagnostics report successfully written to: {report_file.absolute()}")
