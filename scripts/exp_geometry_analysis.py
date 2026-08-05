"""
Geometry Analysis Experiment Suite
===================================
Four targeted experiments to identify structural weaknesses in the current
Global 3D Projection codec and find non-overfitted geometric relationships
that could improve cluster quality and reduce overhead.

Run with B=16, K=4 (fast) or B=16, K=8 for more signal.

  EXP A — Morton Path Coherence
    Are Viterbi cluster switches happening at geometrically natural boundaries
    (high curvature / normal discontinuity), or at arbitrary smooth regions?
    If switches are arbitrary, the Morton sequence is the wrong order.

  EXP B — Geometry → Cluster Predictability
    Can a simple geometric rule (normal direction + planarity + gradient mag)
    predict which cluster wins the Phase 2 cost matrix, without any GFT eval?
    High predictability = we can pre-assign blocks and reduce the search space.
    Low predictability = clusters are not geometrically coherent.

  EXP C — Alternative Sequence Orderings
    For a fixed set of Phase 2 cluster assignments, what sequence ordering
    minimises H(X|X-1)?  Compare:
      (0) Morton (baseline)
      (1) Sort by dominant normal hemisphere (X/Y/Z axis)
      (2) Sort by unconstrained gradient direction (quantised into sectors)
      (3) BFS on block spatial adjacency graph
      (4) Sort by planarity (flat blocks first)
    This directly tests whether a cheap pre-ordering can cut overhead without
    changing any GFT computation.

  EXP D — Cluster Assignment Confidence & Spatial Coherence
    For every block compute the "cost gap" (second_best_cost - best_cost).
    High-gap blocks are strongly committed to their cluster.
    Test:  are high-gap blocks spatially clustered?  If yes, the transforms
    are geometrically meaningful.  If no, clusters are gradient-direction
    averages that don't respect surface topology.
    Also computes within-cluster vs between-cluster mean normal-angle variance
    to measure geometric tightness of each cluster.

Usage:
  python scripts/exp_geometry_analysis.py --block-size 16 --k 4
  python scripts/exp_geometry_analysis.py --block-size 16 --k 8
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

project_root = Path("/home/simao/Documents/Repositories/Cloud-Compression")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block
from oracle_em_tangent import extract_block_tangent_planes


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)

def subsection(title):
    print(f"\n  -- {title} --", flush=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_planarity(block, tangent):
    """
    Planarity ∈ [0, 1]:  1 = perfect plane, 0 = isotropic (sphere-like).
    Derived from the PCA eigenvalue ratio of the block's point cloud.
    """
    V = block.Vblock
    V_c = V - V.mean(axis=0)
    cov = V_c.T @ V_c / len(V)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]  # descending
    lam1, lam2, lam3 = eigvals
    total = lam1 + lam2 + lam3
    if total < 1e-12:
        return 0.0
    # planarity: (lam2 - lam3) / lam1  — high when surface is flat but elongated
    # linearity: (lam1 - lam2) / lam1  — high when surface is linear
    # sphericity: lam3 / lam1
    planarity = (lam2 - lam3) / (lam1 + 1e-12)
    return float(np.clip(planarity, 0, 1))


def compute_normal_angle_deg(n1, n2):
    """Angle in degrees between two surface normals (unsigned, 0-90)."""
    cos_a = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def compute_entropy(labels, K):
    cluster_sizes = np.bincount(labels, minlength=K)
    trans = np.ones((K, K))
    for i in range(1, len(labels)):
        trans[labels[i-1], labels[i]] += 1
    trans_probs = trans / trans.sum(axis=1, keepdims=True)
    h = 0.0
    for prev in range(K):
        p_prev = cluster_sizes[prev] / len(labels)
        for cur in range(K):
            p = trans_probs[prev, cur]
            if p > 0 and p_prev > 0:
                h -= p_prev * p * np.log2(p)
    return h


def spherical_kmeanspp(grads, K_hybrid):
    np.random.seed(42)
    first_idx = np.random.choice(len(grads))
    c0 = grads[first_idx]
    n0 = np.linalg.norm(c0)
    centroids = [c0 / (n0 if n0 > 1e-8 else 1.0)]
    for _ in range(1, K_hybrid):
        dists = np.array([
            max(0.0, min(1.0 - np.abs(np.dot(g, c))**2 for c in centroids))
            for g in grads
        ])
        s = dists.sum()
        probs = dists / s if s > 0 else np.ones(len(grads)) / len(grads)
        nc = grads[np.random.choice(len(grads), p=probs)]
        nc_n = np.linalg.norm(nc)
        centroids.append(nc / (nc_n if nc_n > 1e-8 else 1.0))
    return np.array(centroids)


def em_run(g3d_norm, K_hybrid):
    centroids = spherical_kmeanspp(g3d_norm, K_hybrid)
    labels = np.zeros(len(g3d_norm), dtype=int)
    for _ in range(20):
        prev = labels.copy()
        sims = np.abs(g3d_norm @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev):
            break
        for k in range(K_hybrid):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            sg = np.zeros(3)
            for i in idx:
                sg += np.sign(np.dot(g3d_norm[i], centroids[k])) * g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8:
                centroids[k] = sg / n
    return centroids, labels


def build_block_adjacency(blocks, coords, radius_factor=1.5):
    """
    Build a sparse adjacency list: two blocks are neighbours if their
    centroids are within radius_factor * expected_spacing of each other.
    """
    N = len(blocks)
    # Estimate expected spacing from median NN distance
    if N > 2000:
        sample = np.random.choice(N, 500, replace=False)
        c_sample = coords[sample]
    else:
        c_sample = coords
    dists = []
    for i in range(min(200, len(c_sample))):
        d = np.linalg.norm(c_sample - c_sample[i], axis=1)
        d[i] = np.inf
        dists.append(np.min(d))
    spacing = np.median(dists) * radius_factor

    adj = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(coords[i] - coords[j]) <= spacing:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_ordering(adj, N):
    """BFS traversal order starting from node 0."""
    visited = [False] * N
    order = []
    queue = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    # Append any disconnected nodes
    for i in range(N):
        if not visited[i]:
            order.append(i)
    return np.array(order)


def reorder_entropy(labels, order, K):
    """Compute H(X|X-1) for labels reindexed by the given traversal order."""
    reordered = labels[order]
    return compute_entropy(reordered, K)


# ---------------------------------------------------------------------------
# Cost evaluation worker
# ---------------------------------------------------------------------------

def _eval_row(block_id, Vblock, Ablock, metadata, struct_coeffs,
              v1, v2, centroids_3d, slp, slw, g3d_raw_norm,
              decider_mode, lagrange, q_step, K):
    dec = Decider(decider_mode, lagrange)
    dec._set_vars(q_step)
    gft = GFTStrategyWraper()
    app = Approximator()
    b = Block(metadata)
    b.Vblock = Vblock
    b.Ablock = Ablock

    costs = np.full(K, np.inf)
    try:
        costs[0] = dec._RDcost(struct_coeffs)[0]
    except Exception:
        pass

    for k in range(1, K):
        mu_k = centroids_3d[k - 1]
        s2d = np.array([np.dot(mu_k, v1), np.dot(mu_k, v2)])
        s3d_raw = s2d[0]*v1 + s2d[1]*v2
        rn = np.linalg.norm(s3d_raw)
        if rn < 1e-6:
            s3d_raw = v1 * 1e-4
            rn = np.linalg.norm(s3d_raw)
        target = g3d_raw_norm if g3d_raw_norm > 1e-8 else rn
        s3d = (s3d_raw / rn) * target

        sg = StructuralGraph(metadata)
        sg.set_data(Vblock)
        ag = AttributeGraph(sg, s3d, k, slp[k], slw[k])
        V_rot = app._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ s3d.T
        ag.set_data(Vblock, A_app)
        try:
            _, coeffs = gft(b, ag)
            c, _, _ = dec._RDcost(coeffs)
            costs[k] = c
        except Exception:
            pass
    return costs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--gamma", type=float, default=2000.0,
                   help="Single gamma for Viterbi (used only for exp A)")
    return p.parse_args()


def main():
    args = parse_args()
    B = args.block_size
    K = args.k
    q_step = args.q_step
    gamma = args.gamma

    section(f"Geometry Analysis Suite  (B={B}, K={K}, Q={q_step}, gamma={gamma})")

    # ------------------------------------------------------------------
    # Setup: load data, partition, extract geometry
    # ------------------------------------------------------------------
    log("Loading point cloud...")
    params = load_experiment_config(project_root / "config/base_config.yaml")
    colourist = Colourist()
    pc = PointCloud.from_file(
        project_root / params.sequential_params.point_cloud_path, "ply", params.pointcloud
    )
    pc.transform_attributes(colourist._RGBtoYUV)
    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional

    log(f"Partitioning into blocks of B={B}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=B)
    N = len(blocks)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)
    log(f"  {N} blocks, {total_v} total voxels.")

    log("Extracting tangent planes...")
    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    log("Pre-computing geometry features for all blocks...")
    normals = []
    coords = []
    g3d_norm_all = []
    g3d_raw_norms = []
    planarities = []

    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        c = np.mean(block.Vblock, axis=0)
        coords.append(c)
        normals.append(tangent_data[i]["v3"])

        V_c = block.Vblock - c
        X = np.column_stack((V_c @ tangent_data[i]["v1"], V_c @ tangent_data[i]["v2"]))
        Y = block.Ablock[:, 0] - np.mean(block.Ablock[:, 0])
        s2d, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        g3d = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        raw_norm = np.linalg.norm(g3d)
        g3d_raw_norms.append(raw_norm)
        g3d_norm_all.append(g3d / raw_norm if raw_norm > 1e-8 else np.array([0.,0.,1.]))

        planarities.append(compute_planarity(block, tangent_data[i]))
        block.clear_data()

    normals = np.array(normals)
    coords = np.array(coords)
    g3d_norm_all = np.array(g3d_norm_all)
    g3d_raw_norms = np.array(g3d_raw_norms)
    planarities = np.array(planarities)
    morton_order = np.arange(N)

    log("Pre-computing structural GFT coefficients...")
    gft_comp = GFTStrategyWraper()
    struct_coeffs_map = {}
    for block in tqdm(blocks, desc="Structural GFT", ncols=80):
        block.init_data(pc.V, pc.A)
        sg = StructuralGraph(block.metadata)
        sg.set_data(block.Vblock)
        _, coeffs = gft_comp(block, sg)
        struct_coeffs_map[block.block_id] = coeffs
        block.clear_data()

    # Run spherical EM to get cluster centroids and labels
    log(f"Running Spherical EM (K={K})...")
    K_hybrid = K - 1
    centroids_3d, em_labels = em_run(g3d_norm_all, K_hybrid)
    log(f"  EM converged. Cluster sizes: "
        f"{[int(np.sum(em_labels==k)) for k in range(K_hybrid)]}")

    # Evaluate cost matrix
    log("Evaluating Phase 2 cost matrix (parallel)...")
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    eval_tasks = []
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        eval_tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            struct_coeffs_map[block.block_id],
            tangent_data[i]["v1"], tangent_data[i]["v2"],
            centroids_3d, slp, slw, g3d_raw_norms[i],
            decider_mode, lagrange, q_step, K
        ))
        block.clear_data()

    t0 = time.time()
    cost_rows = Parallel(n_jobs=-1)(delayed(_eval_row)(*t) for t in eval_tasks)
    cost_matrix = np.array(cost_rows)
    log(f"  Cost matrix done in {time.time()-t0:.1f}s.")

    # Phase 2 labels (argmin of cost_matrix)
    p2_labels = np.argmin(cost_matrix, axis=1)
    structural_cost = cost_matrix[:, 0].sum()

    # Run Viterbi
    log("Running Viterbi...")
    nw = np.ones(N)
    for i in range(1, N):
        nw[i] = np.abs(np.dot(normals[i], normals[i-1]))

    dp = np.full((N, K), np.inf)
    paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i-1] + gamma * w * (np.arange(K) != k)
            bp = np.argmin(tc)
            dp[i, k] = cost_matrix[i, k] + tc[bp]
            paths[i, k] = bp
    vit_labels = np.zeros(N, dtype=int)
    vit_labels[N-1] = np.argmin(dp[N-1])
    for i in range(N-2, -1, -1):
        vit_labels[i] = paths[i+1, vit_labels[i+1]]

    vit_entropy = compute_entropy(vit_labels, K)
    vit_cost = sum(cost_matrix[i, vit_labels[i]] for i in range(N))
    vit_savings = (structural_cost - vit_cost) / structural_cost * 100
    log(f"  Viterbi: savings={vit_savings:.3f}%, H={vit_entropy:.4f}")

    # Cost gap per block
    sorted_costs = np.sort(cost_matrix, axis=1)
    cost_gap = sorted_costs[:, 1] - sorted_costs[:, 0]  # 2nd best - best
    cost_gap = np.clip(cost_gap, 0, None)

    # =====================================================================
    # EXP A: Morton Path Coherence
    # =====================================================================
    section("EXP A — Morton Path Coherence")
    log("Checking whether Viterbi switches align with geometric transitions...")

    switch_mask = np.zeros(N, dtype=bool)
    switch_mask[1:] = (vit_labels[1:] != vit_labels[:-1])
    n_switches = switch_mask.sum()

    # Normal angle at each transition
    normal_angles = np.array([
        compute_normal_angle_deg(normals[i], normals[i-1]) for i in range(1, N)
    ])
    gradient_angles = np.array([
        compute_normal_angle_deg(g3d_norm_all[i], g3d_norm_all[i-1]) for i in range(1, N)
    ])

    switch_normal_angles = normal_angles[switch_mask[1:]]
    no_switch_normal_angles = normal_angles[~switch_mask[1:]]
    switch_grad_angles = gradient_angles[switch_mask[1:]]
    no_switch_grad_angles = gradient_angles[~switch_mask[1:]]

    subsection("Normal angles at Morton transitions")
    log(f"  ALL transitions:    mean={normal_angles.mean():.2f}°, "
        f"median={np.median(normal_angles):.2f}°, "
        f"p90={np.percentile(normal_angles,90):.2f}°")
    log(f"  SWITCH transitions: mean={switch_normal_angles.mean():.2f}°, "
        f"median={np.median(switch_normal_angles):.2f}°, "
        f"p90={np.percentile(switch_normal_angles,90):.2f}°  (n={len(switch_normal_angles)})")
    log(f"  STAY transitions:   mean={no_switch_normal_angles.mean():.2f}°, "
        f"median={np.median(no_switch_normal_angles):.2f}°, "
        f"p90={np.percentile(no_switch_normal_angles,90):.2f}°  (n={len(no_switch_normal_angles)})")

    subsection("Gradient direction angles at Morton transitions")
    log(f"  SWITCH: mean={switch_grad_angles.mean():.2f}°, median={np.median(switch_grad_angles):.2f}°")
    log(f"  STAY:   mean={no_switch_grad_angles.mean():.2f}°, median={np.median(no_switch_grad_angles):.2f}°")

    # Key signal: if switch_mean ≈ no_switch_mean → switches are geometrically arbitrary (bad)
    normal_angle_diff = switch_normal_angles.mean() - no_switch_normal_angles.mean()
    grad_angle_diff = switch_grad_angles.mean() - no_switch_grad_angles.mean()
    subsection("Geometric signal at switches (positive = switches at larger angles = good)")
    log(f"  Normal angle delta (switch - stay): {normal_angle_diff:+.2f}°")
    log(f"  Gradient angle delta (switch - stay): {grad_angle_diff:+.2f}°")

    # Distribution of switches by planarity of block
    switch_planarity = planarities[np.where(switch_mask)[0]]
    no_switch_planarity = planarities[np.where(~switch_mask)[0]]
    log(f"  Planarity at SWITCH blocks: mean={switch_planarity.mean():.3f}  "
        f"(no-switch: {no_switch_planarity.mean():.3f})")

    # Fraction of switches where the normal angle IS above a threshold
    for thresh in [5, 10, 20]:
        frac = (switch_normal_angles > thresh).mean()
        log(f"  Fraction of switches where normal_angle > {thresh}°: {frac:.1%}")

    # =====================================================================
    # EXP B: Geometry → Cluster Predictability
    # =====================================================================
    section("EXP B — Geometry-to-Cluster Predictability")
    log("Testing if a simple geometric rule predicts Phase 2 cluster assignment...")

    # Feature set: normal direction (3), dominant gradient direction quantised (hemisphere, 6),
    # planarity (1), gradient magnitude (1)
    # Target: p2_labels

    # Simple test: for each cluster, what are the dominant normal hemisphere fractions?
    subsection("Normal hemisphere distribution per Phase 2 cluster")
    hemispheres = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    hem_idx = np.argmax(np.abs(normals), axis=1) * 2  # axis index * 2
    hem_sign = (normals[np.arange(N), np.argmax(np.abs(normals), axis=1)] > 0).astype(int)
    hem_label = hem_idx + hem_sign  # 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-

    for k in range(K):
        mask = (p2_labels == k)
        n_k = mask.sum()
        if n_k == 0:
            continue
        hem_counts = np.bincount(hem_label[mask], minlength=6)
        dominant = hemispheres[np.argmax(hem_counts)]
        dominant_frac = hem_counts.max() / n_k
        log(f"  Cluster {k}: n={n_k} ({100*n_k/N:.1f}%) | "
            f"dominant normal hemisphere: {dominant} ({100*dominant_frac:.1f}%) | "
            f"mean planarity: {planarities[mask].mean():.3f}")

    subsection("Gradient direction separation between clusters")
    for k in range(K):
        mask = (p2_labels == k)
        if mask.sum() == 0:
            continue
        grads_k = g3d_norm_all[mask]
        if len(grads_k) > 1:
            # Within-cluster mean pairwise angular distance
            sample = grads_k[:min(200, len(grads_k))]
            dots = np.abs(sample @ sample.T)
            np.fill_diagonal(dots, np.nan)
            mean_cos = np.nanmean(dots)
            mean_angle = np.degrees(np.arccos(np.clip(mean_cos, 0, 1)))
            log(f"  Cluster {k}: within-cluster mean gradient angle = {mean_angle:.2f}° "
                f"(0°=tight, 90°=dispersed)")

    # Naive geometry-based prediction: assign each block to the centroid with
    # highest absolute cosine similarity between block's normal and cluster centroid
    # (using centroid as a normal-like direction — a geometry-only oracle)
    em_predictions = np.zeros(N, dtype=int)  # 0 = predict flat
    for i in range(N):
        sims = np.abs(normals[i] @ centroids_3d.T)
        best_k = np.argmax(sims)
        # Only predict a non-flat cluster if the similarity is high enough
        if sims[best_k] > 0.7:
            em_predictions[i] = best_k + 1
        # else: predict flat (cluster 0)

    # Agreement between geometry-only prediction and Phase 2 assignment
    agreement = np.mean(em_predictions == p2_labels)
    log(f"\n  Geometry-only (normal cosine) prediction accuracy vs Phase 2: "
        f"{agreement:.1%}")

    # Gradient-based prediction (EM output)
    grad_predictions = em_labels + 1  # EM assigns 0..K_hybrid-1, Phase 2 uses 1..K-1
    grad_predictions_full = np.zeros(N, dtype=int)
    for i in range(N):
        # Compare unconstrained gradient to each centroid
        sims = np.abs(g3d_norm_all[i] @ centroids_3d.T)
        best_k = np.argmax(sims)
        if g3d_raw_norms[i] > np.percentile(g3d_raw_norms, 30):
            grad_predictions_full[i] = best_k + 1
        # else: low-energy gradient → predict flat

    grad_agreement = np.mean(grad_predictions_full == p2_labels)
    log(f"  Gradient direction prediction accuracy vs Phase 2: {grad_agreement:.1%}")

    flat_frac = (p2_labels == 0).mean()
    log(f"  Baseline (always predict cluster 0): {flat_frac:.1%}")

    # What fraction of Phase 2 cluster assignments come from blocks where
    # the gradient energy is low (likely flat)?
    low_energy_thresh = np.percentile(g3d_raw_norms, 40)
    for k in range(K):
        mask = (p2_labels == k)
        if mask.sum() == 0:
            continue
        low_e = (g3d_raw_norms[mask] < low_energy_thresh).mean()
        log(f"  Cluster {k}: fraction of low-energy blocks = {low_e:.1%} "
            f"({'BAD: non-flat cluster gets flat blocks' if k>0 and low_e>0.4 else 'ok'})")

    # =====================================================================
    # EXP C: Alternative Sequence Orderings
    # =====================================================================
    section("EXP C — Alternative Sequence Orderings (for fixed Phase 2 labels)")
    log("Computing H(X|X-1) for different block traversal orderings...")
    log(f"  (Using Phase 2 cost-minimum labels, K={K})")

    results = {}

    # 0. Morton (baseline)
    h_morton = compute_entropy(p2_labels, K)
    results["(0) Morton order (baseline)"] = h_morton
    log(f"  (0) Morton order:                H = {h_morton:.4f}")

    # 1. Sort by dominant normal hemisphere (groups surface patches together)
    normal_sort_order = np.argsort(hem_label)
    h_normal = reorder_entropy(p2_labels, normal_sort_order, K)
    results["(1) Sort by normal hemisphere"] = h_normal
    log(f"  (1) Sort by normal hemisphere:   H = {h_normal:.4f}  "
        f"(Δ = {h_normal - h_morton:+.4f})")

    # 2. Sort by gradient direction (quantise into 8 octants)
    octants = np.zeros(N, dtype=int)
    for i in range(N):
        g = g3d_norm_all[i]
        octants[i] = (int(g[0] >= 0) * 4 + int(g[1] >= 0) * 2 + int(g[2] >= 0))
    grad_sort_order = np.argsort(octants)
    h_grad = reorder_entropy(p2_labels, grad_sort_order, K)
    results["(2) Sort by gradient octant"] = h_grad
    log(f"  (2) Sort by gradient octant:     H = {h_grad:.4f}  "
        f"(Δ = {h_grad - h_morton:+.4f})")

    # 3. Sort by planarity (most planar first → likely consistent cluster)
    plan_sort_order = np.argsort(-planarities)
    h_plan = reorder_entropy(p2_labels, plan_sort_order, K)
    results["(3) Sort by planarity (desc)"] = h_plan
    log(f"  (3) Sort by planarity desc:      H = {h_plan:.4f}  "
        f"(Δ = {h_plan - h_morton:+.4f})")

    # 4. BFS on spatial adjacency graph (spatially local order)
    log("  Building block adjacency graph for BFS (may take a moment)...")
    t_bfs = time.time()
    adj = build_block_adjacency(blocks, coords, radius_factor=1.8)
    n_adj_edges = sum(len(v) for v in adj.values()) // 2
    log(f"  Adjacency graph: {n_adj_edges} edges (avg degree {2*n_adj_edges/N:.1f}), "
        f"built in {time.time()-t_bfs:.1f}s")
    bfs_order = bfs_ordering(adj, N)
    h_bfs = reorder_entropy(p2_labels, bfs_order, K)
    results["(4) BFS spatial adjacency"] = h_bfs
    log(f"  (4) BFS spatial order:           H = {h_bfs:.4f}  "
        f"(Δ = {h_bfs - h_morton:+.4f})")

    # 5. Sort by cluster label directly (oracle upper bound on compression)
    cluster_sort_order = np.argsort(p2_labels)
    h_oracle = reorder_entropy(p2_labels, cluster_sort_order, K)
    results["(5) Sort by cluster (oracle)"] = h_oracle
    log(f"  (5) Sort by cluster [ORACLE]:    H = {h_oracle:.4f}  "
        f"(Δ = {h_oracle - h_morton:+.4f})  — lower bound on achievable H")

    subsection("Summary")
    best_name = min(results, key=results.get)
    log(f"  Best ordering: '{best_name}' with H = {results[best_name]:.4f}")
    log(f"  Potential overhead reduction vs Morton: "
        f"ΔH = {results[best_name] - h_morton:+.4f}")
    n_voxels_per_block = total_v / N
    overhead_reduction_bpv = (h_morton - results[best_name]) / n_voxels_per_block
    log(f"  Equivalent overhead BPV improvement: ~{overhead_reduction_bpv:.5f} bpv")

    # =====================================================================
    # EXP D: Cluster Assignment Confidence & Spatial Coherence
    # =====================================================================
    section("EXP D — Cluster Confidence & Spatial Coherence")

    subsection("Cost gap distribution per cluster")
    for k in range(K):
        mask = (p2_labels == k)
        n_k = mask.sum()
        if n_k == 0:
            continue
        gaps_k = cost_gap[mask]
        # Clip inf gaps (happens when all alternatives are inf — should be rare)
        finite_gaps = gaps_k[np.isfinite(gaps_k)]
        log(f"  Cluster {k}: n={n_k} | "
            f"gap median={np.median(finite_gaps):.2f}, "
            f"p25={np.percentile(finite_gaps,25):.2f}, "
            f"p75={np.percentile(finite_gaps,75):.2f} | "
            f"high-conf (>50th pct overall): "
            f"{(finite_gaps > np.percentile(cost_gap[np.isfinite(cost_gap)], 50)).mean():.1%}")

    subsection("Spatial clustering of high-confidence blocks")
    global_p50 = np.percentile(cost_gap[np.isfinite(cost_gap)], 50)
    high_conf_mask = np.isfinite(cost_gap) & (cost_gap > global_p50)
    high_conf_idx = np.where(high_conf_mask)[0]
    low_conf_idx = np.where(~high_conf_mask)[0]

    if len(high_conf_idx) > 2 and len(low_conf_idx) > 2:
        # Mean distance to nearest high-conf neighbour vs. random
        sample_hc = high_conf_idx[:min(300, len(high_conf_idx))]
        sample_lc = low_conf_idx[:min(300, len(low_conf_idx))]

        mean_nn_hc = np.mean([
            np.min(np.linalg.norm(coords[high_conf_idx] - coords[i], axis=1) + 1e18*(i==high_conf_idx))
            if len(high_conf_idx) > 1 else 0.0
            for i in sample_hc
        ])
        mean_nn_lc = np.mean([
            np.min(np.linalg.norm(coords[high_conf_idx] - coords[i], axis=1))
            if len(high_conf_idx) > 0 else 0.0
            for i in sample_lc
        ])
        log(f"  Mean NN dist from high-conf → nearest high-conf: {mean_nn_hc:.3f}")
        log(f"  Mean NN dist from low-conf → nearest high-conf:  {mean_nn_lc:.3f}")
        ratio = mean_nn_hc / (mean_nn_lc + 1e-12)
        log(f"  Ratio (<1 = high-conf blocks cluster together): {ratio:.3f}")
        if ratio < 0.8:
            log("  → HIGH-CONF BLOCKS ARE SPATIALLY CLUSTERED. "
                "Transforms are geometrically meaningful.")
        elif ratio > 1.2:
            log("  → HIGH-CONF BLOCKS ARE SPATIALLY DISPERSED. "
                "Cluster boundaries are not geometric.")
        else:
            log("  → MIXED SIGNAL. No strong spatial pattern.")

    subsection("Within-cluster vs between-cluster normal angle variance")
    within_angles = []
    between_angles = []
    sample_per_cluster = 100
    for k in range(K):
        idx_k = np.where(p2_labels == k)[0]
        if len(idx_k) < 2:
            continue
        s = idx_k[:min(sample_per_cluster, len(idx_k))]
        for i in range(len(s)):
            for j in range(i+1, min(i+10, len(s))):
                within_angles.append(compute_normal_angle_deg(normals[s[i]], normals[s[j]]))
        # Between: compare with random block from different cluster
        for _ in range(min(50, len(s))):
            other_mask = p2_labels != k
            other_idx = np.where(other_mask)[0]
            if len(other_idx) == 0:
                continue
            j = np.random.choice(other_idx)
            i_s = np.random.choice(s)
            between_angles.append(compute_normal_angle_deg(normals[i_s], normals[j]))

    if within_angles and between_angles:
        wa = np.array(within_angles)
        ba = np.array(between_angles)
        log(f"  Within-cluster normal angles:  mean={wa.mean():.2f}°, std={wa.std():.2f}°")
        log(f"  Between-cluster normal angles: mean={ba.mean():.2f}°, std={ba.std():.2f}°")
        separation = ba.mean() - wa.mean()
        log(f"  Cluster normal separation (between-within): {separation:+.2f}°")
        if separation > 10:
            log("  → GOOD: clusters are geometrically separated in normal space.")
        elif separation > 3:
            log("  → WEAK: some geometric separation but room to improve.")
        else:
            log("  → POOR: clusters do not separate by normal direction. "
                "Consider using planarity or curvature as a clustering signal.")

    # =====================================================================
    # Final summary and recommendations
    # =====================================================================
    section("SUMMARY & RECOMMENDATIONS")

    log(f"EXP A — Morton coherence:")
    log(f"  Switch normal angle delta: {normal_angle_diff:+.2f}°  "
        f"({'switches at boundaries ✓' if normal_angle_diff > 5 else 'switches are arbitrary ✗'})")
    log(f"  Switch gradient angle delta: {grad_angle_diff:+.2f}°")

    log(f"\nEXP B — Geometry predictability:")
    log(f"  Normal-based accuracy: {agreement:.1%}  "
        f"({'informative ✓' if agreement > flat_frac + 0.10 else 'not better than baseline ✗'})")
    log(f"  Gradient-based accuracy: {grad_agreement:.1%}  "
        f"({'informative ✓' if grad_agreement > flat_frac + 0.10 else 'not better than baseline ✗'})")

    log(f"\nEXP C — Best ordering: '{best_name}'")
    best_h = results[best_name]
    log(f"  H reduction vs Morton: {h_morton - best_h:+.4f}  "
        f"({'significant ✓' if h_morton - best_h > 0.05 else 'marginal'})")

    log(f"\nEXP D — Spatial coherence:")
    if within_angles and between_angles:
        log(f"  Cluster normal separation: {separation:+.2f}°  "
            f"({'well-separated ✓' if separation > 10 else 'poorly separated ✗'})")

    log("\nKey questions answered by these experiments:")
    log("  1. Are switches geometrically motivated? → EXP A")
    log("  2. Can geometry replace GFT evaluation for pre-assignment? → EXP B")
    log("  3. Does the Morton traversal order hurt overhead? → EXP C")
    log("  4. Are clusters geometrically tight or arbitrary? → EXP D")


if __name__ == "__main__":
    main()
