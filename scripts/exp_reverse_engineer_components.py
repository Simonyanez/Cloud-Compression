"""
Reverse-Engineering 3D Connected Components
===========================================
Objective:
  Take the 686 optimal 3D connected components (grouped by Phase 2 min-cost labels)
  and inspect their internal geometric properties to discover what physical/geometric
  signal actually holds a component together.

Questions analyzed:
  1. Internal Geometric Homogeneity:
     What is the mean/max normal variance INSIDE a single component vs BETWEEN components?
     If inside normal variance is low, components are physically flat patches.
  2. Planarity & Curvature Profile:
     Are large connected components formed exclusively on high-planarity (flat) areas?
  3. Gradient Direction Coherence:
     Do blocks within the same component share identical 3D gradient directions?
  4. Boundary Characteristics:
     What geometric feature triggers a component boundary? (Normal angle, planarity drop, or position jump?)

Runtime: ~2-3 min for B=16.
"""

import sys, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
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


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}", flush=True)


def normal_angle_deg(n1, n2):
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0, 1))))


def compute_planarity(Vblock):
    Vc = Vblock - Vblock.mean(0)
    cov = Vc.T @ Vc / len(Vblock)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    l1, l2, l3 = eigvals
    return float(np.clip((l2 - l3) / (l1 + 1e-12), 0, 1))


def project_slope(mu_k, v1, v2, unc_norm, degen=1e-3):
    s3d = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
    rn = np.linalg.norm(s3d)
    if rn < degen: s3d = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s3d/rn)*t


def _eval_row(block_id, Vblock, Ablock, metadata, struct_coeffs,
              v1, v2, centroids_3d, slp, slw, unc_norm, dm, lag, q, K):
    dec = Decider(dm, lag); dec._set_vars(q)
    gft = GFTStrategyWraper(); app = Approximator()
    b = Block(metadata); b.Vblock = Vblock; b.Ablock = Ablock
    costs = np.full(K, np.inf)
    try: costs[0] = dec._RDcost(struct_coeffs)[0]
    except: pass
    for k in range(1, K):
        s3d = project_slope(centroids_3d[k-1], v1, v2, unc_norm)
        sg = StructuralGraph(metadata); sg.set_data(Vblock)
        ag = AttributeGraph(sg, s3d, k, slp[k], slw[k])
        Vr = app._spatial_norm(Vblock); Aa = Ablock.copy(); Aa[:,0] = Vr@s3d.T
        ag.set_data(Vblock, Aa)
        try:
            _, coeffs = gft(b, ag); c, _, _ = dec._RDcost(coeffs); costs[k] = c
        except: pass
    return costs


def spherical_em(g3d_norm, K_hybrid, n_iter=20):
    np.random.seed(42)
    idx0 = np.random.choice(len(g3d_norm), K_hybrid, replace=False)
    centroids = g3d_norm[idx0].copy()
    labels = np.zeros(len(g3d_norm), dtype=int)
    for _ in range(n_iter):
        prev = labels.copy()
        sims = np.abs(g3d_norm @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev): break
        for k in range(K_hybrid):
            m = (labels == k)
            if m.sum() == 0: continue
            sg = np.zeros(3); mu = centroids[k]
            for i in np.where(m)[0]:
                sg += np.sign(np.dot(g3d_norm[i], mu)) * g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n
    return centroids, labels


def build_adjacency(coords, radius):
    N = len(coords)
    adj = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(coords[i]-coords[j]) <= radius:
                adj[i].append(j); adj[j].append(i)
    return adj


def get_connected_components(labels, adj, N):
    comp_id = -np.ones(N, dtype=int)
    comps = []
    c_idx = 0
    for i in range(N):
        if comp_id[i] >= 0: continue
        lbl = labels[i]
        queue = deque([i])
        comp_id[i] = c_idx
        members = []
        while queue:
            node = queue.popleft()
            members.append(node)
            for nb in adj[node]:
                if comp_id[nb] < 0 and labels[nb] == lbl:
                    comp_id[nb] = c_idx
                    queue.append(nb)
        comps.append((lbl, members))
        c_idx += 1
    return comp_id, comps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--q-step", type=int, default=24)
    args = parser.parse_args()

    B, K, q = args.block_size, args.k, args.q_step
    K_hybrid = K - 1

    section(f"Reverse-Engineering Optimal 3D Components (B={B}, K={K})")

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    log(f"Partitioning B={B}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=B)
    N = len(blocks)

    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    normals, planarities, g3d_norm, g3d_raw, coords = [], [], [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        coords.append(np.mean(b.Vblock, axis=0))
        normals.append(tangent_data[i]["v3"])
        planarities.append(compute_planarity(b.Vblock))
        Vc = b.Vblock - b.Vblock.mean(0)
        X = np.column_stack((Vc @ tangent_data[i]["v1"], Vc @ tangent_data[i]["v2"]))
        Y = b.Ablock[:,0] - b.Ablock[:,0].mean()
        s2d, *_ = np.linalg.lstsq(X, Y, rcond=None)
        g = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        rn = np.linalg.norm(g)
        g3d_raw.append(rn)
        g3d_norm.append(g/rn if rn > 1e-8 else np.array([0.,0.,1.]))
        b.clear_data()

    normals = np.array(normals)
    planarities = np.array(planarities)
    g3d_norm = np.array(g3d_norm)
    g3d_raw = np.array(g3d_raw)
    coords = np.array(coords)

    gft_comp = GFTStrategyWraper()
    struct_map = {}
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg); struct_map[b.block_id] = coeffs; b.clear_data()

    log("Computing Phase 2 min-cost labels...")
    centroids_3d, _ = spherical_em(g3d_norm, K_hybrid)
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    tasks = []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                      struct_map[b.block_id],
                      tangent_data[i]["v1"], tangent_data[i]["v2"],
                      centroids_3d, slp, slw, g3d_raw[i], dm, lag, q, K))
        b.clear_data()

    rows = Parallel(n_jobs=-1)(delayed(_eval_row)(*t) for t in tasks)
    cost_matrix = np.array(rows)
    p2_labels = np.argmin(cost_matrix, axis=1)

    log("Building 3D adjacency graph...")
    dists_s = []
    for i in np.random.choice(N, min(200, N), replace=False):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        dists_s.append(d.min())
    radius = np.median(dists_s) * 1.8
    adj = build_adjacency(coords, radius)

    comp_id, comps = get_connected_components(p2_labels, adj, N)
    M = len(comps)

    section(f"Analysis of {M} Optimal 3D Components")

    # 1. Size Distribution
    sizes = [len(m) for _, m in comps]
    log(f"Component Sizes: Min={min(sizes)}, Median={int(np.median(sizes))}, Max={max(sizes)}, Mean={np.mean(sizes):.1f}")
    
    # Categorize components
    large_comps = [c for c in comps if len(c[1]) >= 5]
    small_comps = [c for c in comps if len(c[1]) < 5]
    log(f"  Large Components (size >= 5): {len(large_comps)} (covers {sum(len(c[1]) for c in large_comps)} blocks)")
    log(f"  Small Components (size < 5):  {len(small_comps)} (covers {sum(len(c[1]) for c in small_comps)} blocks)")

    # 2. Internal Normal Variation vs Component Size
    section("1. Internal Normal Variance vs Component Size")
    normal_stds_large = []
    normal_max_angles_large = []
    for lbl, members in large_comps:
        comp_normals = normals[members]
        # Mean pairwise normal angle
        dots = np.abs(comp_normals @ comp_normals.T)
        angles = np.degrees(np.arccos(np.clip(dots, 0, 1)))
        normal_stds_large.append(np.mean(angles))
        normal_max_angles_large.append(np.max(angles))

    log(f"Large Components (>= 5 blocks):")
    log(f"  Mean Pairwise Normal Angle INSIDE Component: {np.mean(normal_stds_large):.2f}°")
    log(f"  Max Pairwise Normal Angle INSIDE Component:  {np.mean(normal_max_angles_large):.2f}°")

    # Random baseline: mean normal angle between random blocks
    rand_angles = []
    for _ in range(1000):
        i, j = np.random.choice(N, 2, replace=False)
        rand_angles.append(normal_angle_deg(normals[i], normals[j]))
    log(f"Random Pairwise Normal Angle Across Entire Cloud: {np.mean(rand_angles):.2f}°")

    # 3. Planarity & Gradient Energy Profile
    section("2. Planarity & Gradient Energy Profile")
    planarity_large = [np.mean(planarities[m]) for _, m in large_comps]
    planarity_small = [np.mean(planarities[m]) for _, m in small_comps]
    grad_norm_large = [np.mean(g3d_raw[m]) for _, m in large_comps]
    grad_norm_small = [np.mean(g3d_raw[m]) for _, m in small_comps]

    log(f"Mean Planarity in Large Components: {np.mean(planarity_large):.3f}")
    log(f"Mean Planarity in Small Components: {np.mean(planarity_small):.3f}")
    log(f"Mean Gradient Energy in Large Components: {np.mean(grad_norm_large):.4f}")
    log(f"Mean Gradient Energy in Small Components: {np.mean(grad_norm_small):.4f}")

    # 4. Boundary Analysis
    section("3. Component Boundary Analysis")
    log("Analyzing geometric properties at edges where adjacent blocks have DIFFERENT component IDs...")

    boundary_normal_angles = []
    boundary_planarity_diffs = []
    same_comp_normal_angles = []

    for i in range(N):
        for nb in adj[i]:
            if nb > i:
                ang = normal_angle_deg(normals[i], normals[nb])
                if comp_id[i] != comp_id[nb]:
                    boundary_normal_angles.append(ang)
                    boundary_planarity_diffs.append(abs(planarities[i] - planarities[nb]))
                else:
                    same_comp_normal_angles.append(ang)

    log(f"Normal Angle across Component BOUNDARIES: Mean = {np.mean(boundary_normal_angles):.2f}°, Median = {np.median(boundary_normal_angles):.2f}°")
    log(f"Normal Angle WITHIN Same Component:        Mean = {np.mean(same_comp_normal_angles):.2f}°, Median = {np.median(same_comp_normal_angles):.2f}°")
    log(f"Planarity Difference across BOUNDARIES:     Mean = {np.mean(boundary_planarity_diffs):.3f}")

    # Threshold predictability test:
    # If boundary_normal_angle > T is a good predictor of component boundaries:
    log("\nTesting if Normal Angle Threshold can predict Component Boundaries:")
    for T in [10, 15, 20, 25, 30]:
        tp = sum(1 for a in boundary_normal_angles if a > T)
        fp = sum(1 for a in same_comp_normal_angles if a > T)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(boundary_normal_angles)
        log(f"  Threshold θ > {T}°: Precision = {precision:.1%}, Recall = {recall:.1%}, F1 = {2*precision*recall/(precision+recall+1e-12):.3f}")


if __name__ == "__main__":
    main()
