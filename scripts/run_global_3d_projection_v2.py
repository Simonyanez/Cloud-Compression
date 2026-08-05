"""
Global 3D Codebook Projection Sweep - Version 2
================================================
Fixes over v1 (run_global_3d_projection_fixed.py):

  FIX 1 - Magnitude Normalization (Experiment 2):
    After projecting a global unit-norm centroid onto the local 2D tangent
    plane, rescale the 3D slope vector so its magnitude matches the block's
    own unconstrained gradient energy:
        s3d_final = (s3d_proj / ||s3d_proj||) * ||unconstrained_g3d_block||
    Prevents near-orthogonal projections from producing near-zero degenerate
    attribute graphs on surface bends.

  FIX 2 - Live Label Update after each R-D Refinement Iteration:
    Phase 1.5 now re-assigns sample block labels after every refinement
    iteration based on R-D cost (not just EM cosine similarity). The grid
    search therefore uses R-D-consistent cluster membership, not stale EM
    labels.

  FIX 3 - Boundary-Aware Grid Search with Adaptive Extension:
    If the (SLP, SLW) winner sits on a grid boundary, we log a WARN and run
    up to 2 additional passes extending the grid past that boundary.

  FIX 4 - Degenerate Projection Guard:
    Near-zero projected slopes get a small tangent-plane perturbation so the
    attribute graph stays non-trivial. Degenerate counts are reported.

  NOTE on SLP:
    SLP == 0 means no self-loops => graph collapses to the flat structural
    graph. That is already Cluster 0. Minimum SLP is 0.05 (never 0).
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

project_root = Path("/home/simao/Documents/Repositories/Cloud-Compression")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block
from oracle_em_tangent import extract_block_tangent_planes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Global 3D Projection v2 (Magnitude Norm + Live Labels + Boundary Search)"
    )
    parser.add_argument("--block-sizes", type=str, default="16,32")
    parser.add_argument("--k-values", type=str, default="4,8,12,16")
    parser.add_argument("--gammas", type=str,
                        default="250,500,750,1000,1500,2000,2500,3000,3500,4000,5000,7500")
    parser.add_argument("--q-step", type=int, default=24)
    parser.add_argument("--rd-iters", type=int, default=5)
    parser.add_argument("--sample-ratio", type=float, default=0.01)
    parser.add_argument("--n-strata", type=int, default=5)
    parser.add_argument("--config", type=str,
                        default=str(project_root / "config/base_config.yaml"))
    parser.add_argument("--output-dir", type=str,
                        default="/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
    parser.add_argument("--output-name", type=str, default="global_3d_v2_results.md")
    parser.add_argument("--degen-thresh", type=float, default=1e-3,
                        help="Min projected slope magnitude before perturbation. Default: 1e-3")
    return parser.parse_args()


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def warn(msg):
    log(msg, level="WARN")


# ---------------------------------------------------------------------------
# Spherical EM helpers
# ---------------------------------------------------------------------------

def spherical_kmeanspp(grads, K_hybrid):
    np.random.seed(42)
    first_idx = np.random.choice(len(grads))
    c0 = grads[first_idx]
    norm0 = np.linalg.norm(c0)
    centroids = [c0 / (norm0 if norm0 > 1e-8 else 1.0)]
    for _ in range(1, K_hybrid):
        dists = np.array([
            max(0.0, min(1.0 - np.abs(np.dot(g, c)) ** 2 for c in centroids))
            for g in grads
        ])
        s = dists.sum()
        probs = dists / s if s > 0 else np.ones(len(grads)) / len(grads)
        nc = grads[np.random.choice(len(grads), p=probs)]
        nc_norm = np.linalg.norm(nc)
        centroids.append(nc / (nc_norm if nc_norm > 1e-8 else 1.0))
    return np.array(centroids)


def em_assign(g3d_norm, centroids_3d):
    K_hybrid = len(centroids_3d)
    sims = np.array([
        [np.abs(np.dot(g, centroids_3d[k])) for k in range(K_hybrid)]
        for g in g3d_norm
    ])
    return np.argmax(sims, axis=1)


def em_update_centroids(g3d_norm, labels_em, K_hybrid, centroids_3d):
    new_centroids = centroids_3d.copy()
    for k in range(K_hybrid):
        idx = np.where(labels_em == k)[0]
        if len(idx) == 0:
            continue
        sum_g = np.zeros(3)
        mu_k = centroids_3d[k]
        for i in idx:
            g = g3d_norm[i]
            sign = np.sign(np.dot(g, mu_k))
            if sign == 0:
                sign = 1.0
            sum_g += sign * g
        norm_sum = np.linalg.norm(sum_g)
        if norm_sum > 1e-8:
            new_centroids[k] = sum_g / norm_sum
        else:
            new_centroids[k] = centroids_3d[k] / (np.linalg.norm(centroids_3d[k]) + 1e-12)
    return new_centroids


# ---------------------------------------------------------------------------
# FIX 1: Magnitude-normalised projection
# ---------------------------------------------------------------------------

def project_centroid_to_slope(mu_k, v1, v2, unconstrained_norm, degen_thresh=1e-3):
    """
    Project global unit-norm centroid mu_k onto the block's local tangent plane,
    then rescale to match the block's unconstrained gradient energy.

    Returns (s3d, is_degenerate).
    NOTE: SLP=0 is never passed here — the caller always uses slp >= 0.05.
    """
    s2d_proj = np.array([np.dot(mu_k, v1), np.dot(mu_k, v2)])
    s3d_raw = s2d_proj[0] * v1 + s2d_proj[1] * v2
    raw_norm = np.linalg.norm(s3d_raw)
    is_degenerate = raw_norm < degen_thresh

    if is_degenerate:
        # Small perturbation in the tangent plane — direction from mu_k seed
        rng = np.random.default_rng(seed=int(abs(mu_k[0] * 1e6)) % (2**31))
        perturb = rng.standard_normal(3)
        # Project onto tangent plane
        perturb = perturb - np.dot(perturb, v1) * v1 - np.dot(perturb, v2) * v2
        pn = np.linalg.norm(perturb)
        s3d_raw = (perturb / pn * degen_thresh) if pn > 1e-12 else (v1 * degen_thresh)
        raw_norm = np.linalg.norm(s3d_raw)

    # Rescale direction to match block's unconstrained gradient energy (FIX 1)
    target_mag = unconstrained_norm if unconstrained_norm > 1e-8 else raw_norm
    s3d = (s3d_raw / raw_norm) * target_mag
    return s3d, is_degenerate


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_stratified_indices(indices, max_samples):
    n = len(indices)
    if n <= max_samples:
        return np.array(indices, dtype=int)
    pos = np.linspace(0, n - 1, max_samples, dtype=int)
    return np.array(indices)[pos]


def compute_entropy(labels, K):
    cluster_sizes = np.bincount(labels, minlength=K)
    trans = np.ones((K, K))
    for i in range(1, len(labels)):
        trans[labels[i - 1], labels[i]] += 1
    trans_probs = trans / trans.sum(axis=1, keepdims=True)
    h = 0.0
    for prev in range(K):
        p_prev = cluster_sizes[prev] / len(labels)
        for cur in range(K):
            p = trans_probs[prev, cur]
            if p > 0 and p_prev > 0:
                h -= p_prev * p * np.log2(p)
    return h


# ---------------------------------------------------------------------------
# Block preparation
# ---------------------------------------------------------------------------

def prepare_blocks(pc, bsize, degen_thresh=1e-3):
    t0 = time.time()
    log(f"Partitioning point cloud into blocks of size B={bsize}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=bsize)
    N = len(blocks)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)
    log(f"Extracted {N} blocks containing {total_v} total voxels.")

    log("Extracting block tangent planes (PCA)...")
    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    coords, normals = [], []
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        coords.append(np.mean(block.Vblock, axis=0))
        normals.append(tangent_data[i]["v3"])
        block.clear_data()

    log("Pre-computing structural GFT coefficients for all blocks...")
    gft_computer = GFTStrategyWraper()
    structural_coeffs_map = {}
    for block in tqdm(blocks, desc="Pre-computing Structural GFT"):
        block.init_data(pc.V, pc.A)
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs = gft_computer(block, s_graph)
        structural_coeffs_map[block.block_id] = coeffs
        block.clear_data()

    log("Extracting 3D attribute gradients (unconstrained)...")
    g3d_norm = []
    g3d_raw_norms = []
    degen_gradient_count = 0
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        V_c = block.Vblock - np.mean(block.Vblock, axis=0)
        X = np.column_stack((V_c @ tangent_data[i]["v1"], V_c @ tangent_data[i]["v2"]))
        Y = block.Ablock[:, 0] - np.mean(block.Ablock[:, 0])
        s2d, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        g3d = s2d[0] * tangent_data[i]["v1"] + s2d[1] * tangent_data[i]["v2"]
        raw_norm = np.linalg.norm(g3d)
        g3d_raw_norms.append(raw_norm)
        if raw_norm > 1e-8:
            g3d_norm.append(g3d / raw_norm)
        else:
            g3d_norm.append(np.array([0.0, 0.0, 1.0]))
            degen_gradient_count += 1
        block.clear_data()

    g3d_norm = np.array(g3d_norm)
    g3d_raw_norms = np.array(g3d_raw_norms)

    if degen_gradient_count > 0:
        warn(f"{degen_gradient_count}/{N} blocks had near-zero unconstrained gradients "
             f"({100*degen_gradient_count/N:.1f}%) — unit direction set to [0,0,1].")

    log(f"Unconstrained gradient energy stats: "
        f"median={np.median(g3d_raw_norms):.4f}, "
        f"p10={np.percentile(g3d_raw_norms, 10):.4f}, "
        f"p90={np.percentile(g3d_raw_norms, 90):.4f}, "
        f"max={g3d_raw_norms.max():.4f}")

    normal_weights = np.ones(N)
    for i in range(1, N):
        normal_weights[i] = np.abs(np.dot(normals[i], normals[i - 1]))

    log(f"Block preparation completed in {time.time() - t0:.2f}s.")
    return {
        "blocks": blocks,
        "N": N,
        "total_v": total_v,
        "tangent_data": tangent_data,
        "structural_coeffs_map": structural_coeffs_map,
        "g3d_norm": g3d_norm,
        "g3d_raw_norms": g3d_raw_norms,
        "normal_weights": normal_weights,
    }


def compute_structural_base_rate(pc, blocks, q_step, sample_ratio=0.01, n_strata=5):
    t0 = time.time()
    gft_computer = GFTStrategyWraper()
    sampler = Sampler(ratio=sample_ratio, n_strata=n_strata)
    sampled_blocks = sampler(pc.V, pc.A, blocks)
    decider = Decider(mode="0", lagrange_proportional=0.8)
    decider._set_vars(q_step)
    total_r_s = 0.0
    num_v = 0
    for block in sampled_blocks:
        block.init_data(pc.V, pc.A)
        num_v += block.Vblock.shape[0]
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs_s = gft_computer(block, s_graph)
        _, r_s, _ = decider._RDcost(coeffs_s)
        total_r_s += r_s
        block.clear_data()
    base_rate = total_r_s / num_v
    log(f"Baseline rate: {base_rate:.5f} bpv (computed in {time.time() - t0:.2f}s)")
    return base_rate


# ---------------------------------------------------------------------------
# Cost evaluation workers (joblib-compatible)
# ---------------------------------------------------------------------------

def _eval_block_cost_single(
    Vblock, Ablock, metadata,
    v1, v2, mu_k, k_cluster, slp_k, slw_k,
    unconstrained_norm, degen_thresh,
    decider_mode, lagrange, q_step
):
    dec = Decider(decider_mode, lagrange)
    dec._set_vars(q_step)
    gft = GFTStrategyWraper()
    app = Approximator()
    b = Block(metadata)
    b.Vblock = Vblock
    b.Ablock = Ablock

    s3d, _ = project_centroid_to_slope(mu_k, v1, v2, unconstrained_norm, degen_thresh)

    s_graph = StructuralGraph(metadata)
    s_graph.set_data(Vblock)
    a_graph = AttributeGraph(s_graph, s3d, k_cluster, slp_k, slw_k)
    V_rot = app._spatial_norm(Vblock)
    A_app = Ablock.copy()
    A_app[:, 0] = V_rot @ s3d.T
    a_graph.set_data(Vblock, A_app)

    try:
        _, coeffs = gft(b, a_graph)
        c, _, _ = dec._RDcost(coeffs)
        return c
    except Exception:
        return 1e18


def _eval_block_full_row(
    block_id, Vblock, Ablock, metadata, struct_coeffs,
    v1, v2, centroids_3d, slp, slw,
    unconstrained_norm, degen_thresh,
    decider_mode, lagrange, q_step, K
):
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
        costs[0] = 1e18

    degens = 0
    for k in range(1, K):
        mu_k = centroids_3d[k - 1]
        s3d, is_degen = project_centroid_to_slope(mu_k, v1, v2, unconstrained_norm, degen_thresh)
        if is_degen:
            degens += 1

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

    return costs, degens


# ---------------------------------------------------------------------------
# FIX 3: Boundary-aware grid search
# ---------------------------------------------------------------------------

def _boundary_check(best_p, best_w, p_grid, w_grid):
    p_hit = (best_p == min(p_grid)) or (best_p == max(p_grid))
    w_hit = (best_w == min(w_grid)) or (best_w == max(w_grid))
    return p_hit, w_hit


def _extend_grid(best_p, best_w, p_grid, w_grid):
    step_p = (p_grid[1] - p_grid[0]) if len(p_grid) > 1 else 0.1
    step_w = (w_grid[1] - w_grid[0]) if len(w_grid) > 1 else 0.3
    SLP_MIN = 0.05  # never 0 — see module docstring
    SLW_MIN = 0.1
    new_p = list(p_grid)
    new_w = list(w_grid)

    if best_p == min(p_grid):
        for step in range(1, 3):
            cand = round(best_p - step * step_p, 4)
            if cand >= SLP_MIN and cand not in new_p:
                new_p.append(cand)
    elif best_p == max(p_grid):
        for step in range(1, 3):
            cand = round(best_p + step * step_p, 4)
            if cand <= 1.0 and cand not in new_p:
                new_p.append(cand)

    if best_w == min(w_grid):
        for step in range(1, 3):
            cand = round(best_w - step * step_w, 4)
            if cand >= SLW_MIN and cand not in new_w:
                new_w.append(cand)
    elif best_w == max(w_grid):
        for step in range(1, 3):
            cand = round(best_w + step * step_w, 4)
            if cand <= 10.0 and cand not in new_w:
                new_w.append(cand)

    return sorted(new_p), sorted(new_w)


def grid_search_slp_slw(
    sample_data, mu_k, k_cluster,
    p_grid, w_grid,
    unconstrained_norms_sample,
    degen_thresh, decider_mode, lagrange, q_step,
    cluster_label, max_extensions=2
):
    """
    Grid search for (SLP, SLW) with boundary-aware adaptive extension.
    SLP minimum is 0.05 — SLP=0 (no self-loops) is reserved for Cluster 0.
    """
    best_cost = 1e18
    best_p = p_grid[len(p_grid) // 2]
    best_w = w_grid[len(w_grid) // 2]

    for extension in range(max_extensions + 1):
        for p in p_grid:
            for w in w_grid:
                tasks = [
                    (Vb, Ab, meta, v1, v2, mu_k, k_cluster, p, w,
                     unc_norm, degen_thresh, decider_mode, lagrange, q_step)
                    for (Vb, Ab, meta, v1, v2), unc_norm
                    in zip(sample_data, unconstrained_norms_sample)
                ]
                costs = Parallel(n_jobs=-1)(
                    delayed(_eval_block_cost_single)(*t) for t in tasks
                )
                c_sum = sum(costs)
                if c_sum < best_cost:
                    best_cost = c_sum
                    best_p, best_w = p, w

        p_hit, w_hit = _boundary_check(best_p, best_w, p_grid, w_grid)
        if (p_hit or w_hit) and extension < max_extensions:
            info = []
            if p_hit:
                info.append(f"SLP={best_p:.3f} boundary=[{min(p_grid):.3f},{max(p_grid):.3f}]")
            if w_hit:
                info.append(f"SLW={best_w:.3f} boundary=[{min(w_grid):.3f},{max(w_grid):.3f}]")
            warn(f"  Cluster {cluster_label} winner at boundary [{', '.join(info)}] "
                 f"— extending grid (pass {extension+1}/{max_extensions}).")
            p_grid, w_grid = _extend_grid(best_p, best_w, p_grid, w_grid)
        else:
            if p_hit or w_hit:
                info = []
                if p_hit:
                    info.append(f"SLP={best_p:.3f}")
                if w_hit:
                    info.append(f"SLW={best_w:.3f}")
                warn(f"  Cluster {cluster_label} still at boundary [{', '.join(info)}] "
                     f"after {max_extensions} extensions. Accepting.")
            break

    return best_p, best_w


# ---------------------------------------------------------------------------
# Main codebook training + scoring
# ---------------------------------------------------------------------------

def train_and_score_codebook(ctx, K, decider_mode, lagrange, q_step,
                              rd_iters=5, degen_thresh=1e-3):
    blocks = ctx["blocks"]
    N = ctx["N"]
    tangent_data = ctx["tangent_data"]
    structural_coeffs_map = ctx["structural_coeffs_map"]
    g3d_norm = ctx["g3d_norm"]
    g3d_raw_norms = ctx["g3d_raw_norms"]
    K_hybrid = K - 1

    # -- PHASE 1: Spherical EM -----------------------------------------------
    log(f"--- PHASE 1: Spherical EM (K={K}, K_hybrid={K_hybrid}) ---")
    t0 = time.time()
    centroids_3d = spherical_kmeanspp(g3d_norm, K_hybrid)
    labels_em = em_assign(g3d_norm, centroids_3d)

    for it in range(1, 21):
        prev_labels = labels_em.copy()
        centroids_3d = em_update_centroids(g3d_norm, labels_em, K_hybrid, centroids_3d)
        labels_em = em_assign(g3d_norm, centroids_3d)
        changed = np.sum(labels_em != prev_labels)
        if changed == 0:
            log(f"  EM converged at iteration {it}.")
            break

    log(f"Phase 1 EM completed in {time.time() - t0:.2f}s.")
    for k in range(K_hybrid):
        c = centroids_3d[k]
        n = np.sum(labels_em == k)
        log(f"  Centroid {k+1}: [{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}] "
            f"norm={np.linalg.norm(c):.4f} | EM blocks: {n} ({100*n/N:.1f}%)")

    # FIX 1 diagnostic: per-cluster raw projection magnitude analysis
    log("--- FIX 1 Diagnostic: Per-Cluster Projected Slope Magnitude Analysis ---")
    for k in range(K_hybrid):
        mu_k = centroids_3d[k]
        cluster_idx = np.where(labels_em == k)[0]
        sample_size = min(500, len(cluster_idx))
        if sample_size == 0:
            continue
        proj_mags = []
        for i in cluster_idx[:sample_size]:
            v1, v2 = tangent_data[i]["v1"], tangent_data[i]["v2"]
            s3d_raw = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
            proj_mags.append(np.linalg.norm(s3d_raw))
        proj_mags = np.array(proj_mags)
        n_degen = int(np.sum(proj_mags < degen_thresh))
        log(f"  Centroid {k+1} (sample={sample_size}): "
            f"proj_mag median={np.median(proj_mags):.4f}, "
            f"p10={np.percentile(proj_mags,10):.4f}, "
            f"p90={np.percentile(proj_mags,90):.4f} | "
            f"near-degen(<{degen_thresh}): {n_degen} ({100*n_degen/sample_size:.1f}%)")

    # -- PHASE 1.5: R-D Refinement with Live Label Updates (FIX 2) -----------
    log(f"--- PHASE 1.5: R-D Refinement ({rd_iters} iters) + Live Label Update ---")
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.45)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 1.5)])

    strat_idx = get_stratified_indices(
        np.arange(N), max_samples=min(1000, max(300, int(0.30 * N)))
    )
    log(f"Selected {len(strat_idx)} stratified sample blocks for R-D refinement.")

    # FIX 2: initialise sample labels from EM, then update each iteration
    sample_labels = labels_em[strat_idx].copy()

    for rd_iter in range(1, rd_iters + 1):
        t_iter = time.time()
        rd_tasks = []
        for idx in strat_idx:
            block = blocks[idx]
            block.init_data(ctx["pc_V"], ctx["pc_A"])
            rd_tasks.append((
                block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
                structural_coeffs_map[block.block_id],
                tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                centroids_3d, slp, slw,
                g3d_raw_norms[idx], degen_thresh,
                decider_mode, lagrange, q_step, K
            ))
            block.clear_data()

        results = Parallel(n_jobs=-1)(delayed(_eval_block_full_row)(*t) for t in rd_tasks)
        sample_costs = np.array([r[0] for r in results])

        # FIX 2: update labels from R-D cost (not EM cosine)
        new_sample_labels = np.argmin(sample_costs, axis=1)
        label_changes = int(np.sum(new_sample_labels != sample_labels))
        sample_labels = new_sample_labels

        log(f"  Iter {rd_iter}/{rd_iters}: "
            f"R-D label changes={label_changes}/{len(strat_idx)} "
            f"({100*label_changes/len(strat_idx):.1f}%) | "
            f"{time.time()-t_iter:.2f}s")

        # Update centroids from R-D-assigned sample labels
        for k_idx in range(K_hybrid):
            k = k_idx + 1
            assigned = strat_idx[sample_labels == k]
            if len(assigned) >= 3:
                sum_g = np.zeros(3)
                mu_k = centroids_3d[k_idx]
                for idx in assigned:
                    g = g3d_norm[idx]
                    sign = np.sign(np.dot(g, mu_k))
                    if sign == 0:
                        sign = 1.0
                    sum_g += sign * g
                norm_sum = np.linalg.norm(sum_g)
                if norm_sum > 1e-8:
                    centroids_3d[k_idx] = sum_g / norm_sum
            else:
                log(f"    Centroid {k_idx+1}: only {len(assigned)} R-D blocks "
                    f"in sample — frozen this iter.")

    log("--- Post-Refinement Centroids (R-D) ---")
    for k in range(K_hybrid):
        c = centroids_3d[k]
        n_sample = int(np.sum(sample_labels == (k + 1)))
        log(f"  Centroid {k+1}: [{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}] "
            f"norm={np.linalg.norm(c):.4f} | "
            f"R-D sample blocks: {n_sample}/{len(strat_idx)}")

    # -- Grid search (SLP, SLW): FIX 3 (boundary-aware), FIX 2 (R-D labels) --
    # SLP minimum = 0.05; SLP=0 is FORBIDDEN (see module docstring).
    p_grid_base = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    w_grid_base = [0.5, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]

    log(f"Grid search (SLP in [{min(p_grid_base)},{max(p_grid_base)}], "
        f"SLW in [{min(w_grid_base)},{max(w_grid_base)}]) "
        f"using R-D-assigned sample blocks per cluster...")

    for k in range(1, K):
        c_idx = k - 1
        # FIX 2: use R-D-consistent labels (sample_labels), not stale labels_em
        rd_assigned = strat_idx[sample_labels == k]
        c_dir = centroids_3d[c_idx]
        c_mag = np.linalg.norm(c_dir)

        if len(rd_assigned) < 5:
            warn(f"  Cluster {k}: only {len(rd_assigned)} R-D-assigned sample blocks "
                 f"— skipping grid search, keeping defaults slp={slp[k]:.2f}, slw={slw[k]:.2f}.")
            continue

        sample_idx = get_stratified_indices(rd_assigned, max_samples=200)
        sample_data, unc_norms = [], []
        for idx in sample_idx:
            block = blocks[idx]
            block.init_data(ctx["pc_V"], ctx["pc_A"])
            sample_data.append((
                block.Vblock.copy(), block.Ablock.copy(), block.metadata,
                tangent_data[idx]["v1"], tangent_data[idx]["v2"]
            ))
            unc_norms.append(g3d_raw_norms[idx])
            block.clear_data()

        log(f"  Cluster {k}: grid search on {len(sample_data)} R-D blocks "
            f"(centroid [{c_dir[0]:.3f},{c_dir[1]:.3f},{c_dir[2]:.3f}])...")
        best_p, best_w = grid_search_slp_slw(
            sample_data, centroids_3d[c_idx], k,
            list(p_grid_base), list(w_grid_base),
            unc_norms, degen_thresh, decider_mode, lagrange, q_step,
            cluster_label=k, max_extensions=2
        )
        slp[k], slw[k] = best_p, best_w
        log(f"  Cluster {k} result: slp={best_p:.3f}, slw={best_w:.3f} "
            f"(mag={c_mag:.4f}, {len(sample_data)} blocks)")

    # -- PHASE 2: Full N x K cost matrix -------------------------------------
    log(f"--- PHASE 2: Evaluating {N} x {K} Cost Matrix ---")
    t0 = time.time()
    eval_tasks = []
    for i, block in enumerate(blocks):
        block.init_data(ctx["pc_V"], ctx["pc_A"])
        eval_tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            structural_coeffs_map[block.block_id],
            tangent_data[i]["v1"], tangent_data[i]["v2"],
            centroids_3d, slp, slw,
            g3d_raw_norms[i], degen_thresh,
            decider_mode, lagrange, q_step, K
        ))
        block.clear_data()

    results = Parallel(n_jobs=-1)(delayed(_eval_block_full_row)(*t) for t in eval_tasks)
    cost_matrix = np.array([r[0] for r in results])
    total_degens = sum(r[1] for r in results)
    log(f"Phase 2 completed in {time.time() - t0:.2f}s.")

    if total_degens > 0:
        pct = 100 * total_degens / (N * (K - 1))
        warn(f"Phase 2: {total_degens}/{N*(K-1)} block-cluster evaluations were "
             f"near-degenerate ({pct:.2f}%) — tangent perturbation applied.")

    min_assignments = np.argmin(cost_matrix, axis=1)
    cluster_counts = np.bincount(min_assignments, minlength=K)
    log(f"\n--- Phase 2 Block Allocation (Total: {N}) ---")
    log(f"  Cluster 0 (Flat/Structural): "
        f"{cluster_counts[0]} ({100*cluster_counts[0]/N:.2f}%)")
    for k in range(1, K):
        c_dir = centroids_3d[k - 1]
        log(f"  Cluster {k} "
            f"[{c_dir[0]:.3f},{c_dir[1]:.3f},{c_dir[2]:.3f}] "
            f"slp={slp[k]:.3f} slw={slw[k]:.3f}: "
            f"{cluster_counts[k]} ({100*cluster_counts[k]/N:.2f}%)")

    dead = [k for k in range(1, K) if cluster_counts[k] == 0]
    if dead:
        warn(f"Dead clusters (0 blocks in Phase 2): {dead}. "
             f"Consider reducing K or reviewing centroid quality.")

    return cost_matrix


# ---------------------------------------------------------------------------
# Viterbi (Phase 3)
# ---------------------------------------------------------------------------

def run_viterbi(ctx, cost_matrix, K, gamma0):
    N = ctx["N"]
    nw = ctx["normal_weights"]
    dp = np.zeros((N, K))
    paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i - 1] + gamma0 * w * (np.arange(K) != k)
            bp = np.argmin(tc)
            dp[i, k] = cost_matrix[i, k] + tc[bp]
            paths[i, k] = bp
    labels = np.zeros(N, dtype=int)
    labels[N - 1] = np.argmin(dp[N - 1])
    for i in range(N - 2, -1, -1):
        labels[i] = paths[i + 1, labels[i + 1]]
    return labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    block_sizes = parse_int_list(args.block_sizes)
    k_values = parse_int_list(args.k_values)
    gammas = parse_float_list(args.gammas)
    q_step = args.q_step
    degen_thresh = args.degen_thresh

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / args.output_name

    log(f"Global 3D Projection v2 — output: {report_file}")
    log("Active fixes: [1] Magnitude Normalisation | [2] Live Label Update | "
        "[3] Boundary Grid Search | [4] Degen Guard")
    log(f"B={block_sizes}, K={k_values}, gamma={gammas}, Q={q_step}, "
        f"rd_iters={args.rd_iters}, degen_thresh={degen_thresh}")

    with open(report_file, "w") as f:
        f.write("# Global 3D Projection v2 — Results\n\n")
        f.write("| B | K | Method | Baseline γ₀ | GFT Savings % | H(X|X-1) | "
                "Overhead BPV | Net Markov BPV | Switches (Spatial) |\n")
        f.write("|---:|---:|:---|---:|---:|---:|---:|---:|---:|\n")
        f.flush()

    params = load_experiment_config(Path(args.config))
    colourist = Colourist()
    pc = PointCloud.from_file(
        project_root / params.sequential_params.point_cloud_path, "ply", params.pointcloud
    )
    pc.transform_attributes(colourist._RGBtoYUV)
    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional

    for bsize in block_sizes:
        log(f"\n{'='*60}")
        log(f"STARTING BLOCK SIZE B = {bsize}")
        log(f"{'='*60}")
        ctx = prepare_blocks(pc, bsize, degen_thresh=degen_thresh)
        ctx["pc_V"] = pc.V
        ctx["pc_A"] = pc.A
        total_v = ctx["total_v"]
        N = ctx["N"]

        base_rate = compute_structural_base_rate(
            pc, ctx["blocks"], q_step,
            sample_ratio=args.sample_ratio, n_strata=args.n_strata,
        )

        for K in k_values:
            log(f"\n>>>> Codebook K={K}, Block Size B={bsize} <<<<")
            cost_matrix = train_and_score_codebook(
                ctx, K, decider_mode, lagrange, q_step,
                rd_iters=args.rd_iters, degen_thresh=degen_thresh
            )
            structural_cost = cost_matrix[:, 0].sum()

            log(f"--- PHASE 3: Viterbi Sweep ({len(gammas)} gamma values) ---")
            for gamma0 in gammas:
                labels = run_viterbi(ctx, cost_matrix, K, gamma0)

                gft_cost = sum(cost_matrix[i, labels[i]] for i in range(N))
                savings_pct = (structural_cost - gft_cost) / structural_cost * 100
                entropy = compute_entropy(labels, K)
                overhead = entropy * N / total_v
                net_bpv = base_rate * (savings_pct / 100.0) - overhead
                switches = int(np.sum(labels[1:] != labels[:-1]))

                v_counts = np.bincount(labels, minlength=K)
                flat_pct = v_counts[0] / N * 100.0
                dir_pct = (N - v_counts[0]) / N * 100.0

                row = (
                    f"| {bsize} | {K} | Global 3D v2 | {gamma0:.1f} | "
                    f"{savings_pct:.3f}% | {entropy:.4f} | {overhead:.5f} | "
                    f"{net_bpv:+.5f} | {switches} |"
                )
                log(f"RESULT (γ={gamma0:.0f} | Flat:{v_counts[0]}[{flat_pct:.1f}%] "
                    f"Dir:{N-v_counts[0]}[{dir_pct:.1f}%]): {row}")

                with open(report_file, "a") as f:
                    f.write(row + "\n")
                    f.flush()
                    os.fsync(f.fileno())

    log(f"\nSweep completed. Results: {report_file}")


if __name__ == "__main__":
    main()
