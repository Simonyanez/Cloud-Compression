"""
Advanced Oracle-Initialized EM Codebook Training
================================================
Includes:
  1. Closed-form least-squares slope update (analytical, exact)
  2. 2D coarse→fine grid search over (slp, slw) with slope fixed
  3. Stable Potts model flat transition penalty (gamma)
  4. Linear Gamma Ramp: Ramps gamma up slowly from 0 to gamma_target
     to avoid shocking block assignments and let clusters contract smoothly.
  5. Centroid Freezing: Option to freeze slopes after warmup iterations,
     preventing the spatial transition penalty from degrading GFT representation.

Usage:
  python3 scripts/oracle_em_advanced.py \\
    --block_size 16 --clusters 6 --q_step 24 \\
    --sample_rate 0.20 --gamma 1500.0 --gamma_warmup 3 \\
    --freeze_slopes --max_iters 6 --min_iters 4 \\
    --out_dir experiments/oracle_em_frozen_b16
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

# Path setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block


# ===========================================================================
# STEP 1 — Initialization
# ===========================================================================

def build_initial_codebook_from_pc(blocks: List, pc_V: np.ndarray, pc_A: np.ndarray,
                                   n_hybrid_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build initial codebook directly from target point cloud PCA-rotated slopes.
    Uses block fit RMSE as weight for K-Means.
    """
    print(f"\n[Init] Fitting luminance slopes on PCA-rotated coordinates...")
    approximator = Approximator()
    slopes_list = []
    rmse_list = []
    for block in tqdm(blocks, desc="Fitting luminance"):
        block.init_data(pc_V, pc_A)
        fit_result = approximator(block)
        # fit_result.coeffs contains [intercept, slope_x, slope_y, slope_z]
        slopes_list.append(fit_result.coeffs[1:])
        rmse_list.append(max(0.1, fit_result.rmse))
        block.clear_data()

    slopes_arr = np.array(slopes_list)  # (N, 3)
    rmse_arr = np.array(rmse_list)      # (N,)

    # Weighted K-Means to cluster the slopes
    print(f"[Init] Clustering {len(slopes_arr)} PCA-rotated slopes into {n_hybrid_clusters} modes (weighted by RMSE)...")
    # Normalize features to [0,1] per dimension for equal weighting
    feat_min = slopes_arr.min(axis=0)
    feat_max = slopes_arr.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 1e-8, feat_max - feat_min, 1.0)
    X = (slopes_arr - feat_min) / feat_range

    # Weighted K-Means
    w = rmse_arr / rmse_arr.sum()
    rng = np.random.default_rng(42)
    centers = X[rng.choice(len(X), size=n_hybrid_clusters, replace=False, p=w)].copy()

    for _ in range(50):
        dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(n_hybrid_clusters):
            mask = (labels == j)
            if mask.sum() == 0:
                furthest = np.argmax(np.min(dists[:, [m for m in range(n_hybrid_clusters) if m != j]], axis=1))
                new_centers[j] = X[furthest]
            else:
                new_centers[j] = np.average(X[mask], axis=0, weights=w[mask])
        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers

    hybrid_slopes = centers * feat_range + feat_min

    # Build full codebook (cluster 0 = structural)
    slopes = np.vstack([np.zeros((1, 3)), hybrid_slopes])  # (K+1, 3)
    # Initialize self-loops to the oracle averages (p=0.65, w=1.4)
    slp    = np.concatenate([[0.0], np.full(n_hybrid_clusters, 0.65)])
    slw    = np.concatenate([[1.0], np.full(n_hybrid_clusters, 1.4)])

    print("\n[Init] Initial codebook (correct PCA-rotated coordinate system):")
    print(f"  {'Cluster':>8}  {'slope_x':>8} {'slope_y':>8} {'slope_z':>8}  {'slp':>6} {'slw':>6}")
    for k in range(len(slopes)):
        tag = "(structural)" if k == 0 else ""
        print(f"  {k:>8}  {slopes[k,0]:>8.3f} {slopes[k,1]:>8.3f} {slopes[k,2]:>8.3f}  "
              f"{slp[k]:>6.3f} {slw[k]:>6.3f}  {tag}")

    return slopes, slp, slw


# ===========================================================================
# STEP 2 — E-step: Parallel RD evaluation + Potts transition penalty
# ===========================================================================

def _evaluate_block_costs_worker(block_id, Vblock, Ablock, metadata,
                                  slopes, slp_vals, slw_vals,
                                  structural_coeffs, q_step,
                                  lagrange_proportional, decider_mode):
    """
    Worker: evaluate RD cost of one block against ALL K clusters.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from pcadc.decider import Decider
    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator
    from pcadc.blocks import Block

    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    approximator = Approximator()

    block = Block(metadata)
    block.Vblock = Vblock
    block.Ablock = Ablock

    K = len(slopes)
    costs = np.full(K, np.inf)

    for k, slope in enumerate(slopes):
        if k == 0:
            c, _, _ = decider._RDcost(structural_coeffs)
            costs[k] = c
        else:
            slw = slw_vals[k]
            slp = slp_vals[k]
            s_graph = StructuralGraph(metadata)
            s_graph.set_data(Vblock)
            a_graph = AttributeGraph(s_graph, slope, k, slp, slw)

            V_rot = approximator._spatial_norm(Vblock)
            A_app = Ablock.copy()
            A_app[:, 0] = V_rot @ slope.T
            a_graph.set_data(Vblock, A_app)

            _, coeffs = gft_computer(block, a_graph)
            c, _, _ = decider._RDcost(coeffs)
            costs[k] = c

    return costs


def e_step(blocks: List, structural_coeffs_map: dict,
           slopes: np.ndarray, slp: np.ndarray, slw: np.ndarray,
           q_step: int, lagrange_proportional: float, decider_mode: str,
           vertices: np.ndarray, attributes: np.ndarray,
           gamma: float, prev_labels: Optional[np.ndarray],
           iteration: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    E-step: evaluate all K clusters per block in parallel, then apply
    transition penalty sequentially using the provided gamma.
    """
    tasks = []
    for block in blocks:
        block.init_data(vertices, attributes)
        tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            slopes, slp, slw,
            structural_coeffs_map[block.block_id],
            q_step, lagrange_proportional, decider_mode
        ))
        block.clear_data()

    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_block_costs_worker)(*t)
        for t in tqdm(tasks, desc=f"  E-step (iter {iteration}, γ={gamma:.1f})")
    )

    cost_matrix = np.array(results)
    N, K = cost_matrix.shape
    new_labels = np.zeros(N, dtype=int)

    if gamma == 0.0 or prev_labels is None:
        new_labels = np.argmin(cost_matrix, axis=1)
    else:
        new_labels[0] = np.argmin(cost_matrix[0])
        for i in range(1, N):
            adj_costs = cost_matrix[i].copy()
            mask = np.ones(K, dtype=bool)
            mask[new_labels[i-1]] = False
            adj_costs[mask] += gamma
            new_labels[i] = np.argmin(adj_costs)

    return new_labels, cost_matrix


# ===========================================================================
# STEP 3 — M-step: Parallel grid search & closed-form slope
# ===========================================================================

def _compute_slope_closed_form(Vblocks: List[np.ndarray],
                                Ablocks: List[np.ndarray]) -> np.ndarray:
    approximator = Approximator()
    V_stack = []
    Y_stack = []
    for V, A in zip(Vblocks, Ablocks):
        if V.shape[0] < 2:
            continue
        V_norm = approximator._spatial_norm(V)
        V_stack.append(V_norm)
        Y_stack.append(A[:, 0])

    if not V_stack:
        return np.zeros(3)

    V_all = np.vstack(V_stack)
    Y_all = np.concatenate(Y_stack)
    slope, _, _, _ = np.linalg.lstsq(V_all, Y_all, rcond=None)
    return slope


def _eval_grid_point(Vblocks_list, Ablocks_list, metadata_list,
                     slope, slp_val, slw_val,
                     q_step, lagrange_proportional, decider_mode):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from pcadc.decider import Decider
    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator
    from pcadc.blocks import Block

    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()
    approximator = Approximator()

    total_cost = 0.0
    for Vblock, Ablock, metadata in zip(Vblocks_list, Ablocks_list, metadata_list):
        block = Block(metadata)
        block.Vblock = Vblock
        block.Ablock = Ablock

        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)

        V_rot = approximator._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ slope.T

        a_graph = AttributeGraph(s_graph, slope, 1, slp_val, slw_val)
        a_graph.set_data(Vblock, A_app)
        _, coeffs = gft_computer(block, a_graph)
        c, _, _ = decider._RDcost(coeffs)
        total_cost += c

    return total_cost


def _evaluate_grid_parallel(Vblocks_list, Ablocks_list, metadata_list,
                             slope, slp_vals, slw_vals,
                             q_step, lagrange_proportional, decider_mode) -> np.ndarray:
    grid_points = [(pi, wi, p, w)
                   for pi, p in enumerate(slp_vals)
                   for wi, w in enumerate(slw_vals)]

    results = Parallel(n_jobs=-1)(
        delayed(_eval_grid_point)(
            Vblocks_list, Ablocks_list, metadata_list,
            slope, p, w, q_step, lagrange_proportional, decider_mode
        )
        for _, _, p, w in grid_points
    )

    total_costs = np.zeros((len(slp_vals), len(slw_vals)))
    for (pi, wi, _, _), cost in zip(grid_points, results):
        total_costs[pi, wi] = cost
    return total_costs


def m_step(blocks: List, labels: np.ndarray,
           slopes: np.ndarray, slp: np.ndarray, slw: np.ndarray,
           vertices: np.ndarray, attributes: np.ndarray,
           q_step: int, lagrange_proportional: float, decider_mode: str,
           min_cluster_size: int = 10, sample_rate: float = 1.0,
           freeze_slopes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step: update parameters for each cluster. Skip slope updates if freeze_slopes is True.
    """
    K = len(slopes)
    new_slopes = slopes.copy()
    new_slp = slp.copy()
    new_slw = slw.copy()

    slp_coarse = np.linspace(0.02, 0.98, 10)
    slw_coarse = np.linspace(0.2, 6.0, 10)

    for k in range(1, K):
        mask = (labels == k)
        n_assigned = mask.sum()

        if n_assigned < min_cluster_size:
            print(f"  [M-step] Cluster {k}: only {n_assigned} blocks — keeping previous params (revival guard)")
            continue

        print(f"  [M-step] Cluster {k}: {n_assigned} blocks (sampling={sample_rate:.2f})")

        Vblocks, Ablocks, metadatas = [], [], []
        assigned_indices = np.where(mask)[0]
        if sample_rate < 1.0:
            n_sample = max(min_cluster_size, int(round(len(assigned_indices) * sample_rate)))
            if n_sample < len(assigned_indices):
                assigned_indices = np.random.choice(assigned_indices, size=n_sample, replace=False)
                print(f"         Sub-sampled to {n_sample} blocks for training")

        for idx in assigned_indices:
            block = blocks[idx]
            block.init_data(vertices, attributes)
            Vblocks.append(block.Vblock.copy())
            Ablocks.append(block.Ablock.copy())
            metadatas.append(block.metadata)
            block.clear_data()

        # (a) Slope Update
        if freeze_slopes:
            new_slope = slopes[k].copy()
            print(f"         slope: [{new_slope[0]:+.3f}, {new_slope[1]:+.3f}, {new_slope[2]:+.3f}] (FROZEN)")
        else:
            new_slope = _compute_slope_closed_form(Vblocks, Ablocks)
            new_slopes[k] = new_slope
            print(f"         slope: [{new_slope[0]:+.3f}, {new_slope[1]:+.3f}, {new_slope[2]:+.3f}]")

        # (b) Coarse search (slp, slw) with new_slope fixed
        print(f"         Grid search: {len(slp_coarse)}x{len(slw_coarse)} coarse...")
        coarse_costs = _evaluate_grid_parallel(
            Vblocks, Ablocks, metadatas,
            new_slope, slp_coarse, slw_coarse,
            q_step, lagrange_proportional, decider_mode
        )
        best_pi, best_wi = np.unravel_index(np.argmin(coarse_costs), coarse_costs.shape)
        best_p_c = slp_coarse[best_pi]
        best_w_c = slw_coarse[best_wi]

        # (c) Fine refinement around coarse winner
        slp_fine = np.linspace(max(0.01, best_p_c - 0.12), min(0.99, best_p_c + 0.12), 5)
        slw_fine = np.linspace(max(0.1,  best_w_c - 0.8),  max(0.5, best_w_c + 0.8),  5)
        print(f"         Grid search: 5x5 fine around slp={best_p_c:.3f} slw={best_w_c:.3f}...")
        fine_costs = _evaluate_grid_parallel(
            Vblocks, Ablocks, metadatas,
            new_slope, slp_fine, slw_fine,
            q_step, lagrange_proportional, decider_mode
        )
        best_pf, best_wf = np.unravel_index(np.argmin(fine_costs), fine_costs.shape)

        new_slp[k] = slp_fine[best_pf]
        new_slw[k] = slw_fine[best_wf]
        print(f"         slp={new_slp[k]:.3f}  slw={new_slw[k]:.3f}")

    return new_slopes, new_slp, new_slw


# ===========================================================================
# STEP 4 — Convergence check + Metrics
# ===========================================================================

def compute_metrics(blocks: List, labels: np.ndarray, cost_matrix: np.ndarray,
                    vertices: np.ndarray, q_step: int) -> dict:
    K = cost_matrix.shape[1]
    N = len(labels)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)

    total_cost = sum(cost_matrix[i, labels[i]] for i in range(N))
    cluster_sizes = np.bincount(labels, minlength=K)
    active = int((cluster_sizes > 0).sum())

    probs = cluster_sizes / cluster_sizes.sum()
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

    trans = np.ones((K, K))
    for i in range(1, N):
        trans[labels[i-1], labels[i]] += 1
    trans_probs = trans / trans.sum(axis=1, keepdims=True)
    h_cond = 0.0
    for prev in range(K):
        p_prev = cluster_sizes[prev] / N
        for cur in range(K):
            p = trans_probs[prev, cur]
            if p > 0 and p_prev > 0:
                h_cond -= p_prev * p * np.log2(p)

    overhead_marginal  = entropy  * N / total_v
    overhead_markov    = h_cond   * N / total_v

    structural_cost = sum(cost_matrix[i, 0] for i in range(N))
    rate_reduction_pct = (structural_cost - total_cost) / structural_cost * 100

    return {
        "total_cost":          total_cost,
        "structural_cost":     structural_cost,
        "rate_reduction_pct":  rate_reduction_pct,
        "entropy_marginal":    entropy,
        "entropy_markov":      h_cond,
        "overhead_bpv_marginal": overhead_marginal,
        "overhead_bpv_markov":   overhead_markov,
        "active_clusters":     active,
        "cluster_sizes":       cluster_sizes.tolist(),
    }


def converged(history: list, window: int = 3,
              cost_tol: float = 1e-3, hamming_tol: float = 0.005) -> bool:
    if len(history) < window:
        return False

    costs = [h["total_cost"] for h in history[-window:]]
    rel_diff = (max(costs) - min(costs)) / (abs(min(costs)) + 1e-10)
    if rel_diff < cost_tol:
        print(f"  [Convergence] Cost stable (rel_diff={rel_diff:.2e})")
        return True

    labels_hist = [h["labels"] for h in history[-window:]]
    if len(labels_hist) >= 2:
        N = len(labels_hist[0])
        diffs = [np.mean(labels_hist[i] != labels_hist[i+1])
                 for i in range(len(labels_hist)-1)]
        avg_hamming = np.mean(diffs)
        if avg_hamming < hamming_tol:
            print(f"  [Convergence] Assignments stable (hamming={avg_hamming:.4f})")
            return True

    return False


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = ArgumentParser(description="Advanced Oracle EM codebook training")
    parser.add_argument("--block_size",  type=int,   required=True)
    parser.add_argument("--clusters",    type=int,   required=True,
                        help="Total number of clusters including structural cluster 0")
    parser.add_argument("--q_step",      type=int,   required=True)
    parser.add_argument("--gamma",       type=float, default=0.0,
                        help="Target transition penalty weight (0=disabled)")
    parser.add_argument("--max_iters",   type=int,   default=10)
    parser.add_argument("--min_iters",   type=int,   default=3)
    parser.add_argument("--gamma_warmup",type=int,   default=2,
                        help="Iterations before gamma penalty starts ramping up")
    parser.add_argument("--freeze_slopes",action="store_true",
                        help="Freeze slopes after gamma_warmup iterations to prevent degradation")
    parser.add_argument("--out_dir",     type=str,   required=True)
    parser.add_argument("--config",      type=str,   default="config/base_config.yaml")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="M-step training blocks sub-sampling rate (0.0 to 1.0)")
    args = parser.parse_args()

    n_hybrid = args.clusters - 1
    if n_hybrid < 1:
        raise ValueError("--clusters must be >= 2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_experiment_config(Path(args.config))
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)

    print(f"\n[Setup] Point cloud: {pc.V.shape[0]:,} points")
    print(f"[Setup] Block size: {args.block_size}  Clusters: {args.clusters} (1 structural + {n_hybrid} adaptive)")
    print(f"[Setup] Q-step: {args.q_step}  Target Gamma: {args.gamma}  Max iters: {args.max_iters}")
    print(f"[Setup] Freeze slopes: {args.freeze_slopes}  Warmup iters: {args.gamma_warmup}")

    _, blocks = MortonBlockPartition().partition(pc, bsize=args.block_size)
    total_v = pc.V.shape[0]
    print(f"[Setup] Total blocks: {len(blocks):,}")

    print("\n[Precompute] Structural GFT coefficients for all blocks...")
    gft_computer = GFTStrategyWraper()
    structural_coeffs_map = {}
    for block in tqdm(blocks, desc="Structural GFT"):
        block.init_data(pc.V, pc.A)
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs = gft_computer(block, s_graph)
        structural_coeffs_map[block.block_id] = coeffs
        block.clear_data()

    # Step 1: PCA-rotated K-Means initialization
    slopes, slp, slw = build_initial_codebook_from_pc(blocks, pc.V, pc.A, n_hybrid)

    history = []
    prev_labels = None
    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional

    min_cluster_size = max(5, int(0.01 * len(blocks)))

    print(f"\n{'='*60}")
    print(f"EM LOOP  (max_iters={args.max_iters}  min_iters={args.min_iters})")
    print(f"{'='*60}")

    for iteration in range(1, args.max_iters + 1):
        # Calculate effective gamma with linear ramp
        if iteration <= args.gamma_warmup:
            effective_gamma = 0.0
        else:
            steps = args.max_iters - args.gamma_warmup
            current_step = iteration - args.gamma_warmup
            effective_gamma = args.gamma * (current_step / steps)

        print(f"\n── Iteration {iteration}/{args.max_iters} (γ={effective_gamma:.1f}) ──────────────────────────────")

        # ── E-step ──────────────────────────────────────────────────────
        labels, cost_matrix = e_step(
            blocks, structural_coeffs_map,
            slopes, slp, slw,
            args.q_step, lagrange, decider_mode,
            pc.V, pc.A,
            gamma=effective_gamma,
            prev_labels=prev_labels,
            iteration=iteration
        )

        # ── Cluster revival check ────────────────────────────────────────
        cluster_sizes = np.bincount(labels, minlength=len(slopes))
        for k in range(1, len(slopes)):
            if cluster_sizes[k] < min_cluster_size:
                print(f"  [Revival] Cluster {k} has {cluster_sizes[k]} blocks — triggering revival")
                gain_potential = cost_matrix[:, 0] - np.min(cost_matrix[:, 1:], axis=1)
                candidates = np.where(labels != k)[0]
                if len(candidates) > 0:
                    best_cand = candidates[np.argmax(gain_potential[candidates])]
                    labels[best_cand] = k
                    print(f"         Forced block {best_cand} into cluster {k}")

        # ── Metrics ──────────────────────────────────────────────────────
        metrics = compute_metrics(blocks, labels, cost_matrix, pc.V, args.q_step)
        metrics["labels"] = labels.copy()
        metrics["iteration"] = iteration
        history.append(metrics)

        print(f"  Cost:           {metrics['total_cost']:>14.1f}  (structural: {metrics['structural_cost']:.1f})")
        print(f"  RD savings:     {metrics['rate_reduction_pct']:>10.2f}%  vs structural baseline")
        print(f"  H(X) marginal:  {metrics['entropy_marginal']:>10.4f} bits  → overhead {metrics['overhead_bpv_marginal']:.4f} bpv")
        print(f"  H(X|X-1):       {metrics['entropy_markov']:>10.4f} bits  → overhead {metrics['overhead_bpv_markov']:.4f} bpv")
        print(f"  Active clusters:{metrics['active_clusters']:>10d} / {len(slopes)}")
        print(f"  Cluster sizes:  {metrics['cluster_sizes']}")

        # ── M-step ──────────────────────────────────────────────────────
        if iteration < args.max_iters:
            freeze = args.freeze_slopes and (iteration >= args.gamma_warmup)
            slopes, slp, slw = m_step(
                blocks, labels, slopes, slp, slw,
                pc.V, pc.A,
                args.q_step, lagrange, decider_mode,
                min_cluster_size=min_cluster_size,
                sample_rate=args.sample_rate,
                freeze_slopes=freeze
            )

        prev_labels = labels.copy()

        # ── Convergence check ────────────────────────────────────────────
        if iteration >= args.min_iters and converged(history):
            print(f"\n[Converged] Stopping at iteration {iteration}")
            break

    # Save final results
    final = history[-1]
    save_data = {
        "config": {
            "block_size": args.block_size,
            "clusters": args.clusters,
            "q_step": args.q_step,
            "gamma": args.gamma,
            "max_iters": args.max_iters,
            "freeze_slopes": args.freeze_slopes,
            "gamma_warmup": args.gamma_warmup,
        },
        "final_slopes":  slopes.tolist(),
        "final_slp":     slp.tolist(),
        "final_slw":     slw.tolist(),
        "iterations": [
            {k: v for k, v in h.items() if k != "labels"}
            for h in history
        ],
        "final_labels": final["labels"].tolist(),
    }
    out_file = out_dir / f"oracle_em_B{args.block_size}_C{args.clusters}_Q{args.q_step}_G{args.gamma:.0f}.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
