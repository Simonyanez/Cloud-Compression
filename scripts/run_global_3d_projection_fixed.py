"""
Global 3D Codebook Projection Sweep (Real Run - Live Output & High-Rigor)
======================================================================
Phase 1: Unconstrained Global 3D Spherical EM Training.
Phase 1.5: Deep Stratified Rate-Distortion Centroid Refinement & Fine Grid Optimization.
Phase 2: Multi-Core Parallel Evaluation of the N x K Rate-Distortion Cost Matrix.
Phase 3: Fast Index-Based Decoupled Viterbi Sequence Optimization.
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

# Setup project paths
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global 3D Codebook Projection Sweep (Live-Writing & Fine-Grained)"
    )
    parser.add_argument(
        "--block-sizes", type=str, default="16,32",
        help="Comma-separated list of block sizes, e.g. '16,32'. Default: 16,32",
    )
    parser.add_argument(
        "--k-values", type=str, default="4,8,12,16",
        help="Comma-separated list of K (codebook size) values, e.g. '4,8,12,16'. Default: 4,8,12,16",
    )
    parser.add_argument(
        "--gammas", type=str,
        default="250,500,750,1000,1500,2000,2500,3000,3500,4000,5000,7500",
        help="Comma-separated list of gamma0 values for fine sweep. Default: 250..7500",
    )
    parser.add_argument(
        "--q-step", type=int, default=24,
        help="Quantization step used by the Decider. Default: 24",
    )
    parser.add_argument(
        "--rd-iters", type=int, default=5,
        help="Number of R-D centroid refinement iterations in Phase 1.5. Default: 5",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=0.01,
        help="Sampling ratio used when computing structural baseline rate. Default: 0.01",
    )
    parser.add_argument(
        "--n-strata", type=int, default=5,
        help="Number of strata used when computing structural baseline rate. Default: 5",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(project_root / "config/base_config.yaml"),
        help="Path to the experiment config yaml.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb",
        help="Directory to write the results report into.",
    )
    parser.add_argument(
        "--output-name", type=str,
        default="real_run_global_3d_results.md",
        help="Filename for the results report.",
    )
    return parser.parse_args()


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def log(msg):
    """Timestamped console log for tracking execution in real time."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def spherical_kmeanspp(grads, K_hybrid):
    """Spherical K-Means++ Initialization on 3D direction vectors."""
    np.random.seed(42)
    centroids = []
    first_idx = np.random.choice(len(grads))
    c0 = grads[first_idx]
    norm0 = np.linalg.norm(c0)
    centroids.append(c0 / (norm0 if norm0 > 1e-8 else 1.0))

    for _ in range(1, K_hybrid):
        dists = []
        for g in grads:
            min_d = max(0.0, min(1.0 - np.abs(np.dot(g, c)) ** 2 for c in centroids))
            dists.append(min_d)
        dists = np.array(dists)
        sum_dists = dists.sum()
        if sum_dists == 0.0:
            probs = np.ones(len(grads)) / len(grads)
        else:
            probs = dists / sum_dists
        next_c = grads[np.random.choice(len(grads), p=probs)]
        next_norm = np.linalg.norm(next_c)
        centroids.append(next_c / (next_norm if next_norm > 1e-8 else 1.0))
    return np.array(centroids)


def get_stratified_indices(indices, max_samples):
    """Draws an evenly distributed stratified sample along the Morton ordering."""
    n = len(indices)
    if n <= max_samples:
        return np.array(indices, dtype=int)
    idx_positions = np.linspace(0, n - 1, max_samples, dtype=int)
    return np.array(indices)[idx_positions]


def compute_entropy(labels, K):
    """Computes conditional entropy H(X_i | X_{i-1})."""
    cluster_sizes = np.bincount(labels, minlength=K)
    trans = np.ones((K, K))
    for i in range(1, len(labels)):
        trans[labels[i - 1], labels[i]] += 1
    trans_probs = trans / trans.sum(axis=1, keepdims=True)
    h_cond = 0.0
    for prev in range(K):
        p_prev = cluster_sizes[prev] / len(labels)
        for cur in range(K):
            p = trans_probs[prev, cur]
            if p > 0 and p_prev > 0:
                h_cond -= p_prev * p * np.log2(p)
    return h_cond


def prepare_blocks(pc, bsize):
    """Pre-computes geometry, tangents, structural GFTs, and 3D gradients."""
    t0 = time.time()
    log(f"Partitioning point cloud into blocks of size B={bsize}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=bsize)
    N = len(blocks)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)
    log(f"Extracted {N} blocks containing {total_v} total voxels.")

    log("Extracting block tangent planes...")
    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    coords = []
    normals = []
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        coords.append(np.mean(block.Vblock, axis=0))
        normals.append(tangent_data[i]["v3"])
        block.clear_data()

    coords = np.array(coords)
    normals = np.array(normals)

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

    log("Extracting 3D attribute gradients...")
    g3d_norm = []
    for i, block in enumerate(blocks):
        block.init_data(pc.V, pc.A)
        V_centered = block.Vblock - np.mean(block.Vblock, axis=0)
        X = np.column_stack(
            (V_centered @ tangent_data[i]["v1"], V_centered @ tangent_data[i]["v2"])
        )
        Y = block.Ablock[:, 0] - np.mean(block.Ablock[:, 0])
        s2d, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        g3d = s2d[0] * tangent_data[i]["v1"] + s2d[1] * tangent_data[i]["v2"]
        norm = np.linalg.norm(g3d)
        if norm > 1e-8:
            g3d_norm.append(g3d / norm)
        else:
            g3d_norm.append(np.array([0.0, 0.0, 1.0]))
        block.clear_data()
    g3d_norm = np.array(g3d_norm)

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
        "normal_weights": normal_weights,
    }


def compute_structural_base_rate(pc, blocks, q_step, sample_ratio=0.01, n_strata=5):
    """Computes the structural-only baseline rate (bits per voxel)."""
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
        _, r_s, d_s = decider._RDcost(coeffs_s)

        total_r_s += r_s
        block.clear_data()

    base_rate = total_r_s / num_v
    log(f"Baseline rate calculated: {base_rate:.5f} bpv (computed in {time.time() - t0:.2f}s)")
    return base_rate


def _eval_single_block_cluster_k_cost(
    Vblock, Ablock, metadata, v1, v2, mu_k, k, p, w, decider_mode, lagrange, q_step
):
    dec = Decider(decider_mode, lagrange)
    dec._set_vars(q_step)
    gft = GFTStrategyWraper()
    app = Approximator()

    b = Block(metadata)
    b.Vblock = Vblock
    b.Ablock = Ablock

    s2d_proj = np.array([np.dot(mu_k, v1), np.dot(mu_k, v2)])
    s3d = s2d_proj[0] * v1 + s2d_proj[1] * v2

    s_graph = StructuralGraph(metadata)
    s_graph.set_data(Vblock)
    a_graph = AttributeGraph(s_graph, s3d, k, p, w)

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


def _eval_block_k_costs_worker(
    block_id, Vblock, Ablock, metadata, struct_coeffs,
    v1, v2, centroids_3d, slp, slw, decider_mode, lagrange, q_step, K
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

    for k in range(1, K):
        c_idx = k - 1
        mu_k = centroids_3d[c_idx]

        s2d_proj = np.array([np.dot(mu_k, v1), np.dot(mu_k, v2)])
        s3d = s2d_proj[0] * v1 + s2d_proj[1] * v2

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

    return costs


def train_and_score_codebook(ctx, K, decider_mode, lagrange, q_step, rd_iters=5):
    blocks = ctx["blocks"]
    N = ctx["N"]
    tangent_data = ctx["tangent_data"]
    structural_coeffs_map = ctx["structural_coeffs_map"]
    g3d_norm = ctx["g3d_norm"]

    K_hybrid = K - 1

    log(f"--- PHASE 1: Geometric Spherical EM Training (K={K}) ---")
    t0 = time.time()
    centroids_3d = spherical_kmeanspp(g3d_norm, K_hybrid)
    labels_em = np.zeros(N, dtype=int)

    for it in range(1, 21):
        for i in range(N):
            labels_em[i] = np.argmax([np.abs(np.dot(g3d_norm[i], c)) for c in centroids_3d])

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
                centroids_3d[k] = sum_g / norm_sum
            else:
                centroids_3d[k] = centroids_3d[k] / (np.linalg.norm(centroids_3d[k]) + 1e-12)

    log(f"Phase 1 EM converged in {time.time() - t0:.2f}s.")
    log(f"--- Phase 1 EM Directional Centroids & Initial Counts ---")
    for k in range(K_hybrid):
        c_dir = centroids_3d[k]
        c_mag = np.linalg.norm(c_dir)
        n_assigned = np.sum(labels_em == k)
        log(f"  Cluster {k+1} Direction: [{c_dir[0]:.3f}, {c_dir[1]:.3f}, {c_dir[2]:.3f}] | Norm: {c_mag:.4f} | EM Blocks: {n_assigned} ({n_assigned/N*100:.1f}%)")

    log(f"--- PHASE 1.5: Deep R-D Centroid Refinement ({rd_iters} Iters) & Grid Search ---")
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.45)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 1.5)])

    strat_sample_idx = get_stratified_indices(np.arange(N), max_samples=min(1000, max(300, int(0.30 * N))))
    log(f"Selected {len(strat_sample_idx)} stratified sample blocks for R-D refinement.")

    for rd_iter in range(1, rd_iters + 1):
        t_iter = time.time()
        rd_tasks = []
        for idx in strat_sample_idx:
            block = blocks[idx]
            block.init_data(ctx["pc_V"], ctx["pc_A"])
            rd_tasks.append((
                block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
                structural_coeffs_map[block.block_id],
                tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                centroids_3d, slp, slw, decider_mode, lagrange, q_step, K
            ))
            block.clear_data()

        sample_costs = Parallel(n_jobs=-1)(delayed(_eval_block_k_costs_worker)(*t) for t in rd_tasks)
        sample_costs = np.array(sample_costs)

        min_assignments = np.argmin(sample_costs, axis=1)
        for k_idx in range(K_hybrid):
            k = k_idx + 1
            assigned_mask = (min_assignments == k)
            assigned_blocks = strat_sample_idx[assigned_mask]

            if len(assigned_blocks) >= 3:
                sum_g = np.zeros(3)
                mu_k = centroids_3d[k_idx]
                for idx in assigned_blocks:
                    g = g3d_norm[idx]
                    sign = np.sign(np.dot(g, mu_k))
                    if sign == 0:
                        sign = 1.0
                    sum_g += sign * g
                norm_sum = np.linalg.norm(sum_g)
                if norm_sum > 1e-8:
                    centroids_3d[k_idx] = sum_g / norm_sum

        log(f"  Refinement Iter {rd_iter}/{rd_iters} completed in {time.time() - t_iter:.2f}s.")

    # Restrained candidate grid based on strict physical/graph bounds
    p_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    w_grid = [0.5, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    log(f"Optimizing graph parameters over constrained grid ({len(p_grid)}x{len(w_grid)} candidates: SLP in [0.2, 0.9], SLW in [0.5, 3.0])...")

    for k in range(1, K):
        c_idx = k - 1
        mask = (labels_em == c_idx)
        cluster_indices = np.where(mask)[0]
        c_dir = centroids_3d[c_idx]
        c_mag = np.linalg.norm(c_dir)

        if len(cluster_indices) < 5:
            log(f"   Cluster {k} (Centroid: [{c_dir[0]:.3f}, {c_dir[1]:.3f}, {c_dir[2]:.3f}] | Mag: {c_mag:.4f}): Skipped (too few EM blocks: {len(cluster_indices)}). Default slp={slp[k]:.2f}, slw={slw[k]:.2f}")
            continue

        sample_idx = get_stratified_indices(cluster_indices, max_samples=300)
        sample_data = []
        for idx in sample_idx:
            block = blocks[idx]
            block.init_data(ctx["pc_V"], ctx["pc_A"])
            sample_data.append((
                block.Vblock.copy(), block.Ablock.copy(), block.metadata,
                tangent_data[idx]["v1"], tangent_data[idx]["v2"]
            ))
            block.clear_data()

        best_cost = 1e18
        best_p, best_w = slp[k], slw[k]

        for p in p_grid:
            for w in w_grid:
                grid_tasks = [
                    (Vb, Ab, meta, v1, v2, centroids_3d[c_idx], k, p, w, decider_mode, lagrange, q_step)
                    for Vb, Ab, meta, v1, v2 in sample_data
                ]
                costs = Parallel(n_jobs=-1)(delayed(_eval_single_block_cluster_k_cost)(*t) for t in grid_tasks)
                c_sum = sum(costs)

                if c_sum < best_cost:
                    best_cost = c_sum
                    best_p, best_w = p, w

        slp[k], slw[k] = best_p, best_w
        log(f"   Cluster {k} (Centroid: [{c_dir[0]:.3f}, {c_dir[1]:.3f}, {c_dir[2]:.3f}] | Mag: {c_mag:.4f} | EM Blocks: {len(cluster_indices)}): Best slp = {best_p:.2f}, slw = {best_w:.2f}")

    log(f"--- PHASE 2: Evaluating Full {N} x {K} Cost Matrix across CPU cores ---")
    t0 = time.time()
    eval_tasks = []
    for i, block in enumerate(blocks):
        block.init_data(ctx["pc_V"], ctx["pc_A"])
        eval_tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            structural_coeffs_map[block.block_id],
            tangent_data[i]["v1"], tangent_data[i]["v2"],
            centroids_3d, slp, slw, decider_mode, lagrange, q_step, K
        ))
        block.clear_data()

    cost_matrix = Parallel(n_jobs=-1)(delayed(_eval_block_k_costs_worker)(*t) for t in eval_tasks)
    cost_matrix = np.array(cost_matrix)
    log(f"Phase 2 Cost Matrix completed in {time.time() - t0:.2f}s.")

    # Phase 2 Allocation Breakdown Logging
    min_assignments = np.argmin(cost_matrix, axis=1)
    cluster_counts = np.bincount(min_assignments, minlength=K)
    log(f"\n--- Phase 2 Direct Min-Cost Block Allocation Breakdown (Total: {N} blocks) ---")
    log(f"  Cluster 0 (Flat Transform Fallback) : {cluster_counts[0]} blocks ({cluster_counts[0]/N*100:.2f}%)")
    for k in range(1, K):
        c_dir = centroids_3d[k - 1]
        c_mag = np.linalg.norm(c_dir)
        log(f"  Cluster {k} (Dir: [{c_dir[0]:.3f}, {c_dir[1]:.3f}, {c_dir[2]:.3f}], Mag: {c_mag:.4f}, slp={slp[k]:.2f}, slw={slw[k]:.2f}) : {cluster_counts[k]} blocks ({cluster_counts[k]/N*100:.2f}%)")

    return cost_matrix


def run_viterbi(ctx, cost_matrix, K, gamma0):
    """Phase 3: Standard Index-Based Decoupled Viterbi Pass."""
    N = ctx["N"]
    normal_weights = ctx["normal_weights"]

    dp = np.zeros((N, K))
    paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]

    for i in range(1, N):
        w = normal_weights[i]
        for k in range(K):
            transition_costs = dp[i - 1] + gamma0 * w * (np.arange(K) != k)
            best_prev = np.argmin(transition_costs)
            dp[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
            paths[i, k] = best_prev

    viterbi_labels = np.zeros(N, dtype=int)
    viterbi_labels[N - 1] = np.argmin(dp[N - 1])
    for i in range(N - 2, -1, -1):
        viterbi_labels[i] = paths[i + 1, viterbi_labels[i + 1]]

    return viterbi_labels


def main():
    args = parse_args()
    block_sizes = parse_int_list(args.block_sizes)
    k_values = parse_int_list(args.k_values)
    gammas = parse_float_list(args.gammas)
    q_step = args.q_step

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / args.output_name

    log(f"Initializing Experiment. Streaming results directly to: {report_file.absolute()}")

    with open(report_file, "w") as f:
        f.write("# Global 3D Projection Decoupled Viterbi Results (Real Run)\n\n")
        f.write(
            "| B | K | Method | Baseline $\\gamma_0$ | GFT Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches (Spatial) |\n"
        )
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
        log(f"\n============================================================")
        log(f"STARTING BLOCK SIZE B = {bsize}")
        log(f"============================================================")
        ctx = prepare_blocks(pc, bsize)
        ctx["pc_V"] = pc.V
        ctx["pc_A"] = pc.A
        total_v = ctx["total_v"]
        N = ctx["N"]

        base_rate = compute_structural_base_rate(
            pc, ctx["blocks"], q_step,
            sample_ratio=args.sample_ratio, n_strata=args.n_strata,
        )

        for K in k_values:
            log(f"\n>>>> Starting Codebook K = {K} (Block Size B = {bsize}) <<<<")
            cost_matrix = train_and_score_codebook(
                ctx, K, decider_mode, lagrange, q_step, rd_iters=args.rd_iters
            )
            structural_cost = sum(cost_matrix[i, 0] for i in range(N))

            log(f"--- PHASE 3: Running Fast Viterbi Sweeps across {len(gammas)} Gamma values ---")
            for gamma0 in gammas:
                viterbi_labels = run_viterbi(ctx, cost_matrix, K, gamma0)

                gft_cost_viterbi = sum(cost_matrix[i, viterbi_labels[i]] for i in range(N))
                savings_pct = (structural_cost - gft_cost_viterbi) / structural_cost * 100
                entropy = compute_entropy(viterbi_labels, K)
                overhead = entropy * N / total_v
                net_bpv = base_rate * (savings_pct / 100.0) - overhead
                switches = int(np.sum(viterbi_labels[1:] != viterbi_labels[:-1]))

                v_counts = np.bincount(viterbi_labels, minlength=K)
                flat_pct = v_counts[0] / N * 100.0
                dir_pct = (N - v_counts[0]) / N * 100.0

                row_str = (
                    f"| {bsize} | {K} | Global 3D Projection | {gamma0:.1f} | {savings_pct:.3f}% | "
                    f"{entropy:.4f} | {overhead:.5f} | {net_bpv:+.5f} | {switches} |"
                )

                log(f"RESULT (Gamma={gamma0:.1f} | Flat: {v_counts[0]} [{flat_pct:.1f}%], Dir: {N - v_counts[0]} [{dir_pct:.1f}%]): {row_str}")
                with open(report_file, "a") as f:
                    f.write(row_str + "\n")
                    f.flush()
                    os.fsync(f.fileno())

    log(f"\nSweep completed successfully! All results saved to: {report_file.absolute()}")


if __name__ == "__main__":
    main()
