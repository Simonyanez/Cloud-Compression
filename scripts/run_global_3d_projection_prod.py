"""
Global 3D Codebook Projection Sweep
==================================
Phase 1: Unconstrained Global 3D EM Training (gamma = 0) on block 3D gradients.
Phase 2: Per-Block Local 2D Projection Operator.
Phase 3: Standard Index-Based Decoupled Viterbi Pass.

Refactored to accept multiple block sizes, K values, gamma values, and a
q_step, all via CLI flags. Logic is unchanged from the original single-run
script -- block size and K are now outer loops (since they change block
partitioning / codebook training), gamma stays the cheap inner loop exactly
as before.

The structural baseline rate (previously a hardcoded `base_rate = 0.1480`,
only valid for B=16) is now computed directly for the active block size and
q_step via `compute_structural_base_rate`, which mirrors the structural
branch of the RD-point calculation. It is computed once per block size
(rate doesn't depend on K or gamma), on a 1%-strata sample for speed, same
as the original RD-point sweep.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

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
        description="Global 3D Codebook Projection Sweep (parameterized)"
    )
    parser.add_argument(
        "--block-sizes", type=str, default="16",
        help="Comma-separated list of block sizes, e.g. '4,8,16,32'. Default: 16",
    )
    parser.add_argument(
        "--k-values", type=str, default="4",
        help="Comma-separated list of K (codebook size) values, e.g. '3,4,6,8'. Default: 4",
    )
    parser.add_argument(
        "--gammas", type=str, default="2000,3000,4000",
        help="Comma-separated list of gamma0 values, e.g. '2000,3000,4000'. Default: 2000,3000,4000",
    )
    parser.add_argument(
        "--q-step", type=int, default=24,
        help="Quantization step used by the Decider. Default: 24",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=0.01,
        help="Sampling ratio used when computing the structural baseline rate "
             "(mirrors get_rd_points' Sampler(ratio=...)). Default: 0.01",
    )
    parser.add_argument(
        "--n-strata", type=int, default=5,
        help="Number of strata used when computing the structural baseline rate "
             "(mirrors get_rd_points' Sampler(n_strata=...)). Default: 5",
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
        default="global_3d_projection_results.md",
        help="Filename for the results report.",
    )
    return parser.parse_args()


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# Spherical K-Means++ Initialization helper
def spherical_kmeanspp(grads, K_hybrid):
    np.random.seed(42)
    centroids = [grads[np.random.choice(len(grads))]]
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
        centroids.append(grads[np.random.choice(len(grads), p=probs)])
    return np.array(centroids)


# Helper to compute conditional entropy
def compute_entropy(labels, K):
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
    """Everything that depends only on block size (not K)."""
    _, blocks = MortonBlockPartition().partition(pc, bsize=bsize)
    N = len(blocks)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)

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

    print("Extracting unconstrained 3D block gradients...")
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
    """
    Computes the structural-only baseline rate (bits per voxel) for the
    given block partition at a given q_step, replacing the previously
    hardcoded `base_rate = 0.1480` constant (which was only valid for B=16).

    This mirrors the structural branch of `get_rd_points` exactly (same
    Sampler, same Decider(mode="0", lagrange_proportional=0.8) config, same
    RD-cost call) -- the hybrid branch is dropped since the main sweep only
    ever used the structural rate as its baseline.
    """
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

    return total_r_s / num_v


def train_and_score_codebook(ctx, K, decider_mode, lagrange, q_step):
    """Everything that depends on block size AND K (but not gamma)."""
    blocks = ctx["blocks"]
    N = ctx["N"]
    tangent_data = ctx["tangent_data"]
    structural_coeffs_map = ctx["structural_coeffs_map"]
    g3d_norm = ctx["g3d_norm"]

    K_hybrid = K - 1

    print(f"\n============================================================")
    print(f"PHASE 1: TRAINING UNCONSTRAINED GLOBAL 3D EM")
    print(f"============================================================")

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
            if norm_sum > 0:
                centroids_3d[k] = sum_g / norm_sum

    print("\n[Phase 1 Done] Global 3D Codebook Centroids Learned:")
    for k in range(K_hybrid):
        print(f"  Centroid {k+1}: [{centroids_3d[k,0]:+.3f}, {centroids_3d[k,1]:+.3f}, {centroids_3d[k,2]:+.3f}]")

    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.65)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 1.4)])

    print("\nOptimizing loop parameters (slp, slw) on projected slopes...")
    for k in range(1, K):
        c_idx = k - 1
        mask = (labels_em == c_idx)
        n_ass = mask.sum()
        if n_ass < 5:
            continue

        Vb_list, Ab_list, meta_list, v1s, v2s = [], [], [], [], []
        sample_idx = np.random.choice(np.where(mask)[0], size=min(100, n_ass), replace=False)

        for idx in sample_idx:
            block = blocks[idx]
            block.init_data(ctx["pc_V"], ctx["pc_A"])
            Vb_list.append(block.Vblock.copy())
            Ab_list.append(block.Ablock.copy())
            meta_list.append(block.metadata)
            v1s.append(tangent_data[idx]["v1"])
            v2s.append(tangent_data[idx]["v2"])
            block.clear_data()

        best_cost = 1e18
        best_p, best_w = slp[k], slw[k]

        for p in [0.293, 0.447, 0.720]:
            for w in [0.325, 0.550, 0.872]:
                c_sum = 0.0
                for Vb, Ab, meta, v1, v2 in zip(Vb_list, Ab_list, meta_list, v1s, v2s):
                    b = Block(meta)
                    b.Vblock = Vb
                    b.Ablock = Ab

                    s2d_proj = np.array([np.dot(centroids_3d[c_idx], v1), np.dot(centroids_3d[c_idx], v2)])
                    s3d = s2d_proj[0] * v1 + s2d_proj[1] * v2

                    s_graph = StructuralGraph(meta)
                    s_graph.set_data(Vb)
                    a_graph = AttributeGraph(s_graph, s3d, k, p, w)

                    app = Approximator()
                    V_rot = app._spatial_norm(Vb)
                    A_app = Ab.copy()
                    A_app[:, 0] = V_rot @ s3d.T
                    a_graph.set_data(Vb, A_app)

                    try:
                        gft = GFTStrategyWraper()
                        dec = Decider(decider_mode, lagrange)
                        dec._set_vars(q_step)
                        _, coeffs = gft(b, a_graph)
                        c, _, _ = dec._RDcost(coeffs)
                        c_sum += c
                    except Exception:
                        c_sum += 1e18
                if c_sum < best_cost:
                    best_cost = c_sum
                    best_p = p
                    best_w = w
        slp[k] = best_p
        slw[k] = best_w

    print(f"Loop parameters optimized: slp = {slp[1:]}, slw = {slw[1:]}")

    print("\nEvaluating GFT cost matrix with projected global codebook...")
    cost_matrix = []
    for i, block in enumerate(blocks):
        block.init_data(ctx["pc_V"], ctx["pc_A"])
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
            c_idx = k - 1
            s2d_proj = np.array([
                np.dot(centroids_3d[c_idx], tangent_data[i]["v1"]),
                np.dot(centroids_3d[c_idx], tangent_data[i]["v2"]),
            ])
            s3d = s2d_proj[0] * tangent_data[i]["v1"] + s2d_proj[1] * tangent_data[i]["v2"]

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
    return cost_matrix


def run_viterbi(ctx, cost_matrix, K, gamma0):
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

    # 1. LOAD DATA (once)
    params = load_experiment_config(Path(args.config))
    colourist = Colourist()
    pc = PointCloud.from_file(
        project_root / params.sequential_params.point_cloud_path, "ply", params.pointcloud
    )
    pc.transform_attributes(colourist._RGBtoYUV)

    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional

    results_table = []
    results_table.append("# Global 3D Projection Decoupled Viterbi Results\n")
    results_table.append(
        "| B | K | Method | Baseline $\\gamma_0$ | GFT Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches (Spatial) |"
    )
    results_table.append("|---:|---:|:---|---:|---:|---:|---:|---:|---:|")

    for bsize in block_sizes:
        print(f"\n############################################################")
        print(f"# BLOCK SIZE = {bsize}")
        print(f"############################################################")
        ctx = prepare_blocks(pc, bsize)
        ctx["pc_V"] = pc.V
        ctx["pc_A"] = pc.A
        total_v = ctx["total_v"]
        N = ctx["N"]

        print(f"Computing structural baseline rate for B={bsize}, q_step={q_step}...")
        base_rate = compute_structural_base_rate(
            pc, ctx["blocks"], q_step,
            sample_ratio=args.sample_ratio, n_strata=args.n_strata,
        )
        print(f"  -> base_rate = {base_rate:.5f} bpv")

        for K in k_values:
            print(f"\n---- K = {K} (block size {bsize}) ----")
            cost_matrix = train_and_score_codebook(ctx, K, decider_mode, lagrange, q_step)
            structural_cost = sum(cost_matrix[i, 0] for i in range(N))

            for gamma0 in gammas:
                print(f"Running Index-Based Viterbi for B={bsize}, K={K}, gamma0={gamma0:.1f}...")
                viterbi_labels = run_viterbi(ctx, cost_matrix, K, gamma0)

                gft_cost_viterbi = sum(cost_matrix[i, viterbi_labels[i]] for i in range(N))
                savings_pct = (structural_cost - gft_cost_viterbi) / structural_cost * 100
                entropy = compute_entropy(viterbi_labels, K)
                overhead = entropy * N / total_v
                net_bpv = base_rate * (savings_pct / 100.0) - overhead
                switches = int(np.sum(viterbi_labels[1:] != viterbi_labels[:-1]))

                results_table.append(
                    f"| {bsize} | {K} | Global 3D Projection | {gamma0:.1f} | {savings_pct:.3f}% | "
                    f"{entropy:.4f} | {overhead:.5f} | {net_bpv:+.5f} | {switches} |"
                )

    # Write report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / args.output_name

    with open(report_file, "w") as f:
        f.write("\n".join(results_table) + "\n")

    print(f"\nExperiment completed successfully! Report written to: {report_file.absolute()}")


if __name__ == "__main__":
    main()
