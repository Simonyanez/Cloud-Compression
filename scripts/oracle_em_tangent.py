"""
Tangent-Space EM Codebook Optimizer (Phase 2)
==============================================
Implements the 2D tangent plane projection method:
  1. For each block, computes principal coordinate eigenvectors v1, v2 (tangent plane)
     and discards the normal vector n.
  2. The cluster parameters are 2D slopes s_2d = [s1, s2] representing luminance variation
     along the surface fabric, which is rotation-invariant.
  3. The 3D slope for E-step and M-step is mapped back locally: s_3d = s1*v1 + s2*v2.
  4. Prevents the "antagonistic slope cancellation" by aligning slopes to local geometry.
"""

import sys
import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
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
# DATABASE SETUP
# ===========================================================================

def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        block_size INTEGER,
        clusters INTEGER,
        q_step INTEGER,
        gamma REAL,
        gamma_warmup INTEGER,
        freeze_slopes INTEGER,
        spatially_adaptive INTEGER,
        max_iters INTEGER,
        dataset TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS iterations (
        run_id INTEGER,
        iteration INTEGER,
        total_cost REAL,
        structural_cost REAL,
        rate_reduction_pct REAL,
        entropy_markov REAL,
        overhead_bpv_markov REAL,
        net_bpv_markov REAL,
        num_switches INTEGER,
        avg_l1_norm REAL,
        structural_l1_norm REAL,
        labels_json TEXT,
        PRIMARY KEY (run_id, iteration)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cluster_states (
        run_id INTEGER,
        iteration INTEGER,
        cluster_id INTEGER,
        cluster_size INTEGER,
        slope_2d_1 REAL,
        slope_2d_2 REAL,
        slp REAL,
        slw REAL,
        PRIMARY KEY (run_id, iteration, cluster_id)
    )
    """)
    conn.commit()
    conn.close()


# ===========================================================================
# LOCAL TANGENT GEOMETRY EXTRACTOR
# ===========================================================================

def extract_block_tangent_planes(blocks: List, pc_V: np.ndarray, pc_A: np.ndarray) -> List[dict]:
    print("\n[Tangent] Pre-computing local coordinate tangent planes (PCA)...")
    tangent_data = []
    for block in tqdm(blocks, desc="PCA Tangent Vectors"):
        block.init_data(pc_V, pc_A)
        V = block.Vblock
        
        # Default fallback
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        V_centered = np.zeros_like(V)
        
        if V.shape[0] >= 4:
            V_centered = V - np.mean(V, axis=0)
            cov = np.dot(V_centered.T, V_centered) / (V.shape[0] - 1)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Sort descending: largest spread is v1 (eigenvectors[:, 2]), second largest is v2 (eigenvectors[:, 1])
                v1 = eigenvectors[:, 2]
                v2 = eigenvectors[:, 1]
                v3 = eigenvectors[:, 0]
            except np.linalg.LinAlgError:
                pass
                
        # Project 3D centered coordinates onto tangent plane -> shape (N, 2)
        X_tangent = np.column_stack((V_centered @ v1, V_centered @ v2))
        
        tangent_data.append({
            "block_id": block.block_id,
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "X_tangent": X_tangent
        })
        block.clear_data()
        
    return tangent_data


# ===========================================================================
# E-STEP WORKER
# ===========================================================================

def _evaluate_block_costs_worker(block_id, Vblock, Ablock, metadata,
                                  slopes_2d, slp_vals, slw_vals,
                                  v1, v2, structural_coeffs, q_step,
                                  lagrange_proportional, decider_mode):
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

    K = len(slopes_2d)
    costs = np.full(K, np.inf)

    for k in range(K):
        if k == 0:
            c, _, _ = decider._RDcost(structural_coeffs)
            costs[k] = c
        else:
            # Map 2D slope back to 3D locally
            s2d = slopes_2d[k]
            slope_3d = s2d[0] * v1 + s2d[1] * v2
            
            slw = slw_vals[k]
            slp = slp_vals[k]
            s_graph = StructuralGraph(metadata)
            s_graph.set_data(Vblock)
            a_graph = AttributeGraph(s_graph, slope_3d, k, slp, slw)

            V_rot = approximator._spatial_norm(Vblock)
            A_app = Ablock.copy()
            A_app[:, 0] = V_rot @ slope_3d.T
            a_graph.set_data(Vblock, A_app)

            _, coeffs = gft_computer(block, a_graph)
            c, _, _ = decider._RDcost(coeffs)
            costs[k] = c

    return costs


def e_step(blocks: List, structural_coeffs_map: dict,
           slopes_2d: np.ndarray, slp: np.ndarray, slw: np.ndarray,
           tangent_data: List[dict], q_step: int, lagrange: float, decider_mode: str,
           vertices: np.ndarray, attributes: np.ndarray,
           gamma: float, prev_labels: Optional[np.ndarray],
           spatial_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tasks = []
    for i, block in enumerate(blocks):
        block.init_data(vertices, attributes)
        tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            slopes_2d, slp, slw,
            tangent_data[i]["v1"], tangent_data[i]["v2"],
            structural_coeffs_map[block.block_id],
            q_step, lagrange, decider_mode
        ))
        block.clear_data()

    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_block_costs_worker)(*t)
        for t in tasks
    )

    cost_matrix = np.array(results)
    N, K = cost_matrix.shape
    new_labels = np.zeros(N, dtype=int)

    if gamma == 0.0 or prev_labels is None:
        new_labels = np.argmin(cost_matrix, axis=1)
    else:
        # Exact Viterbi sequence optimization
        dp = np.zeros((N, K))
        paths = np.zeros((N, K), dtype=int)
        dp[0] = cost_matrix[0]
        
        for i in range(1, N):
            w = spatial_weights[i]
            for k in range(K):
                # Transition costs from all predecessor states
                transition_costs = dp[i-1] + gamma * w * (np.arange(K) != k)
                best_prev = np.argmin(transition_costs)
                dp[i, k] = cost_matrix[i, k] + transition_costs[best_prev]
                paths[i, k] = best_prev
                
        # Backtrack optimal sequence
        new_labels[N-1] = np.argmin(dp[N-1])
        for i in range(N-2, -1, -1):
            new_labels[i] = paths[i+1, new_labels[i+1]]

    return new_labels, cost_matrix


# ===========================================================================
# M-STEP (REGRESSION ON 2D TANGENT COORDINATES)
# ===========================================================================

def _eval_grid_point_tangent(Vblocks_list, Ablocks_list, metadata_list, v1s_list, v2s_list,
                             s_2d_val, slp_val, slw_val,
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
    for Vblock, Ablock, metadata, v1, v2 in zip(Vblocks_list, Ablocks_list, metadata_list, v1s_list, v2s_list):
        block = Block(metadata)
        block.Vblock = Vblock
        block.Ablock = Ablock

        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)

        # Map 2D tangent slope back to local 3D space
        s3d = s_2d_val[0] * v1 + s_2d_val[1] * v2

        V_rot = approximator._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ s3d.T

        a_graph = AttributeGraph(s_graph, s3d, 1, slp_val, slw_val)
        a_graph.set_data(Vblock, A_app)
        
        try:
            _, coeffs = gft_computer(block, a_graph)
            c, _, _ = decider._RDcost(coeffs)
            total_cost += c
        except Exception:
            total_cost += 1e18

    return total_cost


def _evaluate_grid_parallel_tangent(Vblocks_list, Ablocks_list, metadata_list, v1s_list, v2s_list,
                                   s_2d_val, slp_vals, slw_vals,
                                   q_step, lagrange_proportional, decider_mode) -> np.ndarray:
    grid_points = [(pi, wi, p, w)
                   for pi, p in enumerate(slp_vals)
                   for wi, w in enumerate(slw_vals)]

    results = Parallel(n_jobs=-1)(
        delayed(_eval_grid_point_tangent)(
            Vblocks_list, Ablocks_list, metadata_list, v1s_list, v2s_list,
            s_2d_val, p, w,
            q_step, lagrange_proportional, decider_mode
        )
        for _, _, p, w in grid_points
    )

    total_costs = np.zeros((len(slp_vals), len(slw_vals)))
    for (pi, wi, _, _), cost in zip(grid_points, results):
        total_costs[pi, wi] = cost
    return total_costs


def m_step(blocks: List, labels: np.ndarray,
           slopes_2d: np.ndarray, slp: np.ndarray, slw: np.ndarray,
           tangent_data: List[dict], vertices: np.ndarray, attributes: np.ndarray,
           q_step: int, lagrange: float, decider_mode: str,
           min_cluster_size: int, freeze_slopes: bool,
           sample_rate: float = 1.0,
           centroid_learning_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(slopes_2d)
    new_slopes_2d = slopes_2d.copy()
    new_slp = slp.copy()
    new_slw = slw.copy()
    
    slp_coarse = np.linspace(0.02, 0.98, 10)
    slw_coarse = np.linspace(0.2, 6.0, 10)

    for k in range(1, K):
        mask = (labels == k)
        n_assigned = mask.sum()
        if n_assigned < min_cluster_size:
            continue

        print(f"  [M-step] Cluster {k}: {n_assigned} blocks (Tangent 2D regression, sampling={sample_rate:.2f})")
        
        # 1. Update 2D slope parameters
        if freeze_slopes:
            print(f"         slope_2d: [{slopes_2d[k,0]:+.3f}, {slopes_2d[k,1]:+.3f}] (FROZEN)")
        else:
            X_list = []
            Y_list = []
            s_prev = slopes_2d[k]
            for idx in np.where(mask)[0]:
                block = blocks[idx]
                block.init_data(vertices, attributes)
                
                if block.Vblock.shape[0] >= 4:
                    X = tangent_data[idx]["X_tangent"]
                    Y = block.Ablock[:, 0] - np.mean(block.Ablock[:, 0])
                    
                    # Compute local 2D unconstrained slope for alignment check
                    try:
                        s_local, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                        if np.dot(s_local, s_prev) < 0:
                            Y = -Y  # Flip sign to align with the cluster slope direction
                    except Exception:
                        pass
                        
                    X_list.append(X)
                    Y_list.append(Y)
                    
                block.clear_data()

            if X_list:
                X_all = np.vstack(X_list)
                Y_all = np.concatenate(Y_list)
                s2d_new, _, _, _ = np.linalg.lstsq(X_all, Y_all, rcond=None)
                new_s2d = (1.0 - centroid_learning_rate) * s_prev + centroid_learning_rate * s2d_new
                new_slopes_2d[k] = new_s2d
                print(f"         slope_2d: [{new_s2d[0]:+.3f}, {new_s2d[1]:+.3f}]")

        # 2. Fit graph loop parameters (slp, slw) on assigned blocks
        Vblocks, Ablocks, metadatas, v1s, v2s = [], [], [], [], []
        assigned_indices = np.where(mask)[0]
        if sample_rate < 1.0:
            n_sample = max(min_cluster_size, int(round(len(assigned_indices) * sample_rate)))
            if n_sample < len(assigned_indices):
                assigned_indices = np.random.choice(assigned_indices, size=n_sample, replace=False)

        for idx in assigned_indices:
            block = blocks[idx]
            block.init_data(vertices, attributes)
            Vblocks.append(block.Vblock.copy())
            Ablocks.append(block.Ablock.copy())
            metadatas.append(block.metadata)
            v1s.append(tangent_data[idx]["v1"])
            v2s.append(tangent_data[idx]["v2"])
            block.clear_data()

        # Coarse grid search
        s_2d_val = new_slopes_2d[k]
        coarse_costs = _evaluate_grid_parallel_tangent(
            Vblocks, Ablocks, metadatas, v1s, v2s,
            s_2d_val, slp_coarse, slw_coarse,
            q_step, lagrange, decider_mode
        )
        best_pi, best_wi = np.unravel_index(np.argmin(coarse_costs), coarse_costs.shape)
        best_p_c = slp_coarse[best_pi]
        best_w_c = slw_coarse[best_wi]

        # Fine refinement
        slp_fine = np.linspace(max(0.01, best_p_c - 0.12), min(0.99, best_p_c + 0.12), 5)
        slw_fine = np.linspace(max(0.1,  best_w_c - 0.8),  max(0.5, best_w_c + 0.8),  5)
        fine_costs = _evaluate_grid_parallel_tangent(
            Vblocks, Ablocks, metadatas, v1s, v2s,
            s_2d_val, slp_fine, slw_fine,
            q_step, lagrange, decider_mode
        )
        best_pf, best_wf = np.unravel_index(np.argmin(fine_costs), fine_costs.shape)

        new_slp[k] = slp_fine[best_pf]
        new_slw[k] = slw_fine[best_wf]
        print(f"         slp={new_slp[k]:.3f}  slw={new_slw[k]:.3f}")

    return new_slopes_2d, new_slp, new_slw


# ===========================================================================
# REPRESENTATION EVALUATION FOR TANGENT SPACE
# ===========================================================================

def _evaluate_assigned_coeffs_worker(block_id, Vblock, Ablock, metadata,
                                     s2d, slp, slw, k, v1, v2, structural_coeffs):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator
    from pcadc.blocks import Block

    if k == 0:
        coeffs = structural_coeffs
    else:
        s3d = s2d[0] * v1 + s2d[1] * v2
        
        gft_computer = GFTStrategyWraper()
        approximator = Approximator()
        block = Block(metadata)
        block.Vblock = Vblock
        block.Ablock = Ablock

        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, s3d, k, slp, slw)

        V_rot = approximator._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ s3d.T
        a_graph.set_data(Vblock, A_app)

        _, coeffs = gft_computer(block, a_graph)

    l1 = float(np.sum(np.abs(coeffs)))
    return l1


def compute_metrics(blocks: List, labels: np.ndarray, cost_matrix: np.ndarray,
                    vertices: np.ndarray, attributes: np.ndarray,
                    slopes_2d: np.ndarray, slp: np.ndarray, slw: np.ndarray,
                    tangent_data: List[dict], q_step: int, prev_labels: Optional[np.ndarray],
                    base_rate: float, structural_coeffs_map: dict,
                    struct_l1: float) -> dict:
    K = cost_matrix.shape[1]
    N = len(labels)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)

    total_cost = sum(cost_matrix[i, labels[i]] for i in range(N))
    cluster_sizes = np.bincount(labels, minlength=K)
    active = int((cluster_sizes > 0).sum())

    probs = cluster_sizes / cluster_sizes.sum()
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

    overhead_markov = h_cond * N / total_v
    structural_cost = sum(cost_matrix[i, 0] for i in range(N))
    rate_reduction_pct = (structural_cost - total_cost) / structural_cost * 100

    saved_bpv = base_rate * (rate_reduction_pct / 100.0)
    net_bpv_markov = saved_bpv - overhead_markov
    num_switches = int(np.sum(labels[1:] != labels[:-1]))

    tasks = []
    for i, block in enumerate(blocks):
        block.init_data(vertices, attributes)
        k = labels[i]
        tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            slopes_2d[k], slp[k], slw[k], k,
            tangent_data[i]["v1"], tangent_data[i]["v2"],
            structural_coeffs_map[block.block_id]
        ))
        block.clear_data()

    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_assigned_coeffs_worker)(*t)
        for t in tasks
    )
    avg_l1 = float(np.mean(results))

    return {
        "total_cost":          total_cost,
        "structural_cost":     structural_cost,
        "rate_reduction_pct":  rate_reduction_pct,
        "entropy_markov":      h_cond,
        "overhead_bpv_markov":   overhead_markov,
        "net_bpv_markov":      net_bpv_markov,
        "active_clusters":     active,
        "cluster_sizes":       cluster_sizes,
        "num_switches":        num_switches,
        "avg_l1_norm":          avg_l1,
        "structural_l1_norm":   struct_l1,
        "labels":              labels.copy()
    }


def apply_geometric_scan_order(blocks: List, pc_V: np.ndarray, pc_A: np.ndarray) -> Tuple[List, np.ndarray]:
    print("\n[Geometric Scan] Analyzing block normals and sorting by 2D azimuth angle (theta)...")
    normals = []
    for block in tqdm(blocks, desc="PCA Block Normals"):
        block.init_data(pc_V, pc_A)
        V = block.Vblock
        n = np.array([0.0, 0.0, 1.0])
        if V.shape[0] >= 4:
            V_centered = V - np.mean(V, axis=0)
            cov = np.dot(V_centered.T, V_centered) / (V.shape[0] - 1)
            try:
                _, eigenvectors = np.linalg.eigh(cov)
                # 3rd eigenvector corresponds to smallest eigenvalue (normal)
                n = eigenvectors[:, 0]
            except np.linalg.LinAlgError:
                pass
        normals.append(n)
        block.clear_data()
        
    normals = np.array(normals)
    thetas = np.arctan2(normals[:, 1], normals[:, 0])
    geom_order = np.argsort(thetas)
    return [blocks[idx] for idx in geom_order], geom_order


def main():
    parser = ArgumentParser(description="Tangent-Space EM Codebook Optimizer")
    parser.add_argument("--block_size",  type=int,   required=True)
    parser.add_argument("--clusters",    type=int,   required=True)
    parser.add_argument("--q_step",      type=int,   required=True)
    parser.add_argument("--gamma",       type=float, default=0.0)
    parser.add_argument("--max_iters",   type=int,   default=10)
    parser.add_argument("--gamma_warmup",type=int,   default=3)
    parser.add_argument("--freeze_slopes", action="store_true")
    parser.add_argument("--spatially_adaptive", action="store_true")
    parser.add_argument("--geometric_scan", action="store_true")
    parser.add_argument("--adaptive_normals", action="store_true")
    parser.add_argument("--centroid_lr",  type=float, default=1.0)
    parser.add_argument("--normal_threshold", type=float, default=0.0)
    parser.add_argument("--decoupled",    action="store_true")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--db_path",     type=str,   default="experiments/tangent_em.db")
    parser.add_argument("--config",      type=str,   default="config/base_config.yaml")
    args = parser.parse_args()

    n_hybrid = args.clusters - 1
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    init_db(db_path)

    params = load_experiment_config(Path(args.config))
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)

    _, blocks = MortonBlockPartition().partition(pc, bsize=args.block_size)
    
    if args.geometric_scan:
        blocks, geom_order = apply_geometric_scan_order(blocks, pc.V, pc.A)
        
    dataset_name = pc.metadata.source_id
    base_rate = 0.1603 if args.block_size == 8 else 0.1480

    print(f"\n[Tangent EM] Dataset: {dataset_name} | Block size: {args.block_size}")
    print(f"[Tangent EM] SQLite log: {db_path.absolute()}")

    # 1. Precompute local block tangent planes
    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    # 2. Spatially adaptive penalty sequence weights
    if args.spatially_adaptive:
        print("[Spatial] Computing centers...")
        centers = []
        for block in blocks:
            block.init_data(pc.V, pc.A)
            centers.append(np.mean(block.Vblock, axis=0))
            block.clear_data()
        centers = np.array(centers)
        dists = np.sqrt(np.sum((centers[1:] - centers[:-1]) ** 2, axis=1))
        sigma = 1.5 * np.mean(dists)
        spatial_weights = np.exp(-dists / sigma)
        spatial_weights = np.concatenate([[1.0], spatial_weights])
    else:
        spatial_weights = np.ones(len(blocks))

    # Apply adaptive normal alignment weighting if requested
    if args.adaptive_normals:
        print("[Adaptive Normals] Modulating transition weights by normal alignment |n_i . n_{i-1}|...")
        normal_weights = np.ones(len(blocks))
        for i in range(1, len(blocks)):
            n_prev = tangent_data[i-1]["v3"]
            n_curr = tangent_data[i]["v3"]
            val = np.abs(np.dot(n_curr, n_prev))
            if args.normal_threshold > 0.0:
                normal_weights[i] = 1.0 if val >= args.normal_threshold else 0.0
            else:
                normal_weights[i] = val
        spatial_weights = spatial_weights * normal_weights

    # 3. Precompute structural GFT coeffs
    print("\n[Precompute] Structural GFT coefficients...")
    gft_computer = GFTStrategyWraper()
    structural_coeffs_map = {}
    for block in tqdm(blocks, desc="Structural GFT"):
        block.init_data(pc.V, pc.A)
        s_graph = StructuralGraph(block.metadata)
        s_graph.set_data(block.Vblock)
        _, coeffs = gft_computer(block, s_graph)
        structural_coeffs_map[block.block_id] = coeffs
        block.clear_data()

    # Precompute baseline L1 norm
    struct_l1 = float(np.mean([np.sum(np.abs(c)) for c in structural_coeffs_map.values()]))
    print(f"  Structural L1 Norm: {struct_l1:.2f}")

    # Initialize 2D slopes to distribute them across the 2D tangent plane
    # We place them at uniform angles on a circle in the 2D tangent plane
    slopes_2d = np.zeros((args.clusters, 2))
    for k in range(1, args.clusters):
        angle = 2 * np.pi * (k - 1) / n_hybrid
        # Distribute with radius 0.5
        slopes_2d[k] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
        
    slp = np.concatenate([[0.0], np.full(n_hybrid, 0.65)])
    slw = np.concatenate([[1.0], np.full(n_hybrid, 1.4)])

    print("\n[Init] Initial 2D slope parameters:")
    for k in range(len(slopes_2d)):
        tag = "(structural)" if k == 0 else ""
        print(f"  Cluster {k}: s2d = [{slopes_2d[k,0]:+.3f}, {slopes_2d[k,1]:+.3f}] {tag}")

    # Register run
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO runs (
        timestamp, block_size, clusters, q_step, gamma, gamma_warmup, 
        freeze_slopes, spatially_adaptive, max_iters, dataset
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(), args.block_size, args.clusters, args.q_step,
        args.gamma, args.gamma_warmup, 1 if args.freeze_slopes else 0,
        1 if args.spatially_adaptive else 0, args.max_iters, dataset_name
    ))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    prev_labels = None
    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional
    min_cluster_size = max(5, int(0.01 * len(blocks)))
    
    history = []

    for iteration in range(1, args.max_iters + 1):
        if args.decoupled:
            if iteration < args.max_iters:
                effective_gamma = 0.0
            else:
                effective_gamma = args.gamma
        else:
            if iteration <= args.gamma_warmup:
                effective_gamma = 0.0
            else:
                steps = args.max_iters - args.gamma_warmup
                current_step = iteration - args.gamma_warmup
                effective_gamma = args.gamma * (current_step / steps)

        print(f"\n── Iteration {iteration}/{args.max_iters} (γ={effective_gamma:.1f}) ──────────────────────────────")

        # E-step
        labels, cost_matrix = e_step(
            blocks, structural_coeffs_map,
            slopes_2d, slp, slw,
            tangent_data, args.q_step, lagrange, decider_mode,
            pc.V, pc.A,
            gamma=effective_gamma,
            prev_labels=prev_labels,
            spatial_weights=spatial_weights
        )

        # Revival guard
        cluster_sizes = np.bincount(labels, minlength=len(slopes_2d))
        for k in range(1, len(slopes_2d)):
            if cluster_sizes[k] < min_cluster_size:
                print(f"  [Revival] Cluster {k} has {cluster_sizes[k]} blocks — triggering revival")
                gain_potential = cost_matrix[:, 0] - np.min(cost_matrix[:, 1:], axis=1)
                candidates = np.where(labels != k)[0]
                if len(candidates) > 0:
                    best_cand = candidates[np.argmax(gain_potential[candidates])]
                    labels[best_cand] = k

        # Metrics
        metrics = compute_metrics(
            blocks, labels, cost_matrix, pc.V, pc.A,
            slopes_2d, slp, slw, tangent_data, args.q_step, prev_labels,
            base_rate, structural_coeffs_map, struct_l1
        )
        metrics["iteration"] = iteration
        history.append(metrics)

        print(f"  Cost:           {metrics['total_cost']:>14.1f}  (structural: {metrics['structural_cost']:.1f})")
        print(f"  RD savings:     {metrics['rate_reduction_pct']:>10.2f}%  vs structural baseline")
        print(f"  H(X|X-1):       {metrics['entropy_markov']:>10.4f} bits  → Net Markov BPV: {metrics['net_bpv_markov']:+.5f} bpv")
        print(f"  L1 Norm Change: {metrics['avg_l1_norm']:>10.2f}  (baseline: {struct_l1:.2f})")
        print(f"  Switches:       {metrics['num_switches']:>10d} blocks")

        # Save SQLite iteration metrics
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO iterations (
            run_id, iteration, total_cost, structural_cost, rate_reduction_pct,
            entropy_markov, overhead_bpv_markov, net_bpv_markov, num_switches,
            avg_l1_norm, structural_l1_norm, labels_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, iteration, metrics["total_cost"], metrics["structural_cost"],
            metrics["rate_reduction_pct"], metrics["entropy_markov"], metrics["overhead_bpv_markov"],
            metrics["net_bpv_markov"], metrics["num_switches"], metrics["avg_l1_norm"],
            metrics["structural_l1_norm"], json.dumps(labels.tolist())
        ))
        
        for k in range(len(slopes_2d)):
            cursor.execute("""
            INSERT INTO cluster_states (
                run_id, iteration, cluster_id, cluster_size, slope_2d_1, slope_2d_2, slp, slw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, iteration, k, int(cluster_sizes[k]), float(slopes_2d[k,0]), float(slopes_2d[k,1]), float(slp[k]), float(slw[k])
            ))
        conn.commit()
        conn.close()

        # M-step
        if iteration < args.max_iters:
            freeze = args.freeze_slopes and (iteration >= args.gamma_warmup)
            slopes_2d, slp, slw = m_step(
                blocks, labels, slopes_2d, slp, slw,
                tangent_data, pc.V, pc.A,
                args.q_step, lagrange, decider_mode,
                min_cluster_size=min_cluster_size,
                freeze_slopes=freeze,
                sample_rate=args.sample_rate,
                centroid_learning_rate=args.centroid_lr
            )

        prev_labels = labels.copy()

    # Save final JSON file for compatibility/visualization
    save_data = {
        "config": {
            "block_size": args.block_size,
            "clusters": args.clusters,
            "q_step": args.q_step,
            "gamma": args.gamma,
            "max_iters": args.max_iters,
            "freeze_slopes": args.freeze_slopes,
            "gamma_warmup": args.gamma_warmup,
            "spatially_adaptive": args.spatially_adaptive
        },
        "final_slopes_2d": slopes_2d.tolist(),
        "final_slp":     slp.tolist(),
        "final_slw":     slw.tolist(),
        "iterations": [
            {
                "iteration": h["iteration"],
                "rate_reduction_pct": h["rate_reduction_pct"],
                "entropy_markov": h["entropy_markov"],
                "overhead_bpv_markov": h["overhead_bpv_markov"],
                "net_bpv_markov": h["net_bpv_markov"]
            } for h in history
        ]
    }
    
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"oracle_em_tangent_B{args.block_size}_C{args.clusters}.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[Run Finished] Run ID {run_id} successfully written to {db_path.absolute()}")

if __name__ == "__main__":
    main()
