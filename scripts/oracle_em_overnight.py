"""
Overnight Experiment Monitor for Oracle EM Codebook Training
============================================================
Features:
  1. Spatially-Adaptive Potts Penalty: scales gamma by spatial distance
     between consecutive Morton blocks to prevent boundary/Z-jump pollution.
  2. Full SQLite Database Logging: records every hyperparameter, iteration
     metric, cluster parameter, and assignment switches.
  3. Advanced Metrics: records switches, RD cost delta distributions, and cluster sizes.
  4. Centroid Freezing & Linear Gamma Ramping.
  5. Representation-Layer Metrics: tracks average L1 norm, energy compaction (top 25%),
     and sparsity ratio of the GFT coefficients, comparing them directly to the
     structural baseline.

Usage:
  python3 scripts/oracle_em_overnight.py \\
    --block_size 16 --clusters 6 --q_step 24 \\
    --gamma 1200.0 --gamma_warmup 3 --freeze_slopes --max_iters 6 \\
    --spatially_adaptive --db_path experiments/overnight_sweep.db
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
    
    # Schema safety check: if 'iterations' table exists but is missing the new column, drop all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='iterations'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(iterations)")
        columns = [col[1] for col in cursor.fetchall()]
        if "avg_energy_compaction" not in columns:
            print("[Database] Schema mismatch detected. Dropping old tables to initialize new schema...")
            cursor.execute("DROP TABLE IF EXISTS runs")
            cursor.execute("DROP TABLE IF EXISTS iterations")
            cursor.execute("DROP TABLE IF EXISTS cluster_states")
            conn.commit()

    # Run configuration table
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
        sample_rate REAL,
        spatially_adaptive INTEGER,
        max_iters INTEGER,
        min_iters INTEGER,
        dataset TEXT
    )
    """)
    
    # Iteration metrics table (includes advanced representation metrics)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS iterations (
        run_id INTEGER,
        iteration INTEGER,
        total_cost REAL,
        structural_cost REAL,
        rate_reduction_pct REAL,
        entropy_marginal REAL,
        entropy_markov REAL,
        overhead_bpv_marginal REAL,
        overhead_bpv_markov REAL,
        net_bpv_marginal REAL,
        net_bpv_markov REAL,
        active_clusters INTEGER,
        num_switches INTEGER,
        mean_delta_rd_second_best REAL,
        mean_delta_rd_worst_best REAL,
        avg_energy_compaction REAL,
        avg_l1_norm REAL,
        avg_sparsity_ratio REAL,
        structural_energy_compaction REAL,
        structural_l1_norm REAL,
        structural_sparsity_ratio REAL,
        labels_json TEXT,
        PRIMARY KEY (run_id, iteration)
    )
    """)
    
    # Cluster states per iteration
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cluster_states (
        run_id INTEGER,
        iteration INTEGER,
        cluster_id INTEGER,
        cluster_size INTEGER,
        slope_x REAL,
        slope_y REAL,
        slope_z REAL,
        slp REAL,
        slw REAL,
        cluster_cost REAL,
        PRIMARY KEY (run_id, iteration, cluster_id)
    )
    """)
    
    conn.commit()
    conn.close()


def save_run(db_path: Path, args, dataset_name: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO runs (
        timestamp, block_size, clusters, q_step, gamma, gamma_warmup, 
        freeze_slopes, sample_rate, spatially_adaptive, max_iters, min_iters, dataset
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        args.block_size,
        args.clusters,
        args.q_step,
        args.gamma,
        args.gamma_warmup,
        1 if args.freeze_slopes else 0,
        args.sample_rate,
        1 if args.spatially_adaptive else 0,
        args.max_iters,
        args.min_iters,
        dataset_name
    ))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_iteration(db_path: Path, run_id: int, metrics: dict):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO iterations (
        run_id, iteration, total_cost, structural_cost, rate_reduction_pct,
        entropy_marginal, entropy_markov, overhead_bpv_marginal, overhead_bpv_markov,
        net_bpv_marginal, net_bpv_markov, active_clusters, num_switches,
        mean_delta_rd_second_best, mean_delta_rd_worst_best,
        avg_energy_compaction, avg_l1_norm, avg_sparsity_ratio,
        structural_energy_compaction, structural_l1_norm, structural_sparsity_ratio,
        labels_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        metrics["iteration"],
        metrics["total_cost"],
        metrics["structural_cost"],
        metrics["rate_reduction_pct"],
        metrics["entropy_marginal"],
        metrics["entropy_markov"],
        metrics["overhead_bpv_marginal"],
        metrics["overhead_bpv_markov"],
        metrics["net_bpv_marginal"],
        metrics["net_bpv_markov"],
        metrics["active_clusters"],
        metrics["num_switches"],
        metrics["mean_delta_rd_second_best"],
        metrics["mean_delta_rd_worst_best"],
        metrics["avg_energy_compaction"],
        metrics["avg_l1_norm"],
        metrics["avg_sparsity_ratio"],
        metrics["structural_energy_compaction"],
        metrics["structural_l1_norm"],
        metrics["structural_sparsity_ratio"],
        json.dumps(metrics["labels"].tolist())
    ))
    conn.commit()
    conn.close()


def save_clusters(db_path: Path, run_id: int, iteration: int,
                  slopes: np.ndarray, slp: np.ndarray, slw: np.ndarray,
                  cluster_sizes: np.ndarray, cluster_costs: np.ndarray):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for k in range(len(slopes)):
        cursor.execute("""
        INSERT INTO cluster_states (
            run_id, iteration, cluster_id, cluster_size, 
            slope_x, slope_y, slope_z, slp, slw, cluster_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            iteration,
            k,
            int(cluster_sizes[k]),
            float(slopes[k, 0]),
            float(slopes[k, 1]),
            float(slopes[k, 2]),
            float(slp[k]),
            float(slw[k]),
            float(cluster_costs[k])
        ))
    conn.commit()
    conn.close()


# ===========================================================================
# SPATIAL CONTINUITY WEIGHTS
# ===========================================================================

def compute_spatial_weights(blocks: List, pc_V: np.ndarray, block_size: int) -> np.ndarray:
    print("\n[Spatial] Computing 3D centers and sequence transition weights...")
    centers = []
    for block in tqdm(blocks, desc="Block Centers"):
        idxs = block.metadata.return_index()
        centers.append(np.mean(pc_V[idxs, :], axis=0))
    centers = np.array(centers)
    
    dists = np.sqrt(np.sum((centers[1:] - centers[:-1]) ** 2, axis=1))
    mean_dist = np.mean(dists)
    sigma = 1.5 * mean_dist
    weights = np.exp(-dists / sigma)
    weights = np.concatenate([[1.0], weights])
    
    print(f"[Spatial] Mean 3D distance between consecutive blocks: {mean_dist:.2f} (block_size={block_size})")
    print(f"[Spatial] Weights range: [{weights.min():.4f}, {weights.max():.4f}]  Mean weight: {np.mean(weights):.4f}")
    return weights


# ===========================================================================
# INITIALIZATION
# ===========================================================================

def build_initial_codebook_from_pc(blocks: List, pc_V: np.ndarray, pc_A: np.ndarray,
                                   n_hybrid_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"\n[Init] Fitting luminance slopes on PCA-rotated coordinates...")
    approximator = Approximator()
    slopes_list = []
    rmse_list = []
    for block in tqdm(blocks, desc="Fitting luminance"):
        block.init_data(pc_V, pc_A)
        fit_result = approximator(block)
        slopes_list.append(fit_result.coeffs[1:])
        rmse_list.append(max(0.1, fit_result.rmse))
        block.clear_data()

    slopes_arr = np.array(slopes_list)
    rmse_arr = np.array(rmse_list)

    print(f"[Init] Clustering {len(slopes_arr)} PCA-rotated slopes into {n_hybrid_clusters} modes (weighted by RMSE)...")
    feat_min = slopes_arr.min(axis=0)
    feat_max = slopes_arr.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 1e-8, feat_max - feat_min, 1.0)
    X = (slopes_arr - feat_min) / feat_range

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
    slopes = np.vstack([np.zeros((1, 3)), hybrid_slopes])
    slp    = np.concatenate([[0.0], np.full(n_hybrid_clusters, 0.65)])
    slw    = np.concatenate([[1.0], np.full(n_hybrid_clusters, 1.4)])

    print("\n[Init] Initial codebook:")
    print(f"  {'Cluster':>8}  {'slope_x':>8} {'slope_y':>8} {'slope_z':>8}  {'slp':>6} {'slw':>6}")
    for k in range(len(slopes)):
        tag = "(structural)" if k == 0 else ""
        print(f"  {k:>8}  {slopes[k,0]:>8.3f} {slopes[k,1]:>8.3f} {slopes[k,2]:>8.3f}  "
              f"{slp[k]:>6.3f} {slw[k]:>6.3f}  {tag}")

    return slopes, slp, slw


# ===========================================================================
# E-STEP
# ===========================================================================

def _evaluate_block_costs_worker(block_id, Vblock, Ablock, metadata,
                                  slopes, slp_vals, slw_vals,
                                  structural_coeffs, q_step,
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
           spatial_weights: np.ndarray, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
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
            adj_costs[mask] += gamma * spatial_weights[i]
            new_labels[i] = np.argmin(adj_costs)

    return new_labels, cost_matrix


# ===========================================================================
# M-STEP
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
           freeze_slopes: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = len(slopes)
    new_slopes = slopes.copy()
    new_slp = slp.copy()
    new_slw = slw.copy()
    cluster_costs = np.zeros(K)

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

        # (b) Coarse search
        coarse_costs = _evaluate_grid_parallel(
            Vblocks, Ablocks, metadatas,
            new_slope, slp_coarse, slw_coarse,
            q_step, lagrange_proportional, decider_mode
        )
        best_pi, best_wi = np.unravel_index(np.argmin(coarse_costs), coarse_costs.shape)
        best_p_c = slp_coarse[best_pi]
        best_w_c = slw_coarse[best_wi]

        # (c) Fine refinement
        slp_fine = np.linspace(max(0.01, best_p_c - 0.12), min(0.99, best_p_c + 0.12), 5)
        slw_fine = np.linspace(max(0.1,  best_w_c - 0.8),  max(0.5, best_w_c + 0.8),  5)
        fine_costs = _evaluate_grid_parallel(
            Vblocks, Ablocks, metadatas,
            new_slope, slp_fine, slw_fine,
            q_step, lagrange_proportional, decider_mode
        )
        best_pf, best_wf = np.unravel_index(np.argmin(fine_costs), fine_costs.shape)

        new_slp[k] = slp_fine[best_pf]
        new_slw[k] = slw_fine[best_wf]
        cluster_costs[k] = float(np.min(fine_costs))
        print(f"         slp={new_slp[k]:.3f}  slw={new_slw[k]:.3f}")

    return new_slopes, new_slp, new_slw, cluster_costs


# ===========================================================================
# REPRESENTATION-LAYER METRICS WORKER
# ===========================================================================

def _evaluate_assigned_coeffs_worker(block_id, Vblock, Ablock, metadata,
                                     slope, slp, slw, k, structural_coeffs):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator
    from pcadc.blocks import Block

    if k == 0:
        coeffs = structural_coeffs
    else:
        gft_computer = GFTStrategyWraper()
        approximator = Approximator()
        block = Block(metadata)
        block.Vblock = Vblock
        block.Ablock = Ablock

        s_graph = StructuralGraph(metadata)
        s_graph.set_data(Vblock)
        a_graph = AttributeGraph(s_graph, slope, k, slp, slw)

        V_rot = approximator._spatial_norm(Vblock)
        A_app = Ablock.copy()
        A_app[:, 0] = V_rot @ slope.T
        a_graph.set_data(Vblock, A_app)

        _, coeffs = gft_computer(block, a_graph)

    # 1. Energy compaction (top 25%)
    energy_total = np.sum(coeffs**2)
    if energy_total == 0:
        compaction = 0.0
    else:
        coeffs_sorted = np.sort(coeffs**2)[::-1]
        top_k = max(1, coeffs_sorted.shape[0] // 4)
        compaction = float(np.sum(coeffs_sorted[:top_k]) / energy_total)

    # 2. L1 Norm
    l1 = float(np.sum(np.abs(coeffs)))

    # 3. Sparsity Ratio (coeffs below 1.0)
    sparsity = float(np.sum(np.abs(coeffs) < 1.0) / coeffs.size)

    return compaction, l1, sparsity


# ===========================================================================
# METRICS EVALUATION
# ===========================================================================

def compute_metrics(blocks: List, labels: np.ndarray, cost_matrix: np.ndarray,
                    vertices: np.ndarray, attributes: np.ndarray,
                    slopes: np.ndarray, slp: np.ndarray, slw: np.ndarray,
                    q_step: int, prev_labels: Optional[np.ndarray],
                    base_rate: float, structural_coeffs_map: dict,
                    struct_metrics: tuple) -> dict:
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

    saved_bpv = base_rate * (rate_reduction_pct / 100.0)
    net_bpv_marginal = saved_bpv - overhead_marginal
    net_bpv_markov = saved_bpv - overhead_markov

    num_switches = int(np.sum(labels != prev_labels)) if prev_labels is not None else 0

    sorted_costs = np.sort(cost_matrix, axis=1)
    mean_delta_second = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
    mean_delta_worst = float(np.mean(sorted_costs[:, -1] - sorted_costs[:, 0]))

    # Compute assigned representation metrics in parallel
    tasks = []
    for i, block in enumerate(blocks):
        block.init_data(vertices, attributes)
        k = labels[i]
        tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            slopes[k], slp[k], slw[k], k,
            structural_coeffs_map[block.block_id]
        ))
        block.clear_data()

    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_assigned_coeffs_worker)(*t)
        for t in tasks
    )

    avg_compaction = float(np.mean([r[0] for r in results]))
    avg_l1 = float(np.mean([r[1] for r in results]))
    avg_sparsity = float(np.mean([r[2] for r in results]))

    struct_compaction, struct_l1, struct_sparsity = struct_metrics

    return {
        "total_cost":          total_cost,
        "structural_cost":     structural_cost,
        "rate_reduction_pct":  rate_reduction_pct,
        "entropy_marginal":    entropy,
        "entropy_markov":      h_cond,
        "overhead_bpv_marginal": overhead_marginal,
        "overhead_bpv_markov":   overhead_markov,
        "net_bpv_marginal":    net_bpv_marginal,
        "net_bpv_markov":      net_bpv_markov,
        "active_clusters":     active,
        "cluster_sizes":       cluster_sizes,
        "num_switches":        num_switches,
        "mean_delta_rd_second_best": mean_delta_second,
        "mean_delta_rd_worst_best":  mean_delta_worst,
        "avg_energy_compaction": avg_compaction,
        "avg_l1_norm":          avg_l1,
        "avg_sparsity_ratio":    avg_sparsity,
        "structural_energy_compaction": struct_compaction,
        "structural_l1_norm":          struct_l1,
        "structural_sparsity_ratio":    struct_sparsity,
        "labels":              labels.copy()
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = ArgumentParser(description="Advanced Oracle EM Overnight Monitor")
    parser.add_argument("--block_size",  type=int,   required=True)
    parser.add_argument("--clusters",    type=int,   required=True)
    parser.add_argument("--q_step",      type=int,   required=True)
    parser.add_argument("--gamma",       type=float, default=0.0)
    parser.add_argument("--max_iters",   type=int,   default=8)
    parser.add_argument("--min_iters",   type=int,   default=4)
    parser.add_argument("--gamma_warmup",type=int,   default=3)
    parser.add_argument("--freeze_slopes",action="store_true")
    parser.add_argument("--spatially_adaptive", action="store_true")
    parser.add_argument("--db_path",     type=str,   default="experiments/overnight_sweep.db")
    parser.add_argument("--config",      type=str,   default="config/base_config.yaml")
    parser.add_argument("--sample_rate", type=float, default=1.0)
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
    dataset_name = pc.metadata.source_id
    base_rate = 0.1603 if args.block_size == 8 else 0.1480

    print(f"\n[Overnight] Dataset: {dataset_name} | Block size: {args.block_size}")
    print(f"[Overnight] SQLite log database: {db_path.absolute()}")

    if args.spatially_adaptive:
        spatial_weights = compute_spatial_weights(blocks, pc.V, args.block_size)
    else:
        spatial_weights = np.ones(len(blocks))

    # Precompute structural GFT coeffs
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

    # Precompute baseline structural metrics
    print("\n[Precompute] Baseline structural representation metrics...")
    struct_tasks = []
    for block in blocks:
        block.init_data(pc.V, pc.A)
        struct_tasks.append((
            block.block_id, block.Vblock.copy(), block.Ablock.copy(), block.metadata,
            np.zeros(3), 0.0, 1.0, 0,
            structural_coeffs_map[block.block_id]
        ))
        block.clear_data()

    struct_results = Parallel(n_jobs=-1)(
        delayed(_evaluate_assigned_coeffs_worker)(*t)
        for t in struct_tasks
    )
    struct_compaction = float(np.mean([r[0] for r in struct_results]))
    struct_l1 = float(np.mean([r[1] for r in struct_results]))
    struct_sparsity = float(np.mean([r[2] for r in struct_results]))
    struct_metrics = (struct_compaction, struct_l1, struct_sparsity)
    
    print(f"  Structural L1 Norm:       {struct_l1:.2f}")
    print(f"  Structural Compaction:    {struct_compaction:.4f}")
    print(f"  Structural Sparsity Ratio:{struct_sparsity:.4f}")

    # PCA-rotated K-Means init
    slopes, slp, slw = build_initial_codebook_from_pc(blocks, pc.V, pc.A, n_hybrid)

    # Save run info
    run_id = save_run(db_path, args, dataset_name)
    print(f"[Database] Registered Run ID: {run_id}")

    history = []
    prev_labels = None
    decider_mode = params.sequential_params.decider_mode
    lagrange = params.sequential_params.lagrange_proportional
    min_cluster_size = max(5, int(0.01 * len(blocks)))

    cluster_costs = np.zeros(args.clusters)

    for iteration in range(1, args.max_iters + 1):
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
            slopes, slp, slw,
            args.q_step, lagrange, decider_mode,
            pc.V, pc.A,
            gamma=effective_gamma,
            prev_labels=prev_labels,
            spatial_weights=spatial_weights,
            iteration=iteration
        )

        # Revival guard
        cluster_sizes = np.bincount(labels, minlength=len(slopes))
        for k in range(1, len(slopes)):
            if cluster_sizes[k] < min_cluster_size:
                print(f"  [Revival] Cluster {k} has {cluster_sizes[k]} blocks — triggering revival")
                gain_potential = cost_matrix[:, 0] - np.min(cost_matrix[:, 1:], axis=1)
                candidates = np.where(labels != k)[0]
                if len(candidates) > 0:
                    best_cand = candidates[np.argmax(gain_potential[candidates])]
                    labels[best_cand] = k

        # Metrics (includes parallel representation pass)
        metrics = compute_metrics(
            blocks, labels, cost_matrix, pc.V, pc.A,
            slopes, slp, slw, args.q_step, prev_labels,
            base_rate, structural_coeffs_map, struct_metrics
        )
        metrics["iteration"] = iteration
        history.append(metrics)

        print(f"  Cost:           {metrics['total_cost']:>14.1f}  (structural: {metrics['structural_cost']:.1f})")
        print(f"  RD savings:     {metrics['rate_reduction_pct']:>10.2f}%  vs structural baseline")
        print(f"  H(X|X-1):       {metrics['entropy_markov']:>10.4f} bits  → Net Markov BPV: {metrics['net_bpv_markov']:+.5f} bpv")
        print(f"  L1 Norm Change: {metrics['avg_l1_norm']:>10.2f}  (baseline: {struct_l1:.2f})")
        print(f"  Compaction:     {metrics['avg_energy_compaction']:>10.4f}  (baseline: {struct_compaction:.4f})")
        print(f"  Sparsity Ratio: {metrics['avg_sparsity_ratio']:>10.4f}  (baseline: {struct_sparsity:.4f})")
        print(f"  Switches:       {metrics['num_switches']:>10d} blocks")

        # Save SQLite iteration metrics
        save_iteration(db_path, run_id, metrics)
        save_clusters(db_path, run_id, iteration, slopes, slp, slw, cluster_sizes, cluster_costs)

        # M-step
        if iteration < args.max_iters:
            freeze = args.freeze_slopes and (iteration >= args.gamma_warmup)
            slopes, slp, slw, cluster_costs = m_step(
                blocks, labels, slopes, slp, slw,
                pc.V, pc.A,
                args.q_step, lagrange, decider_mode,
                min_cluster_size=min_cluster_size,
                sample_rate=args.sample_rate,
                freeze_slopes=freeze
            )

        prev_labels = labels.copy()

    # Save final JSON file as well for compatibility
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
            "spatially_adaptive": args.spatially_adaptive
        },
        "final_slopes":  slopes.tolist(),
        "final_slp":     slp.tolist(),
        "final_slw":     slw.tolist(),
        "iterations": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items() if k != "labels"}
            for h in history
        ],
    }
    out_dir = Path("experiments")
    out_file = out_dir / f"oracle_em_overnight_B{args.block_size}_C{args.clusters}.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Saved JSON] {out_file}")
    print(f"[Run Finished] Run ID {run_id} successfully written to {db_path.absolute()}")

if __name__ == "__main__":
    main()
