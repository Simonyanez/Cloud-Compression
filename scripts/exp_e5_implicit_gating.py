"""
Experiment: Implicit Fallback Gating & Edge-Focused Centroid Training
======================================================================
Two-part architectural update to eliminate side-information tax in smooth
regions and focus GFT cluster capacity on high-contrast directional edges.

TABLES WRITTEN TO DATABASE:
  1. gating_results         — Per (child_b, q_step, tau_gate) aggregate RD metrics
  2. cluster_centroids       — Trained 3D unit centroid vectors + spherical angles per K
  3. cluster_slopes          — Optimized (slp, slw) per cluster per child block scale
  4. gradient_distribution   — Per-scale gradient energy distribution statistics
  5. parent_block_telemetry  — Per-parent-block variance, active flag, chosen mode, GFT gain
  6. mode_activation_stats   — Per (child_b, q_step, tau_gate, mode_k) activation frequency

Usage:
    python3 scripts/exp_e5_implicit_gating.py [--child-block-sizes 16,8,4]
                                               [--tau-gates 0.0,50.0,150.0,300.0,500.0,800.0,1200.0,1600.0]
                                               [--tau-train-pct 75]
                                               [--q-steps 24,32,48,64]
                                               [--K 10]
                                               [--gamma 1000]
                                               [--db-path results/exp_e5_gating.db]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
logging.disable(logging.CRITICAL)

import sys, time, argparse, sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph
from pcadc.transforms import GFTStrategyWraper
from oracle_em_tangent import extract_block_tangent_planes

from exp_e4_master_benchmark_suite import (
    fit_gradient_3d, spherical_kmeans, build_parent_child_map,
    grid_search, compute_all_raw_coeffs_hierarchical,
    evaluate_hierarchical_decider_on_coeffs, run_viterbi, compute_entropy
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def section(title):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Database — init all 6 tables
# ---------------------------------------------------------------------------

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Table 1: Aggregate RD metrics per (child_b, q_step, tau_gate)
    c.execute("""
    CREATE TABLE IF NOT EXISTS gating_results (
        run_id                    TEXT PRIMARY KEY,
        child_b                   INTEGER,
        q_step                    INTEGER,
        tau_gate                  REAL,
        tau_train_pct             REAL,
        K                         INTEGER,
        gamma                     REAL,
        n_parent_blocks           INTEGER,
        n_child_blocks            INTEGER,
        n_active_parents          INTEGER,
        n_gated_parents           INTEGER,
        gating_ratio_pct          REAL,
        effective_oh_bpv          REAL,
        entropy_active_bits       REAL,
        active_edge_compaction_pct REAL,
        gated_block_compaction_pct REAL,
        baseline_bpv              REAL,
        baseline_psnr_db          REAL,
        hier_transform_bpv        REAL,
        net_bpv                   REAL,
        delta_bpv                 REAL,
        delta_pct                 REAL,
        psnr_db                   REAL,
        psnr_delta_db             REAL,
        mse                       REAL,
        n_viterbi_switches        INTEGER,
        dominant_mode_k           INTEGER,
        dominant_mode_pct         REAL,
        timestamp                 TEXT
    )""")

    # Table 2: Trained 3D cluster centroids + spherical geometry
    c.execute("""
    CREATE TABLE IF NOT EXISTS cluster_centroids (
        run_prefix      TEXT,
        tau_train_pct   REAL,
        K               INTEGER,
        mode_k          INTEGER,
        centroid_x      REAL,
        centroid_y      REAL,
        centroid_z      REAL,
        azimuth_deg     REAL,
        elevation_deg   REAL,
        n_training_blocks INTEGER,
        gradient_energy_threshold REAL,
        PRIMARY KEY (run_prefix, mode_k)
    )""")

    # Table 3: Optimized slope params per cluster per child scale
    c.execute("""
    CREATE TABLE IF NOT EXISTS cluster_slopes (
        run_prefix   TEXT,
        child_b      INTEGER,
        mode_k       INTEGER,
        slp          REAL,
        slw          REAL,
        PRIMARY KEY (run_prefix, child_b, mode_k)
    )""")

    # Table 4: Gradient energy distribution per scale
    c.execute("""
    CREATE TABLE IF NOT EXISTS gradient_distribution (
        run_prefix      TEXT,
        block_scale     INTEGER,
        role            TEXT,
        n_blocks        INTEGER,
        grad_mean       REAL,
        grad_std        REAL,
        grad_min        REAL,
        grad_p25        REAL,
        grad_p50        REAL,
        grad_p75        REAL,
        grad_p90        REAL,
        grad_max        REAL,
        n_edge_blocks   INTEGER,
        edge_pct        REAL,
        PRIMARY KEY (run_prefix, block_scale, role)
    )""")

    # Table 5: Per-parent-block telemetry for each (child_b, q_step, tau_gate)
    c.execute("""
    CREATE TABLE IF NOT EXISTS parent_block_telemetry (
        run_id          TEXT,
        parent_idx      INTEGER,
        variance        REAL,
        gradient_energy REAL,
        is_active       INTEGER,
        chosen_mode_k   INTEGER,
        struct_rate     REAL,
        chosen_rate     REAL,
        rate_delta      REAL,
        rate_savings_pct REAL,
        distortion      REAL,
        n_children      INTEGER,
        PRIMARY KEY (run_id, parent_idx)
    )""")

    # Table 6: Per-mode activation frequency per run
    c.execute("""
    CREATE TABLE IF NOT EXISTS mode_activation_stats (
        run_id          TEXT,
        mode_k          INTEGER,
        n_activations   INTEGER,
        activation_pct  REAL,
        mean_rate_savings_pct REAL,
        mean_distortion REAL,
        PRIMARY KEY (run_id, mode_k)
    )""")

    conn.commit()
    conn.close()


def db_conn(db_path):
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------------------
# Phase 1: Edge-Focused Centroid Training
# ---------------------------------------------------------------------------

def edge_focused_centroids(g3d_norm, g3d_raw, K_hybrid, tau_train_pct=75.0):
    """
    Filter gradient directions by gradient energy magnitude (g3d_raw) at the
    tau_train_pct percentile, then run spherical k-means on the high-energy
    subset. Forces all K clusters to specialize in sharp directional edges.

    Returns:
        centroids: (K_hybrid, 3) edge-focused cluster centroids
        edge_mask: (N,) boolean mask of blocks used in training
        threshold: float gradient energy threshold applied
    """
    if tau_train_pct > 0.0:
        threshold = float(np.percentile(g3d_raw, tau_train_pct))
        edge_mask = g3d_raw >= threshold
    else:
        threshold = 0.0
        edge_mask = np.ones(len(g3d_norm), dtype=bool)

    n_edge = int(edge_mask.sum())
    log(f"  Edge training filter: {n_edge}/{len(g3d_norm)} blocks "
        f"({100*n_edge/len(g3d_norm):.1f}%) retained at "
        f"tau_train={tau_train_pct:.0f}th percentile  (threshold={threshold:.4f})")

    if n_edge < K_hybrid * 3:
        log(f"  Warning: Too few edge blocks ({n_edge}) for K={K_hybrid+1}. Using all blocks.")
        edge_mask = np.ones(len(g3d_norm), dtype=bool)
        threshold = 0.0

    centroids, _ = spherical_kmeans(g3d_norm[edge_mask], K_hybrid)
    return centroids, edge_mask, threshold


# ---------------------------------------------------------------------------
# Phase 2: Implicit Parent-Derived Fallback Gate
# ---------------------------------------------------------------------------

def compute_parent_variances(blocks_p, pc_V, pc_A):
    """
    Compute spatial variance of the luminance attribute channel for each
    parent block. Zero side-information — decoder computes identically.
    """
    variances = np.zeros(len(blocks_p))
    for pi, pb in enumerate(blocks_p):
        pb.init_data(pc_V, pc_A)
        lum = pb.Ablock[:, 0]
        variances[pi] = float(np.var(lum)) if len(lum) > 1 else 0.0
        pb.clear_data()
    return variances


def apply_implicit_gate(cost_mat, parent_variances, tau_gate, Np, K):
    """
    Force flat parent blocks (variance < tau_gate) to k=0, zero overhead.
    Returns gated cost matrix and active/gated boolean mask.
    """
    active_mask = parent_variances >= tau_gate
    gated_cost_mat = cost_mat.copy()
    for pi in range(Np):
        if not active_mask[pi]:
            gated_cost_mat[pi, 1:] = np.inf
    return gated_cost_mat, active_mask


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_centroids(db_path, run_prefix, tau_train_pct, K, centroids,
                   n_training_blocks, threshold):
    conn = db_conn(db_path)
    c = conn.cursor()
    for idx, cent in enumerate(centroids):
        x, y, z = float(cent[0]), float(cent[1]), float(cent[2])
        az  = float(np.degrees(np.arctan2(y, x)))
        el  = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
        c.execute("""INSERT OR REPLACE INTO cluster_centroids VALUES
            (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_prefix, tau_train_pct, K, idx + 1,
             x, y, z, az, el, n_training_blocks, threshold))
    conn.commit(); conn.close()


def save_slopes(db_path, run_prefix, child_b, K, slp, slw):
    conn = db_conn(db_path)
    c = conn.cursor()
    for k in range(K):
        c.execute("""INSERT OR REPLACE INTO cluster_slopes VALUES (?,?,?,?,?)""",
                  (run_prefix, child_b, k, float(slp[k]), float(slw[k])))
    conn.commit(); conn.close()


def save_gradient_distribution(db_path, run_prefix, block_scale, role,
                               g3d_raw, n_edge, edge_pct):
    conn = db_conn(db_path)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO gradient_distribution VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        run_prefix, block_scale, role,
        len(g3d_raw),
        float(np.mean(g3d_raw)), float(np.std(g3d_raw)),
        float(np.min(g3d_raw)),
        float(np.percentile(g3d_raw, 25)),
        float(np.percentile(g3d_raw, 50)),
        float(np.percentile(g3d_raw, 75)),
        float(np.percentile(g3d_raw, 90)),
        float(np.max(g3d_raw)),
        n_edge, edge_pct
    ))
    conn.commit(); conn.close()


def save_gating_result(db_path, rec):
    conn = db_conn(db_path)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO gating_results VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        rec["run_id"], rec["child_b"], rec["q_step"],
        rec["tau_gate"], rec["tau_train_pct"], rec["K"], rec["gamma"],
        rec["n_parent_blocks"], rec["n_child_blocks"],
        rec["n_active_parents"], rec["n_gated_parents"],
        rec["gating_ratio_pct"], rec["effective_oh_bpv"],
        rec["entropy_active_bits"], rec["active_edge_compaction_pct"],
        rec["gated_block_compaction_pct"],
        rec["baseline_bpv"], rec["baseline_psnr_db"],
        rec["hier_transform_bpv"], rec["net_bpv"],
        rec["delta_bpv"], rec["delta_pct"],
        rec["psnr_db"], rec["psnr_delta_db"], rec["mse"],
        rec["n_viterbi_switches"],
        rec["dominant_mode_k"], rec["dominant_mode_pct"],
        rec["timestamp"]
    ))
    conn.commit(); conn.close()


def save_parent_block_telemetry(db_path, run_id, parent_telemetry_rows):
    conn = db_conn(db_path)
    c = conn.cursor()
    c.executemany("""INSERT OR REPLACE INTO parent_block_telemetry VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?)""", parent_telemetry_rows)
    conn.commit(); conn.close()


def save_mode_activation_stats(db_path, run_id, vl, rate_mat, dist_mat, K, Np):
    conn = db_conn(db_path)
    c = conn.cursor()
    for k in range(K):
        idxs = [i for i in range(Np) if vl[i] == k]
        n_act = len(idxs)
        act_pct = 100.0 * n_act / Np
        if n_act > 0 and k > 0:
            savings = [(rate_mat[i, 0] - rate_mat[i, k]) /
                       (rate_mat[i, 0] + 1e-12) * 100.0
                       for i in idxs if not np.isinf(rate_mat[i, k])]
            dists = [dist_mat[i, k] for i in idxs if not np.isinf(dist_mat[i, k])]
            mean_sav  = float(np.mean(savings)) if savings else 0.0
            mean_dist = float(np.mean(dists))   if dists   else 0.0
        else:
            mean_sav  = 0.0
            mean_dist = float(np.mean([dist_mat[i, 0] for i in idxs])) if idxs else 0.0
        c.execute("""INSERT OR REPLACE INTO mode_activation_stats VALUES
            (?,?,?,?,?,?)""",
            (run_id, k, n_act, act_pct, mean_sav, mean_dist))
    conn.commit(); conn.close()


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(db_path, child_block_sizes, tau_gates, tau_train_pct,
                   q_steps, K=10, gamma=1000.0):

    init_db(db_path)

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(
        project_root / params.sequential_params.point_cloud_path,
        "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm      = params.sequential_params.decider_mode
    lag     = params.sequential_params.lagrange_proportional
    total_v = pc.V.shape[0]

    run_prefix = f"e5_tt{tau_train_pct:.0f}_K{K}"

    log(f"Point cloud: {total_v} voxels  |  decider_mode={dm}  |  lag={lag}")
    log(f"K={K}  |  tau_train_pct={tau_train_pct}%  |  gamma={gamma}")

    K_hybrid = K - 1

    # ------------------------------------------------------------------
    # Pre-partition parent blocks (B=32)
    # ------------------------------------------------------------------
    section("PRE-PARTITIONING PARENT BLOCKS (B=32)")
    Bp = 32
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=Bp)
    Np = len(blocks_p)
    log(f"Parent blocks (B={Bp}): {Np}")

    tangents_p = extract_block_tangent_planes(blocks_p, pc.V, pc.A)
    g3d_norm_p, g3d_raw_p, normals_p = [], [], []
    for i, b in enumerate(blocks_p):
        b.init_data(pc.V, pc.A)
        normals_p.append(tangents_p[i]["v3"])
        g, rn = fit_gradient_3d(b.Vblock, b.Ablock,
                                 tangents_p[i]["v1"], tangents_p[i]["v2"])
        g3d_norm_p.append(g); g3d_raw_p.append(rn)
        b.clear_data()
    g3d_norm_p = np.array(g3d_norm_p)
    g3d_raw_p  = np.array(g3d_raw_p)
    normals_p  = np.array(normals_p)

    # Continuity weights for Viterbi
    nw_p = np.ones(Np)
    for i in range(1, Np):
        nw_p[i] = abs(np.dot(normals_p[i], normals_p[i-1]))

    # Save parent gradient distribution
    edge_thr_p = float(np.percentile(g3d_raw_p, tau_train_pct)) if tau_train_pct > 0 else 0.0
    n_edge_p   = int((g3d_raw_p >= edge_thr_p).sum())
    save_gradient_distribution(db_path, run_prefix, Bp, "parent",
                               g3d_raw_p, n_edge_p, 100.0 * n_edge_p / Np)

    # ------------------------------------------------------------------
    # Phase 2 prep: parent variance for implicit gate
    # ------------------------------------------------------------------
    section("COMPUTING PARENT BLOCK VARIANCES (Implicit Gate Metric)")
    parent_variances = compute_parent_variances(blocks_p, pc.V, pc.A)
    log(f"Variance distribution: min={parent_variances.min():.2f}  "
        f"p25={np.percentile(parent_variances,25):.2f}  "
        f"median={np.median(parent_variances):.2f}  "
        f"p75={np.percentile(parent_variances,75):.2f}  "
        f"max={parent_variances.max():.2f}")

    # ------------------------------------------------------------------
    # Phase 1: Edge-focused centroids
    # ------------------------------------------------------------------
    section(f"PHASE 1: EDGE-FOCUSED CENTROID TRAINING (tau_train={tau_train_pct}%)")
    centroids_p, edge_mask_p, grad_threshold = edge_focused_centroids(
        g3d_norm_p, g3d_raw_p, K_hybrid, tau_train_pct)
    n_training = int(edge_mask_p.sum())
    log(f"Centroids computed from {n_training} blocks (threshold={grad_threshold:.4f})")

    # Save centroid geometry (Table 2)
    save_centroids(db_path, run_prefix, tau_train_pct, K,
                   centroids_p, n_training, grad_threshold)

    # ------------------------------------------------------------------
    # Sweep child block sizes
    # ------------------------------------------------------------------
    for Bc in child_block_sizes:
        section(f"CHILD BLOCK SIZE B={Bc}")
        _, blocks_c = MortonBlockPartition().partition(pc, bsize=Bc)
        Nc = len(blocks_c)
        log(f"Child blocks (B={Bc}): {Nc}  |  ~{Nc/Np:.1f} per parent")

        tangents_c = extract_block_tangent_planes(blocks_c, pc.V, pc.A)
        g3d_norm_c, g3d_raw_c = [], []
        for i, b in enumerate(blocks_c):
            b.init_data(pc.V, pc.A)
            g, rn = fit_gradient_3d(b.Vblock, b.Ablock,
                                     tangents_c[i]["v1"], tangents_c[i]["v2"])
            g3d_norm_c.append(g); g3d_raw_c.append(rn)
            b.clear_data()
        g3d_norm_c = np.array(g3d_norm_c)
        g3d_raw_c  = np.array(g3d_raw_c)

        # Save child gradient distribution (Table 4)
        edge_thr_c = float(np.percentile(g3d_raw_c, tau_train_pct)) if tau_train_pct > 0 else 0.0
        n_edge_c   = int((g3d_raw_c >= edge_thr_c).sum())
        save_gradient_distribution(db_path, run_prefix, Bc, "child",
                                   g3d_raw_c, n_edge_c, 100.0 * n_edge_c / Nc)

        # Structural baseline coefficients for child blocks
        gft_comp = GFTStrategyWraper()
        struct_map_c = {}
        for ci, cb in enumerate(blocks_c):
            cb.init_data(pc.V, pc.A)
            sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
            _, coeffs = gft_comp(cb, sg)
            struct_map_c[ci] = coeffs
            cb.clear_data()

        parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc.V, Bp)

        # ------------------------------------------------------------------
        # Baseline rates
        # ------------------------------------------------------------------
        base_res = {}
        for q in q_steps:
            dec = Decider(dm, lag); dec._set_vars(q)
            tot_rate, tot_dist = 0.0, 0.0
            for i in range(Nc):
                _, r, d = dec._RDcost(struct_map_c[i])
                tot_rate += r; tot_dist += d
            bpv  = tot_rate / total_v
            mse  = tot_dist / total_v
            psnr = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
            base_res[q] = (bpv, psnr, mse)
            log(f"  Baseline B={Bc} Q={q}: {bpv:.5f} BPV  |  {psnr:.2f} dB")

        # ------------------------------------------------------------------
        # Grid search slope params
        # ------------------------------------------------------------------
        log("Running grid search for slope parameters...")
        child_edge_mask = g3d_raw_c >= np.percentile(g3d_raw_c, tau_train_pct)
        slp_opt, slw_opt = grid_search(
            blocks_c, tangents_c, centroids_p, g3d_raw_c, g3d_norm_c,
            pc.V, pc.A, dm, lag, 24, K, child_edge_mask)

        # Save slope params (Table 3)
        save_slopes(db_path, run_prefix, Bc, K, slp_opt, slw_opt)
        log(f"  Slopes saved. slp[1]={slp_opt[1]:.3f}  slw[1]={slw_opt[1]:.3f}")

        # ------------------------------------------------------------------
        # Compute hierarchical coefficients
        # ------------------------------------------------------------------
        log("Computing hierarchical GFT coefficients (parallel)...")
        p_coeffs = compute_all_raw_coeffs_hierarchical(
            blocks_p, blocks_c, parent_to_children,
            tangents_p, tangents_c, struct_map_c,
            centroids_p, slp_opt, slw_opt,
            g3d_norm_p, g3d_raw_p, pc.V, pc.A, K)

        # ------------------------------------------------------------------
        # Evaluate across Q steps and tau_gate values
        # ------------------------------------------------------------------
        print(f"\n{'Q':>4} | {'tau_g':>7} | {'Gate%':>6} | {'OH_bpv':>9} | "
              f"{'EdgeCmp%':>8} | {'Base_bpv':>9} | {'Net_bpv':>9} | "
              f"{'Delta_bpv':>10} | {'PSNR_db':>7} | Status")
        print("-" * 100)

        for q in q_steps:
            dec = Decider(dm, lag); dec._set_vars(q)
            cost_mat, rate_mat, dist_mat = evaluate_hierarchical_decider_on_coeffs(
                p_coeffs, K, dec)

            b_bpv, b_psnr, b_mse = base_res[q]

            for tau_gate in tau_gates:
                run_id = f"e5_B{Bc}_q{q}_tg{tau_gate:.1f}_tt{tau_train_pct:.0f}_K{K}"

                # Apply implicit gate
                gated_cost_mat, active_mask = apply_implicit_gate(
                    cost_mat, parent_variances, tau_gate, Np, K)

                # Viterbi
                vl = run_viterbi(gated_cost_mat, K, gamma, nw_p)

                n_active  = int(active_mask.sum())
                n_gated   = Np - n_active
                gate_pct  = 100.0 * n_gated / Np

                # Rate/distortion
                transform_rate_tot = sum(rate_mat[i, vl[i]] for i in range(Np))
                transform_bpv      = transform_rate_tot / total_v

                # Overhead — only active blocks pay signaling
                if n_active > 0:
                    active_labels = np.array([vl[i] for i in range(Np) if active_mask[i]])
                    H_active      = compute_entropy(active_labels, K)
                    oh_bpv        = H_active * n_active / total_v
                else:
                    H_active, oh_bpv = 0.0, 0.0

                net_bpv   = transform_bpv + oh_bpv
                delta_bpv = net_bpv - b_bpv
                delta_pct = delta_bpv / (b_bpv + 1e-12) * 100.0

                tot_dist  = sum(dist_mat[i, vl[i]] for i in range(Np))
                mse       = tot_dist / total_v
                psnr      = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
                psnr_delta = psnr - b_psnr

                # Active edge compaction (non-gated blocks only)
                if n_active > 0:
                    sr_act = sum(rate_mat[i, 0]    for i in range(Np) if active_mask[i])
                    cr_act = sum(rate_mat[i, vl[i]] for i in range(Np) if active_mask[i])
                    edge_cmp_pct = (sr_act - cr_act) / (sr_act + 1e-12) * 100.0
                else:
                    edge_cmp_pct = 0.0

                # Gated block compaction (forced to k=0; baseline already == 0)
                gated_cmp_pct = 0.0  # By definition: gated blocks use baseline

                # Viterbi mode switches (continuity metric)
                vl_arr = np.array(vl)
                n_switches = int((vl_arr[1:] != vl_arr[:-1]).sum())

                # Dominant mode
                mode_counts  = np.bincount(vl_arr, minlength=K)
                dominant_k   = int(np.argmax(mode_counts))
                dominant_pct = float(mode_counts[dominant_k] / Np * 100.0)

                status = "NetGain" if delta_bpv < 0 else "Penalty"

                # ------ Save Table 1: gating_results ------
                save_gating_result(db_path, {
                    "run_id": run_id, "child_b": Bc, "q_step": q,
                    "tau_gate": tau_gate, "tau_train_pct": tau_train_pct,
                    "K": K, "gamma": gamma,
                    "n_parent_blocks": Np, "n_child_blocks": Nc,
                    "n_active_parents": n_active, "n_gated_parents": n_gated,
                    "gating_ratio_pct": gate_pct,
                    "effective_oh_bpv": oh_bpv,
                    "entropy_active_bits": float(H_active),
                    "active_edge_compaction_pct": edge_cmp_pct,
                    "gated_block_compaction_pct": gated_cmp_pct,
                    "baseline_bpv": b_bpv, "baseline_psnr_db": b_psnr,
                    "hier_transform_bpv": transform_bpv,
                    "net_bpv": net_bpv, "delta_bpv": delta_bpv,
                    "delta_pct": delta_pct,
                    "psnr_db": psnr, "psnr_delta_db": psnr_delta,
                    "mse": mse,
                    "n_viterbi_switches": n_switches,
                    "dominant_mode_k": dominant_k,
                    "dominant_mode_pct": dominant_pct,
                    "timestamp": datetime.now().isoformat()
                })

                # ------ Save Table 5: parent_block_telemetry ------
                telem_rows = []
                for pi in range(Np):
                    n_ch   = len(parent_to_children.get(pi, []))
                    s_rate = float(rate_mat[pi, 0])   if not np.isinf(rate_mat[pi, 0])   else None
                    c_rate = float(rate_mat[pi, vl[pi]]) if not np.isinf(rate_mat[pi, vl[pi]]) else None
                    rdelta = (s_rate - c_rate) if (s_rate is not None and c_rate is not None) else None
                    rsav   = (rdelta / (s_rate + 1e-12) * 100.0) if rdelta is not None else None
                    dist   = float(dist_mat[pi, vl[pi]]) if not np.isinf(dist_mat[pi, vl[pi]]) else None
                    telem_rows.append((
                        run_id, pi,
                        float(parent_variances[pi]),
                        float(g3d_raw_p[pi]),
                        int(active_mask[pi]),
                        int(vl[pi]),
                        s_rate, c_rate, rdelta, rsav, dist, n_ch
                    ))
                save_parent_block_telemetry(db_path, run_id, telem_rows)

                # ------ Save Table 6: mode_activation_stats ------
                save_mode_activation_stats(db_path, run_id, vl, rate_mat, dist_mat, K, Np)

                print(f"{q:>4} | {tau_gate:>7.1f} | {gate_pct:>5.1f}% | "
                      f"{oh_bpv:>9.6f} | {edge_cmp_pct:>7.2f}% | "
                      f"{b_bpv:>9.5f} | {net_bpv:>9.5f} | "
                      f"{delta_bpv:>+10.5f} | {psnr:>7.2f} | {status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("EXPERIMENT SUMMARY: BEST CONFIG PER SCALE & Q STEP")
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    print(f"\n{'B':>4} | {'Q':>4} | {'tau_g':>7} | {'Gate%':>6} | "
          f"{'OH_bpv':>9} | {'EdgeCmp%':>8} | {'Base_bpv':>9} | "
          f"{'Net_bpv':>9} | {'Delta_bpv':>10} | {'PSNR':>7} | Status")
    print("-" * 100)
    for Bc in child_block_sizes:
        for q in q_steps:
            cur.execute("""
            SELECT tau_gate, gating_ratio_pct, effective_oh_bpv,
                   active_edge_compaction_pct, baseline_bpv, net_bpv,
                   delta_bpv, psnr_db
            FROM gating_results WHERE child_b=? AND q_step=?
            ORDER BY net_bpv ASC LIMIT 1""", (Bc, q))
            row = cur.fetchone()
            if row:
                tg, gr, oh, ec, bb, nb, db_, ps = row
                s = "NetGain" if db_ < 0 else "Penalty"
                print(f"{Bc:>4} | {q:>4} | {tg:>7.1f} | {gr:>5.1f}% | "
                      f"{oh:>9.6f} | {ec:>7.2f}% | {bb:>9.5f} | "
                      f"{nb:>9.5f} | {db_:>+10.5f} | {ps:>7.2f} | {s}")
    conn.close()
    log(f"All results saved to: {db_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp E5: Implicit Fallback Gating & Edge-Focused Centroid Training")
    p.add_argument("--child-block-sizes", type=str, default="16,8,4")
    p.add_argument("--tau-gates", type=str,
                   default="0.0,50.0,150.0,300.0,500.0,800.0,1200.0,1600.0")
    p.add_argument("--tau-train-pct", type=float, default=75.0)
    p.add_argument("--q-steps", type=str, default="24,32,48,64")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1000.0)
    p.add_argument("--db-path", type=str, default="results/exp_e5_gating.db")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        db_path=args.db_path,
        child_block_sizes=[int(x) for x in args.child_block_sizes.split(",")],
        tau_gates=[float(x) for x in args.tau_gates.split(",")],
        tau_train_pct=args.tau_train_pct,
        q_steps=[int(x) for x in args.q_steps.split(",")],
        K=args.K,
        gamma=args.gamma,
    )
