"""
Fast Validation Sweep Script
============================
Evaluates Baseline vs Hierarchical GFT under the unified Decider formulation:
lambda_mult = lagrange_proportional * q_step^2
using lagrange_proportional = 0.8.
"""
import sys, os, time
import numpy as np
import sqlite3
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block, BlockMetadata
from oracle_em_tangent import extract_block_tangent_planes

from exp_e4_master_benchmark_suite import (
    extract_block_tangent_planes, fit_gradient_3d, spherical_kmeans,
    project_slope, fast_lstsq_2d, build_parent_child_map,
    compute_all_raw_coeffs_hierarchical, evaluate_hierarchical_decider_on_coeffs,
    run_viterbi, compute_entropy
)

def run_fast_validation():
    db_path = "results/fast_validation.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE val_results (
        mode TEXT, child_b INT, q_step INT, net_bpv REAL, overhead_bpv REAL, psnr_db REAL
    )
    """)
    conn.commit()

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / "res/longdress_vox10_1051.ply", "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag_base = params.sequential_params.lagrange_proportional # 0.8
    total_v = pc.V.shape[0]

    q_steps = [24, 32, 48, 64]
    child_scales = [16, 8, 4]
    Bp = 32
    K = 10
    K_hybrid = K - 1
    tau = 0.5
    em = 2
    gamma = 1000.0

    print("Pre-partitioning block scales...")
    partitions = {}
    tangents = {}
    gradients = {}
    normals = {}
    struct_maps = {}

    for B in [32, 16, 8, 4]:
        _, blocks = MortonBlockPartition().partition(pc, bsize=B)
        partitions[B] = blocks
        tangents[B] = extract_block_tangent_planes(blocks, pc.V, pc.A)
        g3d_norm, g3d_raw, norm_v = [], [], []
        for i, b in enumerate(blocks):
            b.init_data(pc.V, pc.A)
            norm_v.append(tangents[B][i]["v3"])
            g, rn = fit_gradient_3d(b.Vblock, b.Ablock, tangents[B][i]["v1"], tangents[B][i]["v2"])
            g3d_norm.append(g); g3d_raw.append(rn)
            b.clear_data()
        gradients[B] = (np.array(g3d_norm), np.array(g3d_raw))
        normals[B] = np.array(norm_v)

        gft_comp = GFTStrategyWraper()
        s_map = {}
        for ci, cb in enumerate(blocks):
            cb.init_data(pc.V, pc.A)
            sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
            _, coeffs = gft_comp(cb, sg)
            s_map[ci] = coeffs
            cb.clear_data()
        struct_maps[B] = s_map

    print("\n========================================================")
    print("MODE 1: BASELINE GFT")
    print("========================================================")
    for B in child_scales:
        blocks = partitions[B]
        s_map = struct_maps[B]
        for q_step in q_steps:
            dec = Decider(dm, lag_base)
            dec._set_vars(q_step)
            tot_rate, tot_dist = 0.0, 0.0
            for i in range(len(blocks)):
                c_val, r_val, d_val = dec._RDcost(s_map[i])
                tot_rate += r_val
                tot_dist += d_val
            net_bpv = tot_rate / total_v
            mse = tot_dist / total_v
            psnr_db = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
            c.execute("INSERT INTO val_results VALUES (?, ?, ?, ?, ?, ?)",
                      ("baseline", B, q_step, net_bpv, 0.0, psnr_db))
            print(f"Baseline B={B:<2} | Q={q_step:<2} | Net BPV: {net_bpv:.5f} | PSNR: {psnr_db:.2f} dB")
    conn.commit()

    print("\n========================================================")
    print("MODE 3: HIERARCHICAL GFT (B=32 -> B_child)")
    print("========================================================")
    blocks_p = partitions[Bp]
    Np = len(blocks_p)
    tangent_p = tangents[Bp]
    g3d_norm_p, g3d_raw_p = gradients[Bp]
    normals_p = normals[Bp]
    nw_p = np.ones(Np)
    for i in range(1, Np): nw_p[i] = abs(np.dot(normals_p[i], normals_p[i-1]))

    for Bc in child_scales:
        blocks_c = partitions[Bc]
        tangent_c = tangents[Bc]
        g3d_norm_c, g3d_raw_c = gradients[Bc]
        struct_map_c = struct_maps[Bc]
        parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc.V, Bp)

        centroids_p, _ = spherical_kmeans(g3d_norm_p, K_hybrid)
        slp_p = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
        slw_p = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

        p_coeffs1 = compute_all_raw_coeffs_hierarchical(
            blocks_p, blocks_c, parent_to_children, tangent_p, tangent_c,
            struct_map_c, centroids_p, slp_p, slw_p, g3d_norm_p, g3d_raw_p,
            pc.V, pc.A, K
        )
        dec_init = Decider(dm, lag_base); dec_init._set_vars(24)
        cost_pass1, _, _ = evaluate_hierarchical_decider_on_coeffs(p_coeffs1, K, dec_init)
        gains1 = (cost_pass1[:, 0] - cost_pass1[:, 1:].min(axis=1)) / (cost_pass1[:, 0] + 1e-12) * 100
        active_mask_p = gains1 > tau
        if active_mask_p.sum() >= K_hybrid * 3:
            centroids_p, _ = spherical_kmeans(g3d_norm_p[active_mask_p], K_hybrid)

        if em == 2:
            p_coeffs2 = compute_all_raw_coeffs_hierarchical(
                blocks_p, blocks_c, parent_to_children, tangent_p, tangent_c,
                struct_map_c, centroids_p, slp_p, slw_p, g3d_norm_p, g3d_raw_p,
                pc.V, pc.A, K
            )
            cost_pass2, _, _ = evaluate_hierarchical_decider_on_coeffs(p_coeffs2, K, dec_init)
            gains2 = (cost_pass2[:, 0] - cost_pass2[:, 1:].min(axis=1)) / (cost_pass2[:, 0] + 1e-12) * 100
            active_mask_p2 = gains2 > tau
            if active_mask_p2.sum() >= K_hybrid * 3:
                centroids_p, _ = spherical_kmeans(g3d_norm_p[active_mask_p2], K_hybrid)

        slp_opt, slw_opt = slp_p, slw_p
        p_coeffs_final = compute_all_raw_coeffs_hierarchical(
            blocks_p, blocks_c, parent_to_children, tangent_p, tangent_c,
            struct_map_c, centroids_p, slp_opt, slw_opt, g3d_norm_p, g3d_raw_p,
            pc.V, pc.A, K
        )

        for q_step in q_steps:
            dec = Decider(dm, lag_base); dec._set_vars(q_step)
            cost_final, rate_final, dist_final = evaluate_hierarchical_decider_on_coeffs(p_coeffs_final, K, dec)
            vl = run_viterbi(cost_final, K, gamma, nw_p)

            transform_rate_tot = sum(rate_final[i, vl[i]] for i in range(Np))
            transform_bpv = transform_rate_tot / total_v
            H = compute_entropy(vl, K)
            oh_bpv = H * Np / total_v
            net_bpv = transform_bpv + oh_bpv
            tot_dist = sum(dist_final[i, vl[i]] for i in range(Np))
            mse = tot_dist / total_v
            psnr_db = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))

            c.execute("INSERT INTO val_results VALUES (?, ?, ?, ?, ?, ?)",
                      ("hierarchical", Bc, q_step, net_bpv, oh_bpv, psnr_db))
            print(f"Hierarchical B=32->{Bc:<2} | Q={q_step:<2} | Net BPV: {net_bpv:.5f} (OH: {oh_bpv:.5f}) | PSNR: {psnr_db:.2f} dB")
    conn.commit()

    print("\n========================================================")
    print("BD-RATE EVALUATION (BASELINE VS HIERARCHICAL)")
    print("========================================================")
    def bd_rate(R1, P1, R2, P2):
        lR1, lR2 = np.log(R1), np.log(R2)
        p1 = np.polyfit(P1, lR1, 3)
        p2 = np.polyfit(P2, lR2, 3)
        min_p = max(min(P1), min(P2))
        max_p = min(max(P1), max(P2))
        p1_int = np.polyint(p1)
        p2_int = np.polyint(p2)
        int1 = np.polyval(p1_int, max_p) - np.polyval(p1_int, min_p)
        int2 = np.polyval(p2_int, max_p) - np.polyval(p2_int, min_p)
        avg_diff = (int2 - int1) / (max_p - min_p)
        return (np.exp(avg_diff) - 1.0) * 100.0

    for B in child_scales:
        c.execute("SELECT net_bpv, psnr_db FROM val_results WHERE mode='baseline' AND child_b=? ORDER BY q_step", (B,))
        b_rows = c.fetchall()
        c.execute("SELECT net_bpv, psnr_db FROM val_results WHERE mode='hierarchical' AND child_b=? ORDER BY q_step", (B,))
        h_rows = c.fetchall()
        R1, P1 = np.array([r[0] for r in b_rows]), np.array([r[1] for r in b_rows])
        R2, P2 = np.array([r[0] for r in h_rows]), np.array([r[1] for r in h_rows])
        bd = bd_rate(R1, P1, R2, P2)
        print(f"Block Scale B={B:<2}: BD-Rate = {bd:+.2f}%")

    conn.close()

if __name__ == "__main__":
    run_fast_validation()
