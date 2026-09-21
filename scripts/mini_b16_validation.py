"""
Multi-K Small Validation Script (B=16 Focus)
============================================
Sweeps codebook capacity K in [4, 6, 8, 10, 12, 16] for Hierarchical (B=32->16) GFT
under the unified Decider formula:
  lambda_mult = lagrange_proportional * q_step^2
with lagrange_proportional = 0.8.
"""
import sys, os, time
import numpy as np
from pathlib import Path

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

def run_multi_k_validation():
    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / "res/longdress_vox10_1051.ply", "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag_base = params.sequential_params.lagrange_proportional  # 0.8
    total_v = pc.V.shape[0]

    q_steps = [24, 32, 48, 64]
    ks_to_test = [4, 6, 8, 10, 12, 16]

    # Partition blocks B=32 and B=16
    _, blocks_32 = MortonBlockPartition().partition(pc, bsize=32)
    _, blocks_16 = MortonBlockPartition().partition(pc, bsize=16)

    tangents_32 = extract_block_tangent_planes(blocks_32, pc.V, pc.A)
    tangents_16 = extract_block_tangent_planes(blocks_16, pc.V, pc.A)

    g3d_norm_32, g3d_raw_32, norm_32 = [], [], []
    for i, b in enumerate(blocks_32):
        b.init_data(pc.V, pc.A)
        norm_32.append(tangents_32[i]["v3"])
        g, rn = fit_gradient_3d(b.Vblock, b.Ablock, tangents_32[i]["v1"], tangents_32[i]["v2"])
        g3d_norm_32.append(g); g3d_raw_32.append(rn)
        b.clear_data()
    g3d_norm_32 = np.array(g3d_norm_32)
    g3d_raw_32 = np.array(g3d_raw_32)

    g3d_norm_16, g3d_raw_16 = [], []
    for i, b in enumerate(blocks_16):
        b.init_data(pc.V, pc.A)
        g, rn = fit_gradient_3d(b.Vblock, b.Ablock, tangents_16[i]["v1"], tangents_16[i]["v2"])
        g3d_norm_16.append(g); g3d_raw_16.append(rn)
        b.clear_data()
    g3d_norm_16 = np.array(g3d_norm_16)
    g3d_raw_16 = np.array(g3d_raw_16)

    gft_comp = GFTStrategyWraper()
    struct_map_16 = {}
    for ci, cb in enumerate(blocks_16):
        cb.init_data(pc.V, pc.A)
        sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
        _, coeffs = gft_comp(cb, sg)
        struct_map_16[ci] = coeffs
        cb.clear_data()

    print("\n========================================================")
    print("1. BASELINE B=16")
    print("========================================================")
    base_res = {}
    for q in q_steps:
        dec = Decider(dm, lag_base)
        dec._set_vars(q)
        tot_rate, tot_dist = 0.0, 0.0
        for i in range(len(blocks_16)):
            _, r_val, d_val = dec._RDcost(struct_map_16[i])
            tot_rate += r_val
            tot_dist += d_val
        net_bpv = tot_rate / total_v
        mse = tot_dist / total_v
        psnr_db = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))
        base_res[q] = (net_bpv, psnr_db)
        print(f"Q={q:<2} | Net BPV: {net_bpv:.5f} | PSNR: {psnr_db:.2f} dB")

    print("\n========================================================")
    print("2. HIERARCHICAL GFT (B=32 -> B=16) MULTI-K SWEEP")
    print("========================================================")
    gamma = 1000.0
    Np = len(blocks_32)
    parent_to_children = build_parent_child_map(blocks_32, blocks_16, pc.V, 32)

    nw_p = np.ones(Np)
    normals_p = np.array(norm_32)
    for i in range(1, Np): nw_p[i] = abs(np.dot(normals_p[i], normals_p[i-1]))

    results_table = []

    for K in ks_to_test:
        K_hybrid = K - 1
        centroids_p, _ = spherical_kmeans(g3d_norm_32, K_hybrid)
        child_act = np.ones(len(blocks_16), dtype=bool)

        slp_opt, slw_opt = grid_search(
            blocks_16, tangents_16, centroids_p, g3d_raw_16, g3d_norm_16,
            pc.V, pc.A, dm, lag_base, 24, K, child_act)

        p_coeffs = compute_all_raw_coeffs_hierarchical(
            blocks_32, blocks_16, parent_to_children, tangents_32, tangents_16,
            struct_map_16, centroids_p, slp_opt, slw_opt, g3d_norm_32, g3d_raw_32,
            pc.V, pc.A, K
        )

        for q in q_steps:
            dec = Decider(dm, lag_base)
            dec._set_vars(q)
            cost_final, rate_final, dist_final = evaluate_hierarchical_decider_on_coeffs(p_coeffs, K, dec)
            vl = run_viterbi(cost_final, K, gamma, nw_p)

            transform_rate_tot = sum(rate_final[i, vl[i]] for i in range(Np))
            transform_bpv = transform_rate_tot / total_v
            H = compute_entropy(vl, K)
            oh_bpv = H * Np / total_v
            net_bpv = transform_bpv + oh_bpv
            tot_dist = sum(dist_final[i, vl[i]] for i in range(Np))
            mse = tot_dist / total_v
            psnr_db = 10.0 * np.log10((255.0**2) / max(mse, 1e-12))

            b_bpv, _ = base_res[q]
            delta = net_bpv - b_bpv
            results_table.append({
                'K': K, 'q_step': q, 'net_bpv': net_bpv,
                'transform_bpv': transform_bpv, 'oh_bpv': oh_bpv,
                'psnr_db': psnr_db, 'delta': delta
            })
            print(f"K={K:>2} | Q={q:<2} | Net BPV: {net_bpv:.5f} (Tr: {transform_bpv:.5f}, OH: {oh_bpv:.5f}) | Delta: {delta:+0.5f} | PSNR: {psnr_db:.2f} dB")

    print("\n========================================================")
    print("3. BEST K SUMMARY PER QUANTIZATION STEP")
    print("========================================================")
    print(f"{'Q Step':>6} | {'Best K':>6} | {'Baseline BPV':>12} | {'Best Hier BPV':>14} | {'Net Delta BPV':>14} | {'Status':>12}")
    print("-" * 75)
    for q in q_steps:
        q_rows = [r for r in results_table if r['q_step'] == q]
        best_r = min(q_rows, key=lambda x: x['net_bpv'])
        b_bpv, _ = base_res[q]
        status = "Net Gain" if best_r['delta'] < 0 else "Penalty"
        print(f"{q:>6} | {best_r['K']:>6} | {b_bpv:>12.5f} | {best_r['net_bpv']:>14.5f} | {best_r['delta']:>+14.5f} | {status:>12}")

if __name__ == "__main__":
    run_multi_k_validation()
