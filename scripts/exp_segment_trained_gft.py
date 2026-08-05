"""
Segment-Trained Codebook Feasibility Test
=========================================
Tests the hypothesis:
  "Do GFT transforms trained SPECIFICALLY on local spatial segments (defined by geometry)
   yield significantly higher GFT rate savings than global 3D centroids?"

Pipeline:
  1. Partition blocks into geometric segments (Rule 1: normal angle breaks).
  2. For each segment, extract all member blocks' unconstrained 3D gradients.
  3. Compute a LOCAL 3D centroid per segment (mean direction of its member blocks).
  4. Compare GFT energy compaction / R-D cost per block:
       - Global Codebook Centroid (current pipeline)
       - Segment-Local Centroid (trained specifically on the segment's blocks)
       - Local Block Unconstrained Gradient (upper bound on single-direction alignment)

If Segment-Local Centroids yield significantly higher GFT savings (> 0.5-1.0% boost),
then training codebooks per spatial segment is worth pursuing.
If the difference is marginal (< 0.1%), global codebooks are already capturing
the directional signal as well as local segment codebooks can.

Runtime: ~2-3 min for B=16.
"""

import sys, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
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


def project_slope(mu_k, v1, v2, unc_norm, degen=1e-3):
    s3d = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
    rn = np.linalg.norm(s3d)
    if rn < degen: s3d = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s3d/rn)*t


def eval_block_single_dir(block, metadata, struct_coeffs, v1, v2, mu_dir, unc_norm,
                           slp, slw, k_cluster, dm, lag, q):
    dec = Decider(dm, lag); dec._set_vars(q)
    gft = GFTStrategyWraper(); app = Approximator()
    s3d = project_slope(mu_dir, v1, v2, unc_norm)
    sg = StructuralGraph(metadata); sg.set_data(block.Vblock)
    ag = AttributeGraph(sg, s3d, k_cluster, slp, slw)
    Vr = app._spatial_norm(block.Vblock)
    Aa = block.Ablock.copy(); Aa[:, 0] = Vr @ s3d.T
    ag.set_data(block.Vblock, Aa)
    try:
        _, coeffs = gft(block, ag)
        c, _, _ = dec._RDcost(coeffs)
        return c
    except Exception:
        return 1e18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--q-step", type=int, default=24)
    args = parser.parse_args()

    B, K, q = args.block_size, args.k, args.q_step
    section(f"Segment-Trained Codebook Test (B={B}, K={K})")

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    log(f"Partitioning B={B}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=B)
    N = len(blocks)
    log(f"  {N} blocks")

    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)

    normals, g3d_norm, g3d_raw = [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        normals.append(tangent_data[i]["v3"])
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
    g3d_norm = np.array(g3d_norm)
    g3d_raw = np.array(g3d_raw)

    gft_comp = GFTStrategyWraper()
    struct_map = {}
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg); struct_map[b.block_id] = coeffs; b.clear_data()

    # Evaluate Global K-Means Centroids (Baseline)
    np.random.seed(42)
    idx0 = np.random.choice(len(g3d_norm), K-1, replace=False)
    global_centroids = g3d_norm[idx0].copy()

    # Test thresholds for segment creation
    theta_list = [20, 30, 45]

    for theta in theta_list:
        section(f"Testing Segment-Local Codebooks (Normal Angle Break θ = {theta}°)")

        # Create segments
        segments = []
        seg_start = 0
        for i in range(1, N):
            if normal_angle_deg(normals[i], normals[i-1]) > theta:
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, N))

        log(f"Created {len(segments)} segments (avg size: {N/len(segments):.1f} blocks)")

        # For each segment, compute local centroid
        local_centroids = []
        for start, end in segments:
            seg_grads = g3d_norm[start:end]
            sg = seg_grads.sum(axis=0)
            n = np.linalg.norm(sg)
            local_centroids.append(sg / n if n > 1e-8 else np.array([0., 0., 1.]))

        # Compare R-D costs across all blocks for 3 modes:
        # Mode 1: Global best centroid (from global codebook)
        # Mode 2: Segment-Local centroid (trained on that segment)
        # Mode 3: Block's own unconstrained gradient (oracle single-block centroid)

        log("Evaluating R-D costs for Global vs Segment-Local vs Block-Unconstrained...")
        cost_global, cost_local, cost_unconstrained = 0.0, 0.0, 0.0
        cost_struct = 0.0

        for s_idx, (start, end) in enumerate(segments):
            loc_c = local_centroids[s_idx]
            for i in range(start, end):
                b = blocks[i]
                b.init_data(pc.V, pc.A)
                v1, v2 = tangent_data[i]["v1"], tangent_data[i]["v2"]
                meta = b.metadata
                st_c = struct_map[b.block_id]
                unc_n = g3d_raw[i]

                # Structural base cost
                dec = Decider(dm, lag); dec._set_vars(q)
                c_str = dec._RDcost(st_c)[0]
                cost_struct += c_str

                # Mode 1: Global best centroid
                costs_g = [eval_block_single_dir(b, meta, st_c, v1, v2, gc, unc_n, 0.30, 0.80, k+1, dm, lag, q)
                           for k, gc in enumerate(global_centroids)]
                cost_global += min(c_str, min(costs_g))

                # Mode 2: Segment-Local centroid
                c_loc = eval_block_single_dir(b, meta, st_c, v1, v2, loc_c, unc_n, 0.30, 0.80, 1, dm, lag, q)
                cost_local += min(c_str, c_loc)

                # Mode 3: Block unconstrained gradient
                c_unc = eval_block_single_dir(b, meta, st_c, v1, v2, g3d_norm[i], unc_n, 0.30, 0.80, 1, dm, lag, q)
                cost_unconstrained += min(c_str, c_unc)

                b.clear_data()

        sav_global = (cost_struct - cost_global) / cost_struct * 100
        sav_local = (cost_struct - cost_local) / cost_struct * 100
        sav_unc = (cost_struct - cost_unconstrained) / cost_struct * 100

        log(f"Results for θ = {theta}°:")
        log(f"  Mode 1 (Global K={K} Codebook):     GFT Savings = {sav_global:.3f}%")
        log(f"  Mode 2 (Segment-Local Centroid):    GFT Savings = {sav_local:.3f}%")
        log(f"  Mode 3 (Block-Unconstrained Oracle): GFT Savings = {sav_unc:.3f}%")
        log(f"  Δ Savings (Segment-Local vs Global): {sav_local - sav_global:+.3f}%")


if __name__ == "__main__":
    main()
