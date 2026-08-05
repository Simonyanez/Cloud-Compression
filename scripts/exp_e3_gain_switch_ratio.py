"""
Experiment E3: "Gain-to-Switch Cost" Ratio Thresholding
========================================================
Goal: Test a dynamic decision boundary that explicitly compares a block's transform
coding gain against its spatial transition cost before allowing a state switch.

Method:
  Standard Viterbi transition cost penalty for switching state is:
    Penalty(k_prev -> k) = gamma * w_i * (k_prev != k)

  With Gain-to-Switch Ratio lambda_switch:
    A block is only allowed to switch to a new cluster if:
      Gain_i > lambda_switch * Penalty_i
    Equivalently, effective switch penalty becomes:
      Penalty_effective = (1 + lambda_switch) * gamma * w_i

  We evaluate lambda_switch in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

Key Metrics Tracked:
  - GFT Savings %
  - Overhead BPV
  - Net Markov BPV
  - Number of Switches
  - 3D Connected Component Count M (does thresholding naturally compress isolated blocks?)

Runtime: ~2-3 min for B=16.
"""

import sys, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
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


def project_slope(mu, v1, v2, unc_norm, degen=1e-3):
    s = np.dot(mu, v1)*v1 + np.dot(mu, v2)*v2
    rn = np.linalg.norm(s)
    if rn < degen: s = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s/rn)*t


def compute_entropy(labels, K):
    cs = np.bincount(labels, minlength=K)
    tr = np.ones((K, K))
    for i in range(1, len(labels)):
        tr[labels[i-1], labels[i]] += 1
    tp = tr / tr.sum(axis=1, keepdims=True)
    h = 0.0
    for p in range(K):
        pp = cs[p]/len(labels)
        for c in range(K):
            v = tp[p, c]
            if v > 0 and pp > 0: h -= pp*v*np.log2(v)
    return h


def _eval_row(block_id, Vblock, Ablock, metadata, struct_coeffs,
              v1, v2, centroids_3d, slp, slw, unc_norm, dm, lag, q, K):
    dec = Decider(dm, lag); dec._set_vars(q)
    gft = GFTStrategyWraper(); app = Approximator()
    b = Block(metadata); b.Vblock = Vblock; b.Ablock = Ablock
    costs = np.full(K, np.inf)
    try: costs[0] = dec._RDcost(struct_coeffs)[0]
    except: pass
    for k in range(1, K):
        s3d = project_slope(centroids_3d[k-1], v1, v2, unc_norm)
        sg = StructuralGraph(metadata); sg.set_data(Vblock)
        ag = AttributeGraph(sg, s3d, k, slp[k], slw[k])
        Vr = app._spatial_norm(Vblock); Aa = Ablock.copy(); Aa[:,0] = Vr@s3d.T
        ag.set_data(Vblock, Aa)
        try:
            _, coeffs = gft(b, ag); c, _, _ = dec._RDcost(coeffs); costs[k] = c
        except: pass
    return costs


def build_cost_matrix(blocks, tangent_data, struct_map, centroids_3d, slp, slw,
                      g3d_raw, pc_V, pc_A, dm, lag, q, K):
    tasks = []
    for i, b in enumerate(blocks):
        b.init_data(pc_V, pc_A)
        tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                      struct_map[b.block_id],
                      tangent_data[i]["v1"], tangent_data[i]["v2"],
                      centroids_3d, slp, slw, g3d_raw[i], dm, lag, q, K))
        b.clear_data()
    rows = Parallel(n_jobs=-1)(delayed(_eval_row)(*t) for t in tasks)
    return np.array(rows)


def build_adjacency(coords, radius):
    N = len(coords)
    adj = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(coords[i]-coords[j]) <= radius:
                adj[i].append(j); adj[j].append(i)
    return adj


def get_3d_components_count(labels, adj, N):
    comp_id = -np.ones(N, dtype=int)
    c_idx = 0
    for i in range(N):
        if comp_id[i] >= 0: continue
        lbl = labels[i]
        queue = deque([i])
        comp_id[i] = c_idx
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if comp_id[nb] < 0 and labels[nb] == lbl:
                    comp_id[nb] = c_idx
                    queue.append(nb)
        c_idx += 1
    return c_idx


def run_viterbi_ratio(cost_matrix, K, gamma, nw, lambda_switch):
    N = len(cost_matrix)
    dp = np.full((N, K), np.inf); paths = np.zeros((N,K), dtype=int)
    dp[0] = cost_matrix[0]
    
    # Effective penalty scaling
    penalty_mult = 1.0 + lambda_switch

    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i-1] + gamma * w * penalty_mult * (np.arange(K) != k)
            bp = np.argmin(tc)
            dp[i, k] = cost_matrix[i, k] + tc[bp]
            paths[i, k] = bp

    vl = np.zeros(N, dtype=int); vl[N-1] = np.argmin(dp[N-1])
    for i in range(N-2, -1, -1): vl[i] = paths[i+1, vl[i+1]]
    return vl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--gamma", type=float, default=2000.0)
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q, gamma = args.block_size, args.k, args.q_step, args.gamma
    K_hybrid = K - 1

    section(f"E3: Gain-to-Switch Ratio Thresholding  (B={B}, K={K}, γ={gamma})")

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    log(f"Partitioning B={B}...")
    _, blocks = MortonBlockPartition().partition(pc, bsize=B)
    N = len(blocks)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks)
    log(f"  {N} blocks, {total_v} voxels")

    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)
    normals, g3d_norm, g3d_raw, coords = [], [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        coords.append(np.mean(b.Vblock, axis=0))
        normals.append(tangent_data[i]["v3"])
        Vc = b.Vblock - b.Vblock.mean(0)
        X = np.column_stack((Vc@tangent_data[i]["v1"], Vc@tangent_data[i]["v2"]))
        Y = b.Ablock[:,0] - b.Ablock[:,0].mean()
        s2d, *_ = np.linalg.lstsq(X, Y, rcond=None)
        g = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        rn = np.linalg.norm(g)
        g3d_raw.append(rn)
        g3d_norm.append(g/rn if rn > 1e-8 else np.array([0.,0.,1.]))
        b.clear_data()
    normals = np.array(normals); g3d_norm = np.array(g3d_norm)
    g3d_raw = np.array(g3d_raw); coords = np.array(coords)

    nw = np.ones(N)
    for i in range(1, N): nw[i] = abs(np.dot(normals[i], normals[i-1]))

    gft_comp = GFTStrategyWraper(); struct_map = {}
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg); struct_map[b.block_id] = coeffs; b.clear_data()

    # Baseline rate
    from pcadc.pointcloud import Sampler
    sampled = Sampler(ratio=0.01, n_strata=5)(pc.V, pc.A, blocks)
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    tr, tv2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tv2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs); tr += r; b.clear_data()
    base_rate = tr / tv2
    log(f"Baseline rate: {base_rate:.5f} bpv")

    # Initial EM centroids & cost matrix
    np.random.seed(42)
    idx0 = np.random.choice(N, K_hybrid, replace=False)
    centroids = g3d_norm[idx0].copy()
    for _ in range(20):
        sims = np.abs(g3d_norm @ centroids.T)
        lbl = np.argmax(sims, axis=1)
        for k in range(K_hybrid):
            m = (lbl==k); sg = np.zeros(3); mu = centroids[k]
            for i in np.where(m)[0]: sg += np.sign(np.dot(g3d_norm[i], mu))*g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n

    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    log("Building Phase 2 cost matrix...")
    cost_matrix = build_cost_matrix(blocks, tangent_data, struct_map, centroids,
                                    slp, slw, g3d_raw, pc.V, pc.A, dm, lag, q, K)
    struct_total = cost_matrix[:, 0].sum()

    # Adjacency for 3D component evaluation
    dists_s = []
    for i in np.random.choice(N, min(200, N), replace=False):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        dists_s.append(d.min())
    radius = np.median(dists_s) * 1.8
    adj = build_adjacency(coords, radius)

    # Sweep lambda_switch
    lambdas = [0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0]

    section("SWEEPING GAIN-TO-SWITCH RATIO (lambda_switch)")
    print(f"\n{'λ_switch':>9} | {'GFT%':>7} | {'Overhead':>10} | "
          f"{'Net BPV':>10} | {'Switches':>8} | {'3D Comps M':>10} | {'ΔNet':>10}")
    print("-" * 75)

    base_net = None
    for lam in lambdas:
        vl = run_viterbi_ratio(cost_matrix, K, gamma, nw, lam)
        gft_c = sum(cost_matrix[i, vl[i]] for i in range(N))
        sav = (struct_total - gft_c)/struct_total*100
        H = compute_entropy(vl, K)
        oh = H*N/total_v
        net = base_rate*(sav/100) - oh
        sw = int((vl[1:]!=vl[:-1]).sum())
        M_comps = get_3d_components_count(vl, adj, N)
        if lam == 0.0:
            base_net = net
            delta_net_str = "---"
        else:
            delta_net = net - base_net
            delta_net_str = f"{delta_net:+10.5f}"

        print(f"{lam:>9.2f} | {sav:>6.3f}% | {oh:>10.5f} | "
              f"{net:>+10.5f} | {sw:>8} | {M_comps:>10} | {delta_net_str:>10}")


if __name__ == "__main__":
    main()
