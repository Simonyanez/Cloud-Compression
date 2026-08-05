"""
Experiment E1-Refinement 2: Pure Iterative tau-Pruning (Warmup -> Target)
========================================================================
Hypothesis:
  Ramping up the hard cutoff threshold tau over 4 micro-iterations purely in the
  signal domain (gamma = 0) avoids abrupt vector jumps in Phase 2 centroids,
  allowing centroids to follow a smooth convergence path.

Schedule across T=4 micro-iterations:
  tau_t in [0.0%, 0.25%, 0.50%, 1.00%]

Method per micro-iteration t:
  1. Calculate per-block gains against current centroids.
  2. Mask active blocks M_i = (Gain_i > tau_t).
  3. Retrain directional centroids on active blocks.
  4. At final step t=3, evaluate full Phase 2 cost matrix + Viterbi to compute Net BPV.

Runtime: ~4-5 min for B=16, K=4.
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


def project_slope(mu, v1, v2, unc_norm, degen=1e-3):
    s = np.dot(mu, v1)*v1 + np.dot(mu, v2)*v2
    rn = np.linalg.norm(s)
    if rn < degen: s = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s/rn)*t


def spherical_kmeans(directions, K, max_iter=40, seed=42):
    np.random.seed(seed)
    if len(directions) < K:
        extra = K - len(directions)
        rand = np.random.randn(extra, 3)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        directions = np.vstack([directions, rand])
    idx = np.random.choice(len(directions), K, replace=False)
    centroids = directions[idx].copy()
    labels = np.zeros(len(directions), dtype=int)
    for _ in range(max_iter):
        prev = labels.copy()
        sims = np.abs(directions @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev): break
        for k in range(K):
            m = directions[labels == k]
            if len(m) == 0: continue
            sg = m.sum(axis=0)
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n
    return centroids, labels


def centroid_separation(centroids):
    K = len(centroids)
    if K < 2: return 0.0
    vals = []
    for i in range(K):
        for j in range(i+1, K):
            vals.append(abs(np.dot(centroids[i], centroids[j])))
    return float(np.mean(vals))


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


def run_viterbi(cost_matrix, K, gamma, nw):
    N = len(cost_matrix)
    dp = np.full((N, K), np.inf); paths = np.zeros((N,K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i-1] + gamma*w*(np.arange(K)!=k)
            bp = np.argmin(tc); dp[i,k] = cost_matrix[i,k]+tc[bp]; paths[i,k] = bp
    vl = np.zeros(N, dtype=int); vl[N-1] = np.argmin(dp[N-1])
    for i in range(N-2,-1,-1): vl[i] = paths[i+1,vl[i+1]]
    return vl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--gammas", type=str, default="1000,2000,4000")
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q = args.block_size, args.k, args.q_step
    K_hybrid = K - 1
    gammas = [float(x) for x in args.gammas.split(",")]

    section(f"Iterative tau-Pruning Schedule  (B={B}, K={K})")

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
    normals, g3d_norm, g3d_raw = [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
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
    normals = np.array(normals); g3d_norm = np.array(g3d_norm); g3d_raw = np.array(g3d_raw)

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

    # Initial EM centroids
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

    # Iterative tau schedule (pure signal domain, gamma = 0 during centroid tuning)
    tau_schedule = [0.0, 0.25, 0.50, 1.00]

    section("ITERATIVE PRUNING PROGRESSION")
    print(f"\n{'Iter':>4} | {'τ threshold':>11} | {'Active%':>7} | {'CentSep':>8}")
    print("-" * 45)

    for t, tau_t in enumerate(tau_schedule):
        # Build cost matrix with current centroids
        cost_matrix = build_cost_matrix(blocks, tangent_data, struct_map, centroids,
                                        slp, slw, g3d_raw, pc.V, pc.A, dm, lag, q, K)
        struct_total = cost_matrix[:, 0].sum()

        # Compute gains
        best_dir_cost = cost_matrix[:, 1:].min(axis=1)
        gains_pct = (cost_matrix[:, 0] - best_dir_cost) / (cost_matrix[:, 0] + 1e-12) * 100
        active_mask = gains_pct > tau_t
        n_active = active_mask.sum()
        active_pct = 100 * n_active / N

        if n_active >= K_hybrid * 3:
            active_grads = g3d_norm[active_mask]
            centroids, _ = spherical_kmeans(active_grads, K_hybrid)

        sep = centroid_separation(centroids)
        print(f"{t+1:>4} | {tau_t:>10.2f}% | {active_pct:>6.1f}% | {sep:>8.4f}")

    # Final evaluation of cost matrix & Viterbi
    log("\nEvaluating final cost matrix and Viterbi across gammas...")
    cost_matrix_final = build_cost_matrix(blocks, tangent_data, struct_map, centroids,
                                          slp, slw, g3d_raw, pc.V, pc.A, dm, lag, q, K)
    struct_total = cost_matrix_final[:, 0].sum()

    section("FINAL PERFORMANCE COMPARISON")
    print(f"\n{'γ':>6} | {'GFT%':>7} | {'Overhead':>10} | {'Net BPV':>10} | {'Switches':>8}")
    print("-" * 55)

    for gamma in gammas:
        vl = run_viterbi(cost_matrix_final, K, gamma, nw)
        gft_c = sum(cost_matrix_final[i, vl[i]] for i in range(N))
        sav = (struct_total - gft_c)/struct_total*100
        H = compute_entropy(vl, K)
        oh = H * N / total_v
        net = base_rate * (sav / 100) - oh
        sw = int((vl[1:] != vl[:-1]).sum())
        print(f"{gamma:>6.0f} | {sav:>6.3f}% | {oh:>10.5f} | {net:>+10.5f} | {sw:>8}")


if __name__ == "__main__":
    main()
