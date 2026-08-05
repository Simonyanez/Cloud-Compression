"""
Experiment E1-Refinement 3: High-Resolution Sub-Clustering on Active Blocks
=============================================================================
Hypothesis:
  ~27% of the point cloud contains strong directional signals, while ~73% defaults
  to Cluster 0 (flat structural fallback). Using only K_dir = 3 directional clusters
  may under-fit those high-variance regions. Expanding directional modes to
  K_dir in [4, 6, 8] (Total K in [5, 7, 9]) specifically on active blocks (Gain > 0.5%)
  could significantly boost raw GFT savings without wasting codebook capacity on flat blocks.

Evaluates:
  Total K in [4, 5, 7, 9] (i.e. K_dir in [3, 4, 6, 8])
  tau_filter = 0.5%

Key Metrics Tracked:
  - Active Block Count
  - Raw GFT Savings %
  - Sequence Entropy H(X|X-1) & Overhead BPV
  - Net BPV across gammas in [1000, 2000, 4000]

Runtime: ~5-7 min for B=16.
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
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--tau-filter", type=float, default=0.5,
                   help="Gain threshold %% to identify active directional blocks")
    p.add_argument("--gammas", type=str, default="1000,2000,4000")
    return p.parse_args()


def main():
    args = parse_args()
    B, q = args.block_size, args.q_step
    tau_filter = args.tau_filter
    gammas = [float(x) for x in args.gammas.split(",")]

    section(f"Active Sub-Clustering  (B={B}, tau_filter={tau_filter}%)")

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

    # Initial K=4 cost matrix to compute active mask
    np.random.seed(42)
    idx0 = np.random.choice(N, 3, replace=False)
    centroids_init = g3d_norm[idx0].copy()
    slp_init = np.concatenate([[0.0], np.full(3, 0.30)])
    slw_init = np.concatenate([[1.0], np.full(3, 0.80)])
    cost_matrix_init = build_cost_matrix(blocks, tangent_data, struct_map, centroids_init,
                                         slp_init, slw_init, g3d_raw, pc.V, pc.A, dm, lag, q, 4)
    
    best_dir_cost = cost_matrix_init[:, 1:].min(axis=1)
    gains_pct = (cost_matrix_init[:, 0] - best_dir_cost) / (cost_matrix_init[:, 0] + 1e-12) * 100
    active_mask = gains_pct > tau_filter
    active_grads = g3d_norm[active_mask]
    log(f"Identified {len(active_grads)} active blocks ({100*len(active_grads)/N:.1f}%) with gain > {tau_filter}%")

    # Sweeping Total K in [4, 5, 7, 9] -> K_dir in [3, 4, 6, 8]
    total_k_values = [4, 5, 7, 9]

    section("ACTIVE SUB-CLUSTERING RESULTS")
    print(f"\n{'Total K':>7} | {'K_dir':>5} | {'γ':>6} | {'GFT%':>7} | "
          f"{'Overhead':>10} | {'Net BPV':>10} | {'Switches':>8}")
    print("-" * 75)

    for total_K in total_k_values:
        K_dir = total_K - 1
        # Train K_dir centroids on active blocks
        centroids_k, _ = spherical_kmeans(active_grads, K_dir)
        slp_k = np.concatenate([[0.0], np.full(K_dir, 0.30)])
        slw_k = np.concatenate([[1.0], np.full(K_dir, 0.80)])

        cost_matrix_k = build_cost_matrix(blocks, tangent_data, struct_map, centroids_k,
                                          slp_k, slw_k, g3d_raw, pc.V, pc.A, dm, lag, q, total_K)
        struct_total = cost_matrix_k[:, 0].sum()

        for gamma in gammas:
            vl = run_viterbi(cost_matrix_k, total_K, gamma, nw)
            gft_c = sum(cost_matrix_k[i, vl[i]] for i in range(N))
            sav = (struct_total - gft_c)/struct_total*100
            H = compute_entropy(vl, total_K)
            oh = H * N / total_v
            net = base_rate * (sav / 100) - oh
            sw = int((vl[1:] != vl[:-1]).sum())
            print(f"{total_K:>7} | {K_dir:>5} | {gamma:>6.0f} | {sav:>6.3f}% | "
                  f"{oh:>10.5f} | {net:>+10.5f} | {sw:>8}")


if __name__ == "__main__":
    main()
