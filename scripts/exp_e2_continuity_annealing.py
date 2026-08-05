"""
Experiment E2: Continuity Annealing Schedule
=============================================
Goal: Determine whether gradually ramping up the spatial continuity penalty gamma
during centroid refinement preserves higher GFT savings than applying a static
large gamma from the start.

Schedule across T=5 outer iterations:
  gamma^(t) = gamma_start * (gamma_target / gamma_start) ** (t / (T - 1))
  e.g. gamma^(t) in [100, 280, 780, 1250, 2000]

Process per outer iteration t:
  1. Evaluate Phase 2 cost matrix with current centroids.
  2. Run Viterbi with current gamma^(t) penalty.
  3. Re-estimate centroids using unconstrained gradients of Viterbi-assigned blocks.
  4. Track GFT savings, sequence entropy H(X|X-1), overhead BPV, net BPV, and switch count.

Comparison:
  Static baseline (gamma = 2000 fixed for all refinement iterations) vs Annealed gamma schedule.

Runtime: ~5-7 min for B=16, K=4.
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
    p.add_argument("--gamma-start", type=float, default=100.0)
    p.add_argument("--gamma-target", type=float, default=2000.0)
    p.add_argument("--iters", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q = args.block_size, args.k, args.q_step
    T = args.iters
    g_start, g_target = args.gamma_start, args.gamma_target
    K_hybrid = K - 1

    section(f"E2: Continuity Annealing Schedule  (B={B}, K={K}, T={T} iters)")

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
    centroids_init = g3d_norm[idx0].copy()
    for _ in range(20):
        sims = np.abs(g3d_norm @ centroids_init.T)
        lbl = np.argmax(sims, axis=1)
        for k in range(K_hybrid):
            m = (lbl==k); sg = np.zeros(3); mu = centroids_init[k]
            for i in np.where(m)[0]: sg += np.sign(np.dot(g3d_norm[i], mu))*g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids_init[k] = sg/n

    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    # Exponential gamma schedule
    gamma_schedule = [g_start * (g_target / g_start) ** (t / (T - 1)) for t in range(T)]
    log(f"Gamma Annealing Schedule: {[round(g, 1) for g in gamma_schedule]}")

    # =========================================================================
    # STATIC BASELINE (Fixed gamma_target)
    # =========================================================================
    section("1. STATIC BASELINE (Fixed gamma = 2000)")
    centroids_static = centroids_init.copy()
    cost_matrix_static = build_cost_matrix(blocks, tangent_data, struct_map,
                                           centroids_static, slp, slw,
                                           g3d_raw, pc.V, pc.A, dm, lag, q, K)
    struct_total = cost_matrix_static[:, 0].sum()
    vl_static = run_viterbi(cost_matrix_static, K, g_target, nw)
    sav_static = (struct_total - sum(cost_matrix_static[i, vl_static[i]] for i in range(N))) / struct_total * 100
    H_static = compute_entropy(vl_static, K)
    oh_static = H_static * N / total_v
    net_static = base_rate * (sav_static / 100) - oh_static
    sw_static = int((vl_static[1:] != vl_static[:-1]).sum())

    log(f"Static Final (γ={g_target:.0f}): GFT={sav_static:.3f}%, H={H_static:.4f}, "
        f"Overhead={oh_static:.5f}, Net={net_static:+.5f}, Switches={sw_static}")

    # =========================================================================
    # ANNEALED SCHEDULE
    # =========================================================================
    section("2. CONTINUITY ANNEALING LOOP")
    centroids_anneal = centroids_init.copy()

    print(f"\n{'Iter':>4} | {'γ^(t)':>7} | {'GFT%':>7} | {'H(X|X-1)':>8} | "
          f"{'Overhead':>10} | {'Net BPV':>10} | {'Switches':>8}")
    print("-" * 75)

    anneal_history = []
    for t in range(T):
        gamma_t = gamma_schedule[t]
        cost_matrix_t = build_cost_matrix(blocks, tangent_data, struct_map,
                                          centroids_anneal, slp, slw,
                                          g3d_raw, pc.V, pc.A, dm, lag, q, K)
        vl_t = run_viterbi(cost_matrix_t, K, gamma_t, nw)
        sav_t = (struct_total - sum(cost_matrix_t[i, vl_t[i]] for i in range(N))) / struct_total * 100
        H_t = compute_entropy(vl_t, K)
        oh_t = H_t * N / total_v
        net_t = base_rate * (sav_t / 100) - oh_t
        sw_t = int((vl_t[1:] != vl_t[:-1]).sum())

        print(f"{t+1:>4} | {gamma_t:>7.1f} | {sav_t:>6.3f}% | {H_t:>8.4f} | "
              f"{oh_t:>10.5f} | {net_t:>+10.5f} | {sw_t:>8}")

        anneal_history.append((gamma_t, sav_t, H_t, oh_t, net_t, sw_t))

        # Re-estimate centroids for next iteration based on Viterbi labels
        if t < T - 1:
            for k_idx in range(K_hybrid):
                k = k_idx + 1
                assigned = np.where(vl_t == k)[0]
                if len(assigned) >= 2:
                    sg = np.zeros(3); mu = centroids_anneal[k_idx]
                    for idx in assigned:
                        g = g3d_norm[idx]
                        sg += np.sign(np.dot(g, mu)) * g
                    n = np.linalg.norm(sg)
                    if n > 1e-8: centroids_anneal[k_idx] = sg / n

    final_anneal = anneal_history[-1]
    delta_net = final_anneal[4] - net_static

    section("SUMMARY COMPARISON")
    log(f"Static Net BPV (γ={g_target:.0f}):   {net_static:+.5f}")
    log(f"Annealed Net BPV (final γ={g_target:.0f}): {final_anneal[4]:+.5f}")
    log(f"Δ Net BPV (Annealed - Static): {delta_net:+.5f}")

    if delta_net > 0.00005:
        log("→ ANNEALING WINS: Ramping up gamma prevents GFT collapse.")
    elif delta_net < -0.00005:
        log("→ STATIC WINS: Early iterations with small gamma created poor centroids.")
    else:
        log("→ DRAW: Continuity annealing has negligible impact.")


if __name__ == "__main__":
    main()
