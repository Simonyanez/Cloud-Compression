"""
Experiment 2: Spatial Segment Grouping + Run-Length Overhead Model
===================================================================
Motivation: EXP C showed H_oracle = 0.044 vs H_morton = 1.378.
The oracle requires knowing cluster labels in advance and sorting by them.
But we can approximate this AFTER Phase 2 by grouping spatially adjacent
same-label blocks into segments before applying the Viterbi.

Key idea: instead of assigning a new code word to every block, assign one
code word per SEGMENT (a run of consecutive same-label blocks in Morton order,
or a connected 3D spatial component). The overhead then scales with the number
of segments rather than the number of blocks.

Two grouping strategies tested:

  STRATEGY A — Morton Run-Length (RLE):
    After Viterbi, merge consecutive same-label blocks into runs.
    Enforce a minimum run size S: isolated blocks (run length < S) are
    absorbed into their cheapest-neighbour cluster.
    Overhead = (n_runs * log2(K) + sum_run_lengths(log2(run_len))) / total_v

  STRATEGY B — 3D Spatial Superpixels (pre-Viterbi):
    Group spatially adjacent same-assignment blocks (from Phase 2 argmin)
    into connected components. Each component votes for a label.
    Run Viterbi on components instead of individual blocks.
    Overhead = H_component_sequence * n_components / total_v

Both are compared against the current Morton block-level Viterbi.

Run time: ~3-5 min for B=16, K=4 (dominated by cost matrix evaluation).
"""

import sys, os, time, argparse
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
from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.transforms import GFTStrategyWraper
from pcadc.blocks import Block
from oracle_em_tangent import extract_block_tangent_planes


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}", flush=True)


# ---------------------------------------------------------------------------
# Geometry + EM (minimal, same as v2)
# ---------------------------------------------------------------------------

def spherical_em(g3d_norm, K_hybrid):
    np.random.seed(42)
    idx0 = np.random.choice(len(g3d_norm), K_hybrid, replace=False)
    centroids = g3d_norm[idx0].copy()
    labels = np.zeros(len(g3d_norm), dtype=int)
    for _ in range(20):
        prev = labels.copy()
        sims = np.abs(g3d_norm @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev): break
        for k in range(K_hybrid):
            m = (labels == k)
            if m.sum() == 0: continue
            sg = np.zeros(3); mu = centroids[k]
            for i in np.where(m)[0]:
                sg += np.sign(np.dot(g3d_norm[i], mu)) * g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n
    return centroids, labels


def project_slope(mu_k, v1, v2, unc_norm, degen=1e-3):
    s3d = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
    rn = np.linalg.norm(s3d)
    if rn < degen: s3d = v1*degen; rn = degen
    target = unc_norm if unc_norm > 1e-8 else rn
    return (s3d/rn)*target


def _eval_block_row(block_id, Vblock, Ablock, metadata, struct_coeffs,
                    v1, v2, centroids_3d, slp, slw, unc_norm,
                    dm, lag, q_step, K):
    dec = Decider(dm, lag); dec._set_vars(q_step)
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


# ---------------------------------------------------------------------------
# Overhead models
# ---------------------------------------------------------------------------

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
            v = tp[p,c]
            if v>0 and pp>0: h -= pp*v*np.log2(v)
    return h


def markov_overhead(labels, K, N, total_v):
    """Current overhead model: H(X|X-1) * N / total_v."""
    return compute_entropy(labels, K) * N / total_v


def rle_overhead(labels, K, total_v):
    """
    Run-length encoding overhead.
    Each run costs: log2(K) bits for label + log2(N/N_runs) bits for length.
    Conservative estimate using fixed-length run coding.
    """
    runs = []
    cur_label, cur_len = labels[0], 1
    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_len += 1
        else:
            runs.append((cur_label, cur_len))
            cur_label, cur_len = labels[i], 1
    runs.append((cur_label, cur_len))
    N_runs = len(runs)
    mean_run = len(labels) / N_runs
    bits_per_run = np.log2(K) + np.log2(max(mean_run, 1))
    return N_runs * bits_per_run / total_v, N_runs, [r[1] for r in runs]


# ---------------------------------------------------------------------------
# Strategy A: Morton RLE with forced minimum segment size
# ---------------------------------------------------------------------------

def force_min_segment(labels, cost_matrix, K, min_size):
    """
    Merge runs shorter than min_size into the cheapest neighbouring cluster.
    Returns new labels and segment count.
    """
    labels = labels.copy()
    N = len(labels)
    changed = True
    max_passes = 5
    for _ in range(max_passes):
        if not changed: break
        changed = False
        # Find runs
        runs = []
        i = 0
        while i < N:
            j = i
            while j < N and labels[j] == labels[i]: j += 1
            runs.append((i, j, labels[i]))  # (start, end_excl, label)
            i = j
        # Merge short runs
        for r_idx, (start, end, lbl) in enumerate(runs):
            if end - start >= min_size: continue
            # Find cheapest cluster for this run
            avg_costs = cost_matrix[start:end].mean(axis=0)
            # Don't allow staying if the run is short
            new_lbl = lbl
            best_cost = np.inf
            for k in range(K):
                if avg_costs[k] < best_cost:
                    best_cost = avg_costs[k]
                    new_lbl = k
            if new_lbl != lbl:
                labels[start:end] = new_lbl
                changed = True
    # Count segments after merging
    n_segs = 1 + int((labels[1:] != labels[:-1]).sum())
    return labels, n_segs


# ---------------------------------------------------------------------------
# Strategy B: 3D Spatial Connected Components
# ---------------------------------------------------------------------------

def build_adjacency_fast(coords, radius):
    """Fast radius-based adjacency (O(N^2) but feasible for N~3500)."""
    N = len(coords)
    adj = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(coords[i]-coords[j]) <= radius:
                adj[i].append(j); adj[j].append(i)
    return adj


def connected_components_3d(p2_labels, adj, N):
    """
    Find connected components where adjacent blocks share the same Phase 2 label.
    Returns component_id per block, and the label of each component.
    """
    comp_id = -np.ones(N, dtype=int)
    comp_labels = []
    current_comp = 0
    for start in range(N):
        if comp_id[start] >= 0: continue
        lbl = p2_labels[start]
        queue = deque([start])
        comp_id[start] = current_comp
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if comp_id[nb] < 0 and p2_labels[nb] == lbl:
                    comp_id[nb] = current_comp
                    queue.append(nb)
        comp_labels.append(lbl)
        current_comp += 1
    return comp_id, np.array(comp_labels)


def viterbi_on_components(comp_labels_arr, comp_cost_matrix, K, gamma,
                           comp_normal_weights):
    """Run Viterbi where each node is a component (not a block)."""
    M = len(comp_labels_arr)
    dp = np.full((M, K), np.inf); paths = np.zeros((M,K), dtype=int)
    dp[0] = comp_cost_matrix[0]
    for i in range(1, M):
        w = comp_normal_weights[i]
        for k in range(K):
            tc = dp[i-1] + gamma*w*(np.arange(K)!=k)
            bp = np.argmin(tc); dp[i,k] = comp_cost_matrix[i,k]+tc[bp]; paths[i,k]=bp
    vl = np.zeros(M, dtype=int); vl[M-1] = np.argmin(dp[M-1])
    for i in range(M-2,-1,-1): vl[i] = paths[i+1,vl[i+1]]
    return vl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--gammas", type=str, default="1000,2000,4000")
    p.add_argument("--min-sizes", type=str, default="1,2,3,5,8",
                   help="Min segment sizes to test for Strategy A")
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q = args.block_size, args.k, args.q_step
    gammas = [float(x) for x in args.gammas.split(",")]
    min_sizes = [int(x) for x in args.min_sizes.split(",")]
    K_hybrid = K - 1

    section(f"EXP 2: Spatial Segment Grouping  (B={B}, K={K})")

    # Setup
    params = load_experiment_config(project_root / "config/base_config.yaml")
    colourist = Colourist()
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
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
        g3d_norm.append(g/rn if rn>1e-8 else np.array([0.,0.,1.]))
        b.clear_data()
    normals = np.array(normals); g3d_norm = np.array(g3d_norm)
    g3d_raw = np.array(g3d_raw); coords = np.array(coords)

    nw = np.ones(N)
    for i in range(1,N): nw[i] = abs(np.dot(normals[i], normals[i-1]))

    gft_comp = GFTStrategyWraper(); struct_map = {}
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg); struct_map[b.block_id] = coeffs; b.clear_data()

    # Baseline rate
    sampler = Sampler(ratio=0.01, n_strata=5); sampled = sampler(pc.V, pc.A, blocks)
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    tr, tv2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tv2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs); tr += r; b.clear_data()
    base_rate = tr / tv2
    log(f"Baseline rate: {base_rate:.5f} bpv")

    # EM + Phase 2 cost matrix (shared for both strategies)
    log("Spherical EM init...")
    centroids_3d, _ = spherical_em(g3d_norm, K_hybrid)

    # Quick R-D refinement (2 iters)
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
    strat_idx = np.linspace(0, N-1, min(500, N), dtype=int)
    sample_labels = np.zeros(len(strat_idx), dtype=int)
    for it in range(2):
        tasks = []
        for idx in strat_idx:
            b = blocks[idx]; b.init_data(pc.V, pc.A)
            tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                          struct_map[b.block_id],
                          tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                          centroids_3d, slp, slw, g3d_raw[idx], dm, lag, q, K))
            b.clear_data()
        res = Parallel(n_jobs=-1)(delayed(_eval_block_row)(*t) for t in tasks)
        sc = np.array(res); nl = np.argmin(sc, axis=1)
        changes = (nl!=sample_labels).sum(); sample_labels = nl
        log(f"  R-D iter {it+1}/2: label changes = {changes}")
        for k_idx in range(K_hybrid):
            assigned = strat_idx[sample_labels==(k_idx+1)]
            if len(assigned)<2: continue
            sg = np.zeros(3); mu = centroids_3d[k_idx]
            for idx in assigned:
                sg += np.sign(np.dot(g3d_norm[idx], mu))*g3d_norm[idx]
            n = np.linalg.norm(sg)
            if n>1e-8: centroids_3d[k_idx] = sg/n

    log(f"Phase 2: evaluating {N}×{K} cost matrix...")
    t0 = time.time()
    tasks = []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                      struct_map[b.block_id],
                      tangent_data[i]["v1"], tangent_data[i]["v2"],
                      centroids_3d, slp, slw, g3d_raw[i], dm, lag, q, K))
        b.clear_data()
    results = Parallel(n_jobs=-1)(delayed(_eval_block_row)(*t) for t in tasks)
    cost_matrix = np.array(results)
    log(f"  Done in {time.time()-t0:.1f}s.")

    structural_cost = cost_matrix[:, 0].sum()
    p2_labels = np.argmin(cost_matrix, axis=1)
    log(f"  Phase 2 allocation: " +
        " | ".join(f"C{k}={int((p2_labels==k).sum())}({100*(p2_labels==k).mean():.0f}%)"
                   for k in range(K)))

    # ==========================================================================
    # Standard Viterbi baseline (current model)
    # ==========================================================================
    section("BASELINE: Standard Block-Level Viterbi")
    baseline_rows = {}
    for gamma in gammas:
        dp = np.full((N,K), np.inf); paths = np.zeros((N,K), dtype=int)
        dp[0] = cost_matrix[0]
        for i in range(1,N):
            w = nw[i]
            for k in range(K):
                tc = dp[i-1]+gamma*w*(np.arange(K)!=k)
                bp = np.argmin(tc); dp[i,k]=cost_matrix[i,k]+tc[bp]; paths[i,k]=bp
        vl = np.zeros(N, dtype=int); vl[N-1]=np.argmin(dp[N-1])
        for i in range(N-2,-1,-1): vl[i]=paths[i+1,vl[i+1]]
        gft_c = sum(cost_matrix[i,vl[i]] for i in range(N))
        sav = (structural_cost-gft_c)/structural_cost*100
        oh = markov_overhead(vl, K, N, total_v)
        net = base_rate*(sav/100)-oh
        sw = int((vl[1:]!=vl[:-1]).sum())
        rle_oh, n_runs, run_lens = rle_overhead(vl, K, total_v)
        log(f"  γ={gamma:.0f}: GFT={sav:.3f}% H_markov={oh:.5f} "
            f"H_rle={rle_oh:.5f} net_markov={net:+.5f} "
            f"switches={sw} runs={n_runs} mean_run={np.mean(run_lens):.1f}")
        baseline_rows[gamma] = (vl, sav, oh, net, sw, n_runs, rle_oh)

    # ==========================================================================
    # STRATEGY A: Forced Minimum Segment Size
    # ==========================================================================
    section("STRATEGY A: Forced Minimum Segment Size (Morton RLE)")
    print(f"\n{'γ':>7} | {'MinSeg':>6} | {'GFT%':>7} | {'Overhead':>10} | "
          f"{'Net BPV':>10} | {'Runs':>6} | {'ΔNet':>10}")
    print("-" * 72)
    for gamma in gammas:
        vl_base, sav_base, oh_base, net_base, sw_base, runs_base, rle_base = baseline_rows[gamma]
        print(f"{gamma:>7.0f} | {'  1 (base)':>6} | {sav_base:>6.3f}% | "
              f"{oh_base:>10.5f} | {net_base:>+10.5f} | {runs_base:>6} | {'---':>10}")
        for min_s in min_sizes:
            if min_s == 1: continue  # already printed
            vl_new, n_segs = force_min_segment(vl_base, cost_matrix, K, min_s)
            gft_c = sum(cost_matrix[i, vl_new[i]] for i in range(N))
            sav_new = (structural_cost - gft_c)/structural_cost*100
            oh_new = markov_overhead(vl_new, K, N, total_v)
            rle_oh_new, n_runs_new, _ = rle_overhead(vl_new, K, total_v)
            net_new = base_rate*(sav_new/100) - oh_new
            delta_net = net_new - net_base
            print(f"{gamma:>7.0f} | {min_s:>6} | {sav_new:>6.3f}% | "
                  f"{oh_new:>10.5f} | {net_new:>+10.5f} | {n_runs_new:>6} | "
                  f"{delta_net:>+10.5f}")
        print()

    # ==========================================================================
    # STRATEGY B: 3D Spatial Connected Components
    # ==========================================================================
    section("STRATEGY B: 3D Spatial Connected Components")

    # Estimate block spacing from coords
    log("Building 3D spatial adjacency graph...")
    t0 = time.time()
    dists_sample = []
    idx_s = np.random.choice(N, min(200, N), replace=False)
    for i in idx_s:
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        dists_sample.append(d.min())
    spacing = np.median(dists_sample)
    radius = spacing * 1.8  # neighbours within 1.8× typical spacing
    log(f"  Estimated block spacing: {spacing:.2f}, adjacency radius: {radius:.2f}")

    adj = build_adjacency_fast(coords, radius)
    n_edges = sum(len(v) for v in adj.values())//2
    log(f"  Adjacency: {n_edges} edges, avg degree {2*n_edges/N:.1f}, "
        f"built in {time.time()-t0:.1f}s")

    # Connected components on Phase 2 labels
    comp_id, comp_labels = connected_components_3d(p2_labels, adj, N)
    M = len(comp_labels)
    log(f"  Phase 2 connected components: {M} (vs {N} blocks) — "
        f"compression ratio {N/M:.1f}×")

    comp_sizes = np.bincount(comp_id)
    log(f"  Component size: min={comp_sizes.min()} mean={comp_sizes.mean():.1f} "
        f"median={int(np.median(comp_sizes))} max={comp_sizes.max()}")
    log(f"  Singleton components (size=1): "
        f"{(comp_sizes==1).sum()} ({100*(comp_sizes==1).mean():.0f}%)")

    # Build component cost matrix: average over blocks in each component
    comp_cost = np.zeros((M, K))
    for i in range(N):
        c = comp_id[i]
        comp_cost[c] += cost_matrix[i]
    for c in range(M):
        comp_cost[c] /= comp_sizes[c]

    # Component normal weights: mean normal for each component, then compute
    # |n_c · n_{c-1}| along Morton ordering of component sequence
    comp_normals = np.zeros((M, 3))
    for i in range(N):
        comp_normals[comp_id[i]] += normals[i]
    for c in range(M):
        n = np.linalg.norm(comp_normals[c])
        if n > 1e-8: comp_normals[c] /= n
        else: comp_normals[c] = np.array([0.,0.,1.])

    # Morton order of components: order by first block index in each component
    first_block = np.zeros(M, dtype=int)
    for i in range(N-1, -1, -1):
        first_block[comp_id[i]] = i
    comp_morton_order = np.argsort(first_block)
    comp_labels_ordered = comp_labels[comp_morton_order]
    comp_cost_ordered = comp_cost[comp_morton_order]
    comp_normals_ordered = comp_normals[comp_morton_order]
    comp_nw = np.ones(M)
    for i in range(1, M):
        comp_nw[i] = abs(np.dot(comp_normals_ordered[i], comp_normals_ordered[i-1]))

    print(f"\n{'γ':>7} | {'Unit':>10} | {'GFT%':>7} | {'Overhead':>10} | "
          f"{'Net BPV':>10} | {'Switches':>8} | {'ΔNet':>10}")
    print("-" * 76)

    for gamma in gammas:
        # Baseline block-level Viterbi
        vl_base, sav_base, oh_base, net_base, sw_base, runs_base, _ = baseline_rows[gamma]
        print(f"{gamma:>7.0f} | {'blocks(N='+str(N)+')':>10} | {sav_base:>6.3f}% | "
              f"{oh_base:>10.5f} | {net_base:>+10.5f} | {sw_base:>8} | {'---':>10}")

        # Component-level Viterbi
        vl_comp = viterbi_on_components(comp_labels_ordered, comp_cost_ordered,
                                         K, gamma, comp_nw)
        # Map component labels back to blocks
        block_labels = np.zeros(N, dtype=int)
        for i in range(N):
            c = comp_id[i]
            c_pos = np.where(comp_morton_order == c)[0]
            if len(c_pos) > 0:
                block_labels[i] = vl_comp[c_pos[0]]

        gft_c = sum(cost_matrix[i, block_labels[i]] for i in range(N))
        sav_new = (structural_cost - gft_c)/structural_cost*100
        oh_new = compute_entropy(vl_comp, K) * M / total_v
        net_new = base_rate*(sav_new/100) - oh_new
        sw_new = int((vl_comp[1:]!=vl_comp[:-1]).sum())
        delta_net = net_new - net_base
        print(f"{gamma:>7.0f} | {'comps(M='+str(M)+')':>10} | {sav_new:>6.3f}% | "
              f"{oh_new:>10.5f} | {net_new:>+10.5f} | {sw_new:>8} | "
              f"{delta_net:>+10.5f}")
        print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    section("KEY METRICS")
    log(f"Block count N={N}, Component count M={M}, N/M ratio={N/M:.1f}×")
    log(f"Singleton fraction: {100*(comp_sizes==1).mean():.0f}% — "
        f"{'HIGH: many isolated blocks, limited grouping benefit' if (comp_sizes==1).mean()>0.4 else 'OK: grouping is effective'}")
    log(f"Oracle H (sort by label): {compute_entropy(np.sort(p2_labels), K):.4f}")
    log(f"Morton H (current): {compute_entropy(p2_labels, K):.4f}")
    log(f"Component-sequence H: {compute_entropy(comp_labels_ordered, K):.4f}")
    delta_oh = (compute_entropy(p2_labels, K) * N -
                compute_entropy(comp_labels_ordered, K) * M) / total_v
    log(f"Overhead reduction from components: {delta_oh:.5f} bpv")


if __name__ == "__main__":
    main()
