"""
Experiment 1: R-D Initialized Clustering (No Geometric EM)
===========================================================
Hypothesis: geometric EM (spherical k-means on gradient directions) produces
incoherent cluster initialisation — the first R-D refinement iteration flips
76% of labels, meaning the EM starting point is nearly useless.

Replace Phase 1 with:
  1. Generate M probe directions uniformly on the sphere (Fibonacci lattice)
  2. Evaluate a sample of blocks against all M probes using actual R-D cost
  3. For each sampled block, record the probe direction that minimises R-D cost
  4. Run spherical k-means on those R-D-optimal directions → K-1 centroids
  5. Proceed with Phase 1.5 R-D refinement and Phase 2/3 as normal

Compare side-by-side with the geometric EM baseline (same R-D refinement,
same grid search, same Viterbi settings).

Run time: ~5-10 min for B=16, K=4, M=16 probes, 600 sample blocks.
"""

import sys, os, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
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
# Direction utilities
# ---------------------------------------------------------------------------

def fibonacci_sphere(n):
    """n approximately uniformly distributed unit directions on the sphere."""
    phi = (1 + np.sqrt(5)) / 2
    dirs = []
    for i in range(n):
        theta = np.arccos(1 - 2*(i+0.5)/n)
        psi = 2*np.pi*i/phi
        dirs.append([np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)])
    return np.array(dirs)


def spherical_kmeans(directions, K, max_iter=30, seed=42):
    """Spherical k-means on unit-norm direction vectors."""
    np.random.seed(seed)
    idx = np.random.choice(len(directions), K, replace=False)
    centroids = directions[idx].copy()
    labels = np.zeros(len(directions), dtype=int)
    for _ in range(max_iter):
        prev = labels.copy()
        sims = np.abs(directions @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev):
            break
        for k in range(K):
            m = (labels == k)
            if m.sum() == 0:
                continue
            sg = directions[m].sum(axis=0)
            n = np.linalg.norm(sg)
            if n > 1e-8:
                centroids[k] = sg / n
    return centroids, labels


# ---------------------------------------------------------------------------
# Project global centroid to local slope (with magnitude normalisation)
# ---------------------------------------------------------------------------

def project_slope(mu_k, v1, v2, unconstrained_norm, degen_thresh=1e-3):
    s3d_raw = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
    rn = np.linalg.norm(s3d_raw)
    if rn < degen_thresh:
        s3d_raw = v1 * degen_thresh
        rn = degen_thresh
    target = unconstrained_norm if unconstrained_norm > 1e-8 else rn
    return (s3d_raw / rn) * target


# ---------------------------------------------------------------------------
# R-D cost workers
# ---------------------------------------------------------------------------

def _eval_block_probe(Vblock, Ablock, metadata, v1, v2, probe_dir,
                      unconstrained_norm, slp, slw, k_cluster,
                      decider_mode, lagrange, q_step):
    """R-D cost for one block under one probe direction."""
    dec = Decider(decider_mode, lagrange); dec._set_vars(q_step)
    gft = GFTStrategyWraper(); app = Approximator()
    b = Block(metadata); b.Vblock = Vblock; b.Ablock = Ablock
    s3d = project_slope(probe_dir, v1, v2, unconstrained_norm)
    sg = StructuralGraph(metadata); sg.set_data(Vblock)
    ag = AttributeGraph(sg, s3d, k_cluster, slp, slw)
    Vr = app._spatial_norm(Vblock)
    Aa = Ablock.copy(); Aa[:, 0] = Vr @ s3d.T
    ag.set_data(Vblock, Aa)
    try:
        _, coeffs = gft(b, ag); c, _, _ = dec._RDcost(coeffs); return c
    except Exception:
        return 1e18


def _eval_block_full_row(block_id, Vblock, Ablock, metadata, struct_coeffs,
                          v1, v2, centroids_3d, slp, slw,
                          unconstrained_norm, decider_mode, lagrange, q_step, K):
    dec = Decider(decider_mode, lagrange); dec._set_vars(q_step)
    gft = GFTStrategyWraper(); app = Approximator()
    b = Block(metadata); b.Vblock = Vblock; b.Ablock = Ablock
    costs = np.full(K, np.inf)
    try: costs[0] = dec._RDcost(struct_coeffs)[0]
    except: pass
    for k in range(1, K):
        s3d = project_slope(centroids_3d[k-1], v1, v2, unconstrained_norm)
        sg = StructuralGraph(metadata); sg.set_data(Vblock)
        ag = AttributeGraph(sg, s3d, k, slp[k], slw[k])
        Vr = app._spatial_norm(Vblock); Aa = Ablock.copy(); Aa[:,0] = Vr @ s3d.T
        ag.set_data(Vblock, Aa)
        try:
            _, coeffs = gft(b, ag); c, _, _ = dec._RDcost(coeffs); costs[k] = c
        except: pass
    return costs


# ---------------------------------------------------------------------------
# Geometric EM baseline (for comparison)
# ---------------------------------------------------------------------------

def geometric_em_init(g3d_norm, K_hybrid):
    """Standard spherical EM on unconstrained gradient directions."""
    np.random.seed(42)
    centroids = g3d_norm[np.random.choice(len(g3d_norm), K_hybrid, replace=False)].copy()
    labels = np.zeros(len(g3d_norm), dtype=int)
    for _ in range(20):
        prev = labels.copy()
        sims = np.abs(g3d_norm @ centroids.T)
        labels = np.argmax(sims, axis=1)
        if np.all(labels == prev): break
        for k in range(K_hybrid):
            m = (labels == k)
            if m.sum() == 0: continue
            sg = sum(np.sign(np.dot(g3d_norm[i], centroids[k]))*g3d_norm[i]
                     for i in np.where(m)[0])
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n
    return centroids, labels


# ---------------------------------------------------------------------------
# Shared: R-D refinement + grid search + Phase 2 + Viterbi
# ---------------------------------------------------------------------------

def rd_refine_and_score(tag, centroids_3d, ctx, K, decider_mode, lagrange, q_step,
                         gammas, base_rate):
    blocks = ctx["blocks"]; N = ctx["N"]; total_v = ctx["total_v"]
    tangent_data = ctx["tangent_data"]
    struct_map = ctx["structural_coeffs_map"]
    g3d_raw = ctx["g3d_raw_norms"]
    g3d_norm = ctx["g3d_norm"]
    normal_weights = ctx["normal_weights"]
    K_hybrid = K - 1

    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.45)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 1.5)])

    # ---- R-D refinement (3 iterations, 600 sample blocks) ----
    strat_idx = np.linspace(0, N-1, min(600, N), dtype=int)
    sample_labels = np.zeros(len(strat_idx), dtype=int)  # init all flat

    log(f"  [{tag}] R-D refinement (3 iters, {len(strat_idx)} blocks)...")
    for rd_iter in range(3):
        tasks = []
        for idx in strat_idx:
            b = blocks[idx]; b.init_data(ctx["pc_V"], ctx["pc_A"])
            tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                          struct_map[b.block_id],
                          tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                          centroids_3d, slp, slw,
                          g3d_raw[idx], decider_mode, lagrange, q_step, K))
            b.clear_data()
        results = Parallel(n_jobs=-1)(delayed(_eval_block_full_row)(*t) for t in tasks)
        sample_costs = np.array(results)
        new_labels = np.argmin(sample_costs, axis=1)
        changes = (new_labels != sample_labels).sum()
        sample_labels = new_labels
        log(f"    iter {rd_iter+1}/3: label changes = {changes}/{len(strat_idx)}")

        for k_idx in range(K_hybrid):
            k = k_idx + 1
            assigned = strat_idx[sample_labels == k]
            if len(assigned) >= 2:
                sg = np.zeros(3); mu = centroids_3d[k_idx]
                for idx in assigned:
                    g = g3d_norm[idx]
                    sg += np.sign(np.dot(g, mu)) * g
                n = np.linalg.norm(sg)
                if n > 1e-8: centroids_3d[k_idx] = sg/n

    # ---- Simplified grid search ----
    log(f"  [{tag}] Grid search (SLP × SLW)...")
    p_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]
    w_grid = [0.5, 0.8, 1.0, 1.4, 1.8, 2.6]
    for k in range(1, K):
        c_idx = k - 1
        assigned = strat_idx[sample_labels == k]
        if len(assigned) < 4:
            log(f"    Cluster {k}: too few R-D blocks ({len(assigned)}) — skipping.")
            continue
        sdata = []
        for idx in assigned[:min(150, len(assigned))]:
            b = blocks[idx]; b.init_data(ctx["pc_V"], ctx["pc_A"])
            sdata.append((b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                          tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                          g3d_raw[idx]))
            b.clear_data()
        best_cost, best_p, best_w = 1e18, slp[k], slw[k]
        for p in p_grid:
            for w in w_grid:
                tasks = [(Vb, Ab, meta, v1, v2, centroids_3d[c_idx],
                          uncn, p, w, k, decider_mode, lagrange, q_step)
                         for Vb, Ab, meta, v1, v2, uncn in sdata]
                costs = Parallel(n_jobs=-1)(
                    delayed(_eval_block_probe)(*t) for t in tasks)
                cs = sum(costs)
                if cs < best_cost: best_cost, best_p, best_w = cs, p, w
        slp[k], slw[k] = best_p, best_w
        mu = centroids_3d[c_idx]
        log(f"    Cluster {k}: [{mu[0]:+.3f},{mu[1]:+.3f},{mu[2]:+.3f}] "
            f"slp={best_p:.2f} slw={best_w:.2f} ({len(sdata)} blocks)")

    # ---- Phase 2: full cost matrix ----
    log(f"  [{tag}] Phase 2: full {N}×{K} cost matrix...")
    t0 = time.time()
    tasks = []
    for i, b in enumerate(blocks):
        b.init_data(ctx["pc_V"], ctx["pc_A"])
        tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                      struct_map[b.block_id],
                      tangent_data[i]["v1"], tangent_data[i]["v2"],
                      centroids_3d, slp, slw,
                      g3d_raw[i], decider_mode, lagrange, q_step, K))
        b.clear_data()
    cost_rows = Parallel(n_jobs=-1)(delayed(_eval_block_full_row)(*t) for t in tasks)
    cost_matrix = np.array(cost_rows)
    log(f"    Done in {time.time()-t0:.1f}s.")

    structural_cost = cost_matrix[:, 0].sum()
    p2_labels = np.argmin(cost_matrix, axis=1)
    counts = np.bincount(p2_labels, minlength=K)
    log(f"    Phase 2 allocation: " +
        " | ".join(f"C{k}={counts[k]}({100*counts[k]/N:.0f}%)" for k in range(K)))

    # ---- Phase 3: Viterbi sweep ----
    def compute_entropy(labels):
        cs = np.bincount(labels, minlength=K)
        tr = np.ones((K,K))
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

    rows = []
    for gamma in gammas:
        dp = np.full((N, K), np.inf); paths = np.zeros((N,K), dtype=int)
        dp[0] = cost_matrix[0]
        for i in range(1, N):
            w = normal_weights[i]
            for k in range(K):
                tc = dp[i-1] + gamma*w*(np.arange(K)!=k)
                bp = np.argmin(tc)
                dp[i,k] = cost_matrix[i,k] + tc[bp]; paths[i,k] = bp
        vl = np.zeros(N, dtype=int); vl[N-1] = np.argmin(dp[N-1])
        for i in range(N-2,-1,-1): vl[i] = paths[i+1,vl[i+1]]
        gft_cost = sum(cost_matrix[i,vl[i]] for i in range(N))
        sav = (structural_cost - gft_cost)/structural_cost*100
        H = compute_entropy(vl)
        oh = H*N/total_v
        net = base_rate*(sav/100) - oh
        sw = int((vl[1:]!=vl[:-1]).sum())
        rows.append((gamma, sav, H, oh, net, sw))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--n-probes", type=int, default=16,
                   help="Number of Fibonacci sphere probe directions. Default: 16")
    p.add_argument("--n-sample", type=int, default=600,
                   help="Blocks to probe for R-D init. Default: 600")
    p.add_argument("--gammas", type=str, default="1000,2000,4000")
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q = args.block_size, args.k, args.q_step
    gammas = [float(x) for x in args.gammas.split(",")]
    K_hybrid = K - 1

    section(f"EXP 1: R-D Initialized Clustering  (B={B}, K={K}, M={args.n_probes} probes)")

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

    log("Extracting tangent planes & geometry...")
    tangent_data = extract_block_tangent_planes(blocks, pc.V, pc.A)
    normals, g3d_norm, g3d_raw, coords = [], [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        coords.append(np.mean(b.Vblock, axis=0))
        normals.append(tangent_data[i]["v3"])
        Vc = b.Vblock - b.Vblock.mean(0)
        X = np.column_stack((Vc @ tangent_data[i]["v1"], Vc @ tangent_data[i]["v2"]))
        Y = b.Ablock[:,0] - b.Ablock[:,0].mean()
        s2d, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        g = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        rn = np.linalg.norm(g)
        g3d_raw.append(rn)
        g3d_norm.append(g/rn if rn > 1e-8 else np.array([0.,0.,1.]))
        b.clear_data()
    normals = np.array(normals); g3d_norm = np.array(g3d_norm); g3d_raw = np.array(g3d_raw)

    nw = np.ones(N)
    for i in range(1, N): nw[i] = abs(np.dot(normals[i], normals[i-1]))

    log("Pre-computing structural GFT coefficients...")
    gft_comp = GFTStrategyWraper(); struct_map = {}
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg)
        struct_map[b.block_id] = coeffs; b.clear_data()

    # Baseline rate
    sampler = Sampler(ratio=0.01, n_strata=5)
    sampled = sampler(pc.V, pc.A, blocks)
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    tot_r, tot_v2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tot_v2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs)
        tot_r += r; b.clear_data()
    base_rate = tot_r / tot_v2
    log(f"Baseline rate: {base_rate:.5f} bpv")

    ctx = {"blocks": blocks, "N": N, "total_v": total_v,
           "tangent_data": tangent_data, "structural_coeffs_map": struct_map,
           "g3d_norm": g3d_norm, "g3d_raw_norms": g3d_raw,
           "normal_weights": nw, "pc_V": pc.V, "pc_A": pc.A}

    # ==========================================================================
    # BASELINE: Geometric EM
    # ==========================================================================
    section("BASELINE: Geometric EM Initialisation")
    log("Running spherical EM on unconstrained gradient directions...")
    t0 = time.time()
    geo_centroids, geo_em_labels = geometric_em_init(g3d_norm, K_hybrid)
    log(f"  EM done in {time.time()-t0:.1f}s. "
        f"Cluster sizes: {[int((geo_em_labels==k).sum()) for k in range(K_hybrid)]}")
    log("  Within-cluster gradient angle (should be low for tight clusters):")
    for k in range(K_hybrid):
        idx_k = np.where(geo_em_labels==k)[0]
        if len(idx_k) < 2: continue
        s = idx_k[:min(300, len(idx_k))]
        dots = np.abs(g3d_norm[s] @ g3d_norm[s].T); np.fill_diagonal(dots, np.nan)
        mean_a = np.degrees(np.arccos(np.clip(np.nanmean(dots), 0, 1)))
        log(f"    Centroid {k+1}: within-cluster gradient angle = {mean_a:.1f}°")

    geo_results = rd_refine_and_score(
        "GeoEM", geo_centroids.copy(), ctx, K, dm, lag, q, gammas, base_rate)

    # ==========================================================================
    # EXP 1: R-D Probe Initialisation
    # ==========================================================================
    section(f"EXP 1: R-D Probe Initialisation (M={args.n_probes} directions)")

    probe_dirs = fibonacci_sphere(args.n_probes)
    log(f"Generated {len(probe_dirs)} probe directions (Fibonacci sphere).")

    # Sample blocks for probing
    sample_idx = np.linspace(0, N-1, min(args.n_sample, N), dtype=int)
    log(f"Probing {len(sample_idx)} blocks × {args.n_probes} directions "
        f"= {len(sample_idx)*args.n_probes} evaluations...")

    t0 = time.time()
    # For each sampled block, evaluate all probes in parallel
    tasks = []
    for idx in sample_idx:
        b = blocks[idx]; b.init_data(pc.V, pc.A)
        for m, probe in enumerate(probe_dirs):
            # Use cluster index 1 (any non-zero for cost eval, doesn't affect direction)
            tasks.append((b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                          tangent_data[idx]["v1"], tangent_data[idx]["v2"],
                          probe, g3d_raw[idx],
                          0.30, 1.0, 1, dm, lag, q))
        b.clear_data()

    all_costs = Parallel(n_jobs=-1)(delayed(_eval_block_probe)(*t) for t in tasks)
    # Reshape: [n_sample, n_probes]
    probe_costs = np.array(all_costs).reshape(len(sample_idx), args.n_probes)
    log(f"  Probe evaluation done in {time.time()-t0:.1f}s.")

    # Best probe direction per block
    best_probe_idx = np.argmin(probe_costs, axis=1)
    best_probe_dirs = probe_dirs[best_probe_idx]

    # Concentration: how many distinct probes are "best"?
    unique_best, unique_counts = np.unique(best_probe_idx, return_counts=True)
    log(f"  Best-probe distribution: {len(unique_best)}/{args.n_probes} probes "
        f"were ever best-in-class.")
    log(f"  Top-3 most-winning probes: " +
        str(sorted(zip(unique_counts.tolist(), unique_best.tolist()), reverse=True)[:3]))

    # Within-cluster angle of best-probe directions (before k-means)
    log(f"  Within-group gradient angle of best-probe directions:")
    for m in unique_best:
        m_mask = (best_probe_idx == m)
        dirs_m = best_probe_dirs[m_mask]
        if len(dirs_m) < 2: continue
        dots = np.abs(dirs_m @ dirs_m.T); np.fill_diagonal(dots, np.nan)
        mean_a = np.degrees(np.arccos(np.clip(np.nanmean(dots), 0, 1)))
        log(f"    Probe {m}: {m_mask.sum()} blocks won, within-group angle = {mean_a:.1f}°")

    # Spherical k-means on best-probe directions → K_hybrid centroids
    log(f"  Running spherical k-means on best-probe directions (K={K_hybrid})...")
    rd_centroids, rd_km_labels = spherical_kmeans(best_probe_dirs, K_hybrid)

    log(f"  R-D init cluster sizes: "
        f"{[int((rd_km_labels==k).sum()) for k in range(K_hybrid)]}")
    for k in range(K_hybrid):
        mu = rd_centroids[k]
        log(f"  Centroid {k+1}: [{mu[0]:+.3f},{mu[1]:+.3f},{mu[2]:+.3f}]")

    log("  Within-cluster gradient angle for R-D init clusters:")
    # Map sample best-probe directions to full population via nearest centroid
    full_sims = np.abs(g3d_norm @ rd_centroids.T)
    full_rd_labels = np.argmax(full_sims, axis=1)
    for k in range(K_hybrid):
        idx_k = np.where(full_rd_labels == k)[0]
        if len(idx_k) < 2: continue
        s = idx_k[:min(300, len(idx_k))]
        dots = np.abs(g3d_norm[s] @ g3d_norm[s].T); np.fill_diagonal(dots, np.nan)
        mean_a = np.degrees(np.arccos(np.clip(np.nanmean(dots), 0, 1)))
        log(f"    Centroid {k+1}: within-cluster gradient angle = {mean_a:.1f}°")

    rd_results = rd_refine_and_score(
        "RD-Init", rd_centroids.copy(), ctx, K, dm, lag, q, gammas, base_rate)

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================
    section("RESULTS COMPARISON")
    header = f"{'γ':>7} | {'Method':<12} | {'GFT%':>7} | {'H':>7} | {'Overhead':>10} | {'Net BPV':>10} | {'Switches':>8}"
    print(header)
    print("-" * len(header))
    for (r_geo, r_rd) in zip(geo_results, rd_results):
        g = r_geo[0]
        print(f"{g:>7.0f} | {'GeoEM':<12} | {r_geo[1]:>6.3f}% | "
              f"{r_geo[2]:>7.4f} | {r_geo[3]:>10.5f} | {r_geo[4]:>+10.5f} | {r_geo[5]:>8}")
        print(f"{g:>7.0f} | {'RD-Probe':<12} | {r_rd[1]:>6.3f}% | "
              f"{r_rd[2]:>7.4f} | {r_rd[3]:>10.5f} | {r_rd[4]:>+10.5f} | {r_rd[5]:>8}")
        delta_net = r_rd[4] - r_geo[4]
        winner = "RD-Probe ✓" if delta_net > 0 else "GeoEM ✓"
        print(f"{'':>7}   {'Δ Net BPV':<12}   {delta_net:>+10.5f}  → {winner}")
        print()


if __name__ == "__main__":
    main()
