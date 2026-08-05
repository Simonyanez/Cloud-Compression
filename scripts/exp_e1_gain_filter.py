"""
Experiment E1: Gain-Filter Centroid Refinement
===============================================
Core idea (Deterministic Annealing / Hard-Negative Mining):
  Directional centroids are currently trained on ALL blocks including those
  where no directional transform outperforms the structural baseline. These
  low-gain blocks dilute the centroids — pulling them toward directions that
  don't actually compress well. Masking them out forces centroids to specialize
  on only the blocks that genuinely need a directional transform.

Per-block gain:
  gain_i (%) = (Cost_struct(i) - min_{k>0} Cost_GFT(i,k)) / Cost_struct(i) * 100

Active mask:
  M_i^(tau) = 1 if gain_i > tau

Two-phase execution:
  Phase 1 (instant): analytical sweep over tau using the existing cost matrix.
    Reports gain distribution, active block fractions, centroid separation
    WITHOUT re-running any GFT evaluations.
  Phase 2 (retraining): for tau in {0.5, 1.0, 2.0}%, retrain centroids on
    active blocks only, re-evaluate full Phase 2 cost matrix, run Viterbi,
    report Net BPV vs baseline.

Runtime: ~6-8 min for B=16, K=4 (1 cost matrix build + 3 retrain evaluations).
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project_slope(mu, v1, v2, unc_norm, degen=1e-3):
    s = np.dot(mu, v1)*v1 + np.dot(mu, v2)*v2
    rn = np.linalg.norm(s)
    if rn < degen: s = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s/rn)*t


def spherical_kmeans(directions, K, max_iter=40, seed=42):
    """Spherical k-means on unit directions. Returns centroids, labels."""
    np.random.seed(seed)
    if len(directions) < K:
        # Pad with random unit vectors if too few blocks
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
    """Mean absolute pairwise dot product. 0=orthogonal (best), 1=identical (worst)."""
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


# ---------------------------------------------------------------------------
# R-D evaluation worker
# ---------------------------------------------------------------------------

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


def grid_search(blocks, tangent_data, centroids_3d, g3d_raw, g3d_norm, pc_V, pc_A,
                dm, lag, q, K, active_mask):
    K_hybrid = K - 1
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
    p_grid = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    w_grid = [0.25, 0.50, 0.80, 1.00, 1.40, 1.80]

    # Assign active blocks to their closest directional centroid
    active_indices = np.where(active_mask)[0]
    if len(active_indices) > 0:
        active_grads = g3d_norm[active_indices]
        sims = np.abs(active_grads @ centroids_3d.T) # (N_active, K_hybrid)
        assigned_cluster = np.argmax(sims, axis=1) # (N_active,)
    else:
        assigned_cluster = np.array([], dtype=int)

    # Hoist common decider, gft, app objects outside loops
    dec = Decider(dm, lag); dec._set_vars(q)
    gft0 = GFTStrategyWraper(); app0 = Approximator()

    for k in range(1, K):
        c_idx = k - 1
        # Get sample blocks assigned specifically to cluster k (c_idx)
        k_active = active_indices[assigned_cluster == c_idx]
        if len(k_active) < 3:
            # Fallback: take top active blocks by directional similarity to centroid c_idx
            if len(active_indices) > 0:
                sim_scores = sims[:, c_idx]
                top_idx = np.argsort(sim_scores)[::-1][:min(80, len(active_indices))]
                k_active = active_indices[top_idx]
            else:
                k_active = np.array([], dtype=int)

        sample_blocks = []
        for i in k_active[:min(80, len(k_active))]:
            b = blocks[i]; b.init_data(pc_V, pc_A)
            sample_blocks.append((b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                                  tangent_data[i]["v1"], tangent_data[i]["v2"],
                                  g3d_raw[i]))
            b.clear_data()

        if len(sample_blocks) < 3:
            continue

        best_cost, best_p, best_w = 1e18, slp[k], slw[k]
        for p in p_grid:
            for w in w_grid:
                costs_kw = []
                for Vb, Ab, meta, v1, v2, unc in sample_blocks:
                    s3d = project_slope(centroids_3d[c_idx], v1, v2, unc)
                    sgp = StructuralGraph(meta); sgp.set_data(Vb)
                    ag = AttributeGraph(sgp, s3d, k, p, w)
                    Vr = app0._spatial_norm(Vb); Aa = Ab.copy(); Aa[:,0] = Vr@s3d.T
                    ag.set_data(Vb, Aa)
                    b_tmp = Block(meta); b_tmp.Vblock = Vb; b_tmp.Ablock = Ab
                    try:
                        _, coeffs = gft0(b_tmp, ag)
                        c, _, _ = dec._RDcost(coeffs)
                        costs_kw.append(c)
                    except:
                        costs_kw.append(1e18)
                cs = sum(costs_kw)
                if cs < best_cost: best_cost, best_p, best_w = cs, p, w
        slp[k], slw[k] = best_p, best_w
        log(f"    k={k}: slp={best_p:.2f} slw={best_w:.2f}")
    return slp, slw


def run_viterbi(cost_matrix, K, gamma, nw):
    N = len(cost_matrix)
    dp = np.full((N, K), np.inf); paths = np.zeros((N,K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i-1]+gamma*w*(np.arange(K)!=k)
            bp = np.argmin(tc); dp[i,k]=cost_matrix[i,k]+tc[bp]; paths[i,k]=bp
    vl = np.zeros(N, dtype=int); vl[N-1]=np.argmin(dp[N-1])
    for i in range(N-2,-1,-1): vl[i]=paths[i+1,vl[i+1]]
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
    p.add_argument("--tau-retrain", type=str, default="0.5,1.0,2.0",
                   help="Gain thresholds %% to retrain centroids for (comma-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    B, K, q = args.block_size, args.k, args.q_step
    K_hybrid = K - 1
    gammas = [float(x) for x in args.gammas.split(",")]
    tau_retrain = [float(x) for x in args.tau_retrain.split(",")]

    section(f"E1: Gain-Filter Centroid Refinement  (B={B}, K={K})")

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
    normals = np.array(normals)
    g3d_norm = np.array(g3d_norm)
    g3d_raw = np.array(g3d_raw)

    nw = np.ones(N)
    for i in range(1,N): nw[i] = abs(np.dot(normals[i], normals[i-1]))

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

    # -------------------------------------------------------------------------
    # Baseline EM + Phase 2
    # -------------------------------------------------------------------------
    log("Baseline EM init...")
    np.random.seed(42)
    idx0 = np.random.choice(N, K_hybrid, replace=False)
    centroids_base = g3d_norm[idx0].copy()
    for _ in range(20):
        sims = np.abs(g3d_norm @ centroids_base.T)
        lbl = np.argmax(sims, axis=1)
        for k in range(K_hybrid):
            m = (lbl==k); sg = np.zeros(3); mu = centroids_base[k]
            for i in np.where(m)[0]: sg += np.sign(np.dot(g3d_norm[i], mu))*g3d_norm[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids_base[k] = sg/n

    slp_base = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw_base = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    log("Building baseline Phase 2 cost matrix...")
    t0 = time.time()
    cost_matrix_base = build_cost_matrix(blocks, tangent_data, struct_map,
                                          centroids_base, slp_base, slw_base,
                                          g3d_raw, pc.V, pc.A, dm, lag, q, K)
    log(f"  Done in {time.time()-t0:.1f}s.")

    struct_costs = cost_matrix_base[:, 0]
    structural_total = struct_costs.sum()

    # Per-block gains (%) against best directional cluster
    best_directional_cost = cost_matrix_base[:, 1:].min(axis=1)
    gains_pct = (struct_costs - best_directional_cost) / (struct_costs + 1e-12) * 100
    # Negative gain = structural is cheaper

    sep_base = centroid_separation(centroids_base)

    # =========================================================================
    # PHASE 1: Analytical Gain Sweep (instant, no re-evaluation)
    # =========================================================================
    section("PHASE 1 — Analytical Gain Distribution (existing cost matrix)")

    log(f"Baseline centroid separation (lower=more distinct): {sep_base:.4f}")
    log(f"\nPer-block gain distribution:")
    for pct in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        log(f"  P{pct:>3}: {np.percentile(gains_pct, pct):+.2f}%")

    n_negative = (gains_pct <= 0).sum()
    log(f"\nBlocks where structural IS cheaper (gain <= 0): "
        f"{n_negative} ({100*n_negative/N:.1f}%)")
    log(f"Blocks where directional IS cheaper (gain > 0): "
        f"{N-n_negative} ({100*(N-n_negative)/N:.1f}%)")

    log("\nActive-block count at each tau threshold:")
    taus_analytical = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f"\n  {'τ%':>6} | {'Active':>6} | {'Active%':>8} | "
          f"{'Mean Gain(Active)':>18} | {'P25 Gain':>9} | {'P75 Gain':>9}")
    print("  " + "-"*65)
    for tau in taus_analytical:
        mask = gains_pct > tau
        n_active = mask.sum()
        if n_active > 0:
            mg = gains_pct[mask].mean()
            p25 = np.percentile(gains_pct[mask], 25)
            p75 = np.percentile(gains_pct[mask], 75)
        else:
            mg, p25, p75 = 0, 0, 0
        print(f"  {tau:>6.1f} | {n_active:>6} | {100*n_active/N:>7.1f}% | "
              f"{mg:>+17.2f}% | {p25:>+8.2f}% | {p75:>+8.2f}%")

    # Baseline Viterbi at each gamma
    log("\nBaseline Net BPV (no gain filtering):")
    baseline_results = {}
    for gamma in gammas:
        vl = run_viterbi(cost_matrix_base, K, gamma, nw)
        gft_c = sum(cost_matrix_base[i, vl[i]] for i in range(N))
        sav = (structural_total - gft_c)/structural_total*100
        H = compute_entropy(vl, K)
        oh = H*N/total_v
        net = base_rate*(sav/100) - oh
        sw = int((vl[1:]!=vl[:-1]).sum())
        baseline_results[gamma] = (sav, H, oh, net, sw)
        log(f"  γ={gamma:.0f}: GFT={sav:.3f}% H={H:.4f} overhead={oh:.5f} "
            f"net={net:+.5f} switches={sw}")

    # =========================================================================
    # PHASE 2: Retrain Centroids with Gain Filter and Re-Evaluate
    # =========================================================================
    section("PHASE 2 — Retrain Centroids on High-Gain Blocks Only")

    all_results = []  # (tau, gamma, sav, H, oh, net, sw, n_active, sep)

    for tau in tau_retrain:
        active_mask = gains_pct > tau
        n_active = active_mask.sum()
        log(f"\n--- τ = {tau:.1f}% | Active blocks: {n_active} ({100*n_active/N:.1f}%) ---")

        if n_active < K_hybrid * 3:
            log(f"  SKIP: fewer than {K_hybrid*3} active blocks, centroids would be degenerate.")
            continue

        # Retrain centroids on active blocks only
        active_grads = g3d_norm[active_mask]
        log(f"  Retraining {K_hybrid} centroids on {len(active_grads)} active blocks...")
        centroids_new, _ = spherical_kmeans(active_grads, K_hybrid)

        sep_new = centroid_separation(centroids_new)
        log(f"  Centroid separation (lower=more distinct): "
            f"baseline={sep_base:.4f} → filtered={sep_new:.4f} "
            f"(Δ = {sep_new - sep_base:+.4f})")

        # Grid search SLP/SLW on active blocks
        log("  Grid search on active blocks...")
        slp_new, slw_new = grid_search(blocks, tangent_data, centroids_new, g3d_raw, g3d_norm,
                                       pc.V, pc.A, dm, lag, q, K, active_mask)

        # Rebuild full Phase 2 cost matrix with new centroids
        log("  Building new Phase 2 cost matrix...")
        t0 = time.time()
        cost_matrix_new = build_cost_matrix(blocks, tangent_data, struct_map,
                                             centroids_new, slp_new, slw_new,
                                             g3d_raw, pc.V, pc.A, dm, lag, q, K)
        log(f"  Done in {time.time()-t0:.1f}s.")

        structural_total_new = cost_matrix_new[:, 0].sum()

        # GFT savings on active blocks specifically
        best_new_active = cost_matrix_new[active_mask, 1:].min(axis=1)
        gains_active = (cost_matrix_new[active_mask, 0] - best_new_active)
        gains_active_pct = gains_active / (cost_matrix_new[active_mask, 0] + 1e-12) * 100
        log(f"  Mean gain on active blocks with NEW centroids: {gains_active_pct.mean():+.2f}%")

        # Compare gain vs baseline on same active blocks
        gains_base_on_active = gains_pct[active_mask]
        log(f"  Mean gain on active blocks with BASE centroids: {gains_base_on_active.mean():+.2f}%")
        log(f"  Gain improvement on active blocks: {gains_active_pct.mean() - gains_base_on_active.mean():+.2f}%")

        # P2 label distribution with new centroids
        p2_new = np.argmin(cost_matrix_new, axis=1)
        log(f"  P2 allocation: " +
            " | ".join(f"C{k}={int((p2_new==k).sum())}({100*(p2_new==k).mean():.0f}%)"
                       for k in range(K)))

        # Viterbi at each gamma
        for gamma in gammas:
            vl = run_viterbi(cost_matrix_new, K, gamma, nw)
            gft_c = sum(cost_matrix_new[i, vl[i]] for i in range(N))
            sav = (structural_total_new - gft_c)/structural_total_new*100
            H = compute_entropy(vl, K)
            oh = H*N/total_v
            net = base_rate*(sav/100) - oh
            sw = int((vl[1:]!=vl[:-1]).sum())
            b_sav, b_H, b_oh, b_net, b_sw = baseline_results[gamma]
            delta_net = net - b_net
            log(f"  γ={gamma:.0f}: GFT={sav:.3f}% H={H:.4f} oh={oh:.5f} "
                f"net={net:+.5f} Δnet={delta_net:+.5f} sw={sw}")
            all_results.append((tau, gamma, sav, H, oh, net, sw, n_active, sep_new))

    # =========================================================================
    # Final Comparison Table
    # =========================================================================
    section("RESULTS COMPARISON TABLE")
    print(f"\n{'τ%':>6} | {'γ':>6} | {'Active%':>7} | {'GFT%':>7} | "
          f"{'Overhead':>10} | {'Net BPV':>10} | {'ΔNet':>10} | {'CentSep':>8}")
    print("-" * 80)
    for gamma in gammas:
        b_sav, b_H, b_oh, b_net, b_sw = baseline_results[gamma]
        print(f"{'0.0':>6} | {gamma:>6.0f} | {'100.0%':>7} | {b_sav:>6.3f}% | "
              f"{b_oh:>10.5f} | {b_net:>+10.5f} | {'---':>10} | {sep_base:>8.4f}")
    for row in all_results:
        tau, gamma, sav, H, oh, net, sw, n_active, sep = row
        b_sav, b_H, b_oh, b_net, b_sw = baseline_results[gamma]
        delta_net = net - b_net
        print(f"{tau:>6.1f} | {gamma:>6.0f} | {100*n_active/N:>6.1f}% | {sav:>6.3f}% | "
              f"{oh:>10.5f} | {net:>+10.5f} | {delta_net:>+10.5f} | {sep:>8.4f}")

    section("KEY DIAGNOSTIC: Does filtering sharpen centroids?")
    for row in all_results:
        tau, gamma, *_ , n_active, sep = row
        if gamma == gammas[1]:  # middle gamma only
            log(f"  τ={tau:.1f}%: Centroid Separation baseline={sep_base:.4f} → filtered={sep:.4f} "
                f"({'MORE DISTINCT ✓' if sep < sep_base else 'LESS DISTINCT ✗'})")


if __name__ == "__main__":
    main()
