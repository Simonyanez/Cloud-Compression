"""
Experiment E3: Hierarchical Decoupled GFT with Iterative Re-Masking (EM Refinement)
====================================================================================
Scientific Motivation:
  1. Top-Down Arm 1 Execution Strategy:
     - Mode selection/Viterbi path optimization at parent level B=32 (N ~ 834).
     - GFT execution on child subblocks: B=16 (up to 8/parent) and B=8 (up to 64/parent).
  2. Strict Single-Index Signaling:
     - 1 mode index per parent block, shared across all its children without child overhead.
  3. Iterative Re-Masking (EM=2):
     - EM=1: Standard 1-pass active filtering (retrain centroids on M1 -> grid search slp/slw -> final matrix).
     - EM=2: 2-pass re-masking (Pass 1 active mask M1 -> retrain centroids C1 -> re-evaluate gains to form M2 -> polish centroids C2 on M2 -> grid search slp/slw -> final matrix).

Run command example:
  python scripts/exp_e3_hierarchical_em_sweep.py --child-block-sizes "16,8" --ks "10,12,16" --em-steps "1,2" --taus "0.5,1.0" --gammas "1000,2000"
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
from pcadc.blocks import Block, BlockMetadata
from oracle_em_tangent import extract_block_tangent_planes


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
def section(t): print(f"\n{'='*75}\n  {t}\n{'='*75}", flush=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project_slope(mu, v1, v2, unc_norm, degen=1e-3):
    s = np.dot(mu, v1)*v1 + np.dot(mu, v2)*v2
    rn = np.linalg.norm(s)
    if rn < degen: s = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s/rn)*t


def fit_gradient_3d(Vblock, Ablock, v1, v2):
    """Fit a 3D gradient vector to a single block via tangent-plane linear regression."""
    Vc = Vblock - Vblock.mean(0)
    X = np.column_stack((Vc @ v1, Vc @ v2))
    Y = Ablock[:, 0] - Ablock[:, 0].mean()
    s2d, *_ = np.linalg.lstsq(X, Y, rcond=None)
    g = s2d[0]*v1 + s2d[1]*v2
    rn = np.linalg.norm(g)
    return (g/rn if rn > 1e-8 else np.array([0., 0., 1.])), rn


def spherical_kmeans(directions, K, max_iter=20, seed=42):
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
            sg = np.zeros(3); mu = centroids[k]
            for i in range(len(m)):
                sg += np.sign(np.dot(m[i], mu)) * m[i]
            n = np.linalg.norm(sg)
            if n > 1e-8: centroids[k] = sg/n
    return centroids, labels


def compute_entropy(labels, K):
    tr = np.ones((K, K))
    for i in range(1, len(labels)):
        tr[labels[i-1], labels[i]] += 1
    tp = tr / tr.sum(axis=1, keepdims=True)
    h = 0.0
    cs = np.bincount(labels, minlength=K)
    for p in range(K):
        pp = cs[p]/len(labels)
        for c in range(K):
            v = tp[p, c]
            if v > 0 and pp > 0: h -= pp*v*np.log2(v)
    return h


def run_viterbi(cost_matrix, K, gamma, nw):
    N = len(cost_matrix)
    dp = np.full((N, K), np.inf); paths = np.zeros((N, K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = nw[i]
        for k in range(K):
            tc = dp[i-1] + gamma*w*(np.arange(K) != k)
            bp = np.argmin(tc); dp[i, k] = cost_matrix[i, k]+tc[bp]; paths[i, k] = bp
    vl = np.zeros(N, dtype=int); vl[N-1] = np.argmin(dp[N-1])
    for i in range(N-2, -1, -1): vl[i] = paths[i+1, vl[i+1]]
    return vl


# ---------------------------------------------------------------------------
# Grid search for slp/slw per cluster (E1 verbatim)
# ---------------------------------------------------------------------------

def grid_search(blocks, tangent_data, centroids_3d, g3d_raw, g3d_norm,
                pc_V, pc_A, dm, lag, q, K, active_mask):
    K_hybrid = K - 1
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
    p_grid = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    w_grid = [0.25, 0.50, 0.80, 1.00, 1.40, 1.80]

    active_indices = np.where(active_mask)[0]
    if len(active_indices) > 0:
        active_grads = g3d_norm[active_indices]
        sims = np.abs(active_grads @ centroids_3d.T)
        assigned_cluster = np.argmax(sims, axis=1)
    else:
        assigned_cluster = np.array([], dtype=int)

    dec = Decider(dm, lag); dec._set_vars(q)
    gft0 = GFTStrategyWraper(); app0 = Approximator()

    for k in range(1, K):
        c_idx = k - 1
        k_active = active_indices[assigned_cluster == c_idx]
        if len(k_active) < 3:
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
                    Vr = app0._spatial_norm(Vb); Aa = Ab.copy(); Aa[:, 0] = Vr @ s3d.T
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
    return slp, slw


# ---------------------------------------------------------------------------
# Parent-Child block mapping
# ---------------------------------------------------------------------------

def build_parent_child_map(blocks_p, blocks_c, pc_V, bsize_p):
    parent_key_map = {}
    for pi, pb in enumerate(blocks_p):
        first_vox = pc_V[pb.metadata.start]
        key = tuple((np.floor(first_vox / bsize_p) * bsize_p).astype(int))
        parent_key_map[key] = pi

    parent_to_children = {pi: [] for pi in range(len(blocks_p))}
    unmatched_children = 0
    for ci, cb in enumerate(blocks_c):
        first_vox = pc_V[cb.metadata.start]
        key = tuple((np.floor(first_vox / bsize_p) * bsize_p).astype(int))
        pi = parent_key_map.get(key)
        if pi is not None:
            parent_to_children[pi].append(ci)
        else:
            unmatched_children += 1

    if unmatched_children > 0:
        log(f"  Warning: {unmatched_children} child blocks could not be matched to a parent.")
    return parent_to_children


# ---------------------------------------------------------------------------
# Cost matrix builders
# ---------------------------------------------------------------------------

def build_cost_matrix_standalone_child(child_blocks, tangent_c, g3d_norm_c, g3d_raw_c,
                                        centroids, slp, slw,
                                        pc_V, pc_A, dm, lag, q, K):
    """
    Standalone baseline for a child block size (one B_child GFT graph per child block).
    Viterbi runs at child level (N_child) paying full overhead tax.
    """
    def _worker(ci, Vc, Ac, meta, g3d, v1, v2, unc, centroids, slp, slw, dm, lag, q, K):
        dec = Decider(dm, lag); dec._set_vars(q)
        gft = GFTStrategyWraper(); app = Approximator()
        costs = np.full(K, np.inf)
        b_tmp = Block(meta); b_tmp.Vblock = Vc; b_tmp.Ablock = Ac
        try:
            sg = StructuralGraph(meta); sg.set_data(Vc)
            _, coeffs = gft(b_tmp, sg); c, _, _ = dec._RDcost(coeffs); costs[0] = c
        except: costs[0] = np.inf

        for k in range(1, K):
            slope_3d = project_slope(centroids[k-1], v1, v2, unc)
            try:
                sg = StructuralGraph(meta); sg.set_data(Vc)
                ag = AttributeGraph(sg, slope_3d, k, slp[k], slw[k])
                Vr = app._spatial_norm(Vc); Aa = Ac.copy(); Aa[:, 0] = Vr @ slope_3d.T
                ag.set_data(Vc, Aa)
                _, coeffs = gft(b_tmp, ag); c, _, _ = dec._RDcost(coeffs); costs[k] = c
            except: costs[k] = np.inf
        return costs

    tasks = []
    for ci, cb in enumerate(child_blocks):
        cb.init_data(pc_V, pc_A)
        tasks.append((ci, cb.Vblock.copy(), cb.Ablock.copy(), cb.metadata,
                      g3d_norm_c[ci], tangent_c[ci]["v1"], tangent_c[ci]["v2"],
                      g3d_raw_c[ci], centroids, slp, slw, dm, lag, q, K))
        cb.clear_data()

    rows = Parallel(n_jobs=-1)(delayed(_worker)(*t) for t in tasks)
    return np.array(rows)


def build_cost_matrix_hierarchical(parent_blocks, child_blocks, parent_to_children,
                                    tangent_p, tangent_c, struct_map_c,
                                    centroids, slp, slw,
                                    g3d_parent, g3d_raw_parent,
                                    pc_V, pc_A, dm, lag, q, K):
    """
    Build N_parent x K cost matrix by summing child GFT costs under projected parent centroid slope.
    """
    tasks = []
    for pi, pb in enumerate(parent_blocks):
        child_idxs = parent_to_children[pi]
        children_data = []
        for ci in child_idxs:
            child_blocks[ci].init_data(pc_V, pc_A)
            children_data.append((ci, child_blocks[ci].Vblock.copy(), child_blocks[ci].Ablock.copy(),
                                  child_blocks[ci].metadata))
            child_blocks[ci].clear_data()

        tasks.append((pi, child_idxs, children_data, tangent_c, struct_map_c,
                      centroids, slp, slw,
                      g3d_parent[pi], tangent_p[pi]["v1"], tangent_p[pi]["v2"],
                      g3d_raw_parent[pi], dm, lag, q, K))

    def _worker(pi, child_idxs, children_data, tangent_c, struct_map_c,
                centroids, slp, slw, g3d, v1, v2, unc, dm, lag, q, K):
        dec = Decider(dm, lag); dec._set_vars(q)
        gft = GFTStrategyWraper(); app = Approximator()
        costs = np.full(K, np.inf)

        # k=0: structural sum over children
        sc = 0.0
        for ci, Vc_arr, Ac_arr, meta in children_data:
            coeffs_ci = struct_map_c.get(ci)
            if coeffs_ci is None: sc = np.inf; break
            try: c, _, _ = dec._RDcost(coeffs_ci); sc += c
            except: sc = np.inf; break
        costs[0] = sc

        # k>0: parent-directed child GFTs
        for k in range(1, K):
            slope_3d = project_slope(centroids[k-1], v1, v2, unc)
            total_k = 0.0
            for ci, Vc_arr, Ac_arr, meta in children_data:
                tc = tangent_c[ci]
                cv1, cv2 = tc["v1"], tc["v2"]
                s2 = np.dot(slope_3d, cv1)*cv1 + np.dot(slope_3d, cv2)*cv2
                rn = np.linalg.norm(s2)
                if rn < 1e-3: s2 = cv1*1e-3; rn = 1e-3
                Vc_c = Vc_arr - Vc_arr.mean(0)
                X = np.column_stack((Vc_c @ cv1, Vc_c @ cv2))
                Y = Ac_arr[:, 0] - Ac_arr[:, 0].mean()
                s2d_c, *_ = np.linalg.lstsq(X, Y, rcond=None)
                g_c = s2d_c[0]*cv1 + s2d_c[1]*cv2
                unc_c = np.linalg.norm(g_c)
                s3d_c = (s2/rn)*(unc_c if unc_c > 1e-8 else rn)
                sg = StructuralGraph(meta); sg.set_data(Vc_arr)
                ag = AttributeGraph(sg, s3d_c, k, slp[k], slw[k])
                Vr = app._spatial_norm(Vc_arr); Aa = Ac_arr.copy(); Aa[:, 0] = Vr @ s3d_c.T
                ag.set_data(Vc_arr, Aa)
                b_tmp = Block(meta); b_tmp.Vblock = Vc_arr; b_tmp.Ablock = Ac_arr
                try:
                    _, coeffs = gft(b_tmp, ag); c, _, _ = dec._RDcost(coeffs); total_k += c
                except: total_k = np.inf; break
            costs[k] = total_k
        return costs

    rows = Parallel(n_jobs=-1)(delayed(_worker)(*t) for t in tasks)
    return np.array(rows)


# ---------------------------------------------------------------------------
# Execution logic per child configuration
# ---------------------------------------------------------------------------

def run_child_configuration(pc, blocks_p, tangent_p, g3d_norm_p, g3d_raw_p, normals_p,
                            Bc, ks, em_steps_list, taus, gammas, dm, lag, q):
    Bp = 32
    Np = len(blocks_p)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks_p)

    log(f"Partitioning child block size B={Bc}...")
    _, blocks_c = MortonBlockPartition().partition(pc, bsize=Bc)
    Nc = len(blocks_c)
    log(f"  Parent B={Bp} ({Np} blocks) -> Child B={Bc} ({Nc} blocks, ~{Nc/Np:.1f} per parent)")

    parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc.V, Bp)
    tangent_c = extract_block_tangent_planes(blocks_c, pc.V, pc.A)

    g3d_norm_c, g3d_raw_c = [], []
    normals_c = []
    for ci, cb in enumerate(blocks_c):
        cb.init_data(pc.V, pc.A)
        normals_c.append(tangent_c[ci]["v3"])
        g, rn = fit_gradient_3d(cb.Vblock, cb.Ablock, tangent_c[ci]["v1"], tangent_c[ci]["v2"])
        g3d_norm_c.append(g); g3d_raw_c.append(rn)
        cb.clear_data()
    g3d_norm_c = np.array(g3d_norm_c); g3d_raw_c = np.array(g3d_raw_c)
    normals_c = np.array(normals_c)

    nw_p = np.ones(Np)
    for i in range(1, Np): nw_p[i] = abs(np.dot(normals_p[i], normals_p[i-1]))

    nw_c = np.ones(Nc)
    for i in range(1, Nc): nw_c[i] = abs(np.dot(normals_c[i], normals_c[i-1]))

    gft_comp = GFTStrategyWraper()
    struct_map_c = {}
    log(f"Computing child-scale structural GFT (B={Bc})...")
    for ci, cb in enumerate(tqdm(blocks_c, desc=f"Structural GFT (B={Bc})", ncols=70)):
        cb.init_data(pc.V, pc.A)
        sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
        _, coeffs = gft_comp(cb, sg)
        struct_map_c[ci] = coeffs
        cb.clear_data()

    from pcadc.pointcloud import Sampler
    sampled = Sampler(ratio=0.01, n_strata=5)(pc.V, pc.A, blocks_c)
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    tr, tv2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tv2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs); tr += r; b.clear_data()
    base_rate = tr / tv2
    log(f"Baseline rate for B={Bc}: {base_rate:.5f} bpv")

    # Evaluate standalone B_child baselines per K
    baselines_bchild = {}  # K -> dict { gamma -> net_bpv }
    for K in ks:
        K_hybrid = K - 1
        log(f"Evaluating Standalone B={Bc} Baseline for K={K}...")
        np.random.seed(42)
        idx0 = np.random.choice(Nc, K_hybrid, replace=False)
        centroids_base = g3d_norm_c[idx0].copy()
        for _ in range(20):
            sims = np.abs(g3d_norm_c @ centroids_base.T)
            lbl = np.argmax(sims, axis=1)
            for k in range(K_hybrid):
                m = (lbl == k); sg_v = np.zeros(3); mu = centroids_base[k]
                for i in np.where(m)[0]: sg_v += np.sign(np.dot(g3d_norm_c[i], mu)) * g3d_norm_c[i]
                n = np.linalg.norm(sg_v)
                if n > 1e-8: centroids_base[k] = sg_v / n

        slp_init = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
        slw_init = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
        cost_mat0 = build_cost_matrix_standalone_child(
            blocks_c, tangent_c, g3d_norm_c, g3d_raw_c,
            centroids_base, slp_init, slw_init,
            pc.V, pc.A, dm, lag, q, K)

        gains0 = (cost_mat0[:, 0] - cost_mat0[:, 1:].min(axis=1)) / (cost_mat0[:, 0] + 1e-12) * 100
        active_mask0 = gains0 > 0.5
        if active_mask0.sum() >= K_hybrid * 3:
            centroids_base, _ = spherical_kmeans(g3d_norm_c[active_mask0], K_hybrid)
            slp0, slw0 = grid_search(blocks_c, tangent_c, centroids_base, g3d_raw_c, g3d_norm_c,
                                     pc.V, pc.A, dm, lag, q, K, active_mask0)
            cost_mat0 = build_cost_matrix_standalone_child(
                blocks_c, tangent_c, g3d_norm_c, g3d_raw_c,
                centroids_base, slp0, slw0,
                pc.V, pc.A, dm, lag, q, K)

        struct_tot0 = cost_mat0[:, 0].sum()
        baselines_bchild[K] = {}
        for gamma in gammas:
            vl_c = run_viterbi(cost_mat0, K, gamma, nw_c)
            gft_c = sum(cost_mat0[i, vl_c[i]] for i in range(Nc))
            sav0 = (struct_tot0 - gft_c) / struct_tot0 * 100
            H0 = compute_entropy(vl_c, K)
            oh0 = H0 * Nc / total_v
            net0 = base_rate * (sav0 / 100) - oh0
            baselines_bchild[K][gamma] = net0
            log(f"  Standalone B={Bc} | K={K} γ={gamma:.0f}: GFT={sav0:.3f}% OH={oh0:.5f} Net={net0:+.5f}")

    results_sweep = []  # dict of records

    for K in ks:
        K_hybrid = K - 1

        for tau in taus:
            log(f"\n=======================================================")
            log(f"Running Hierarchical B={Bp}->{Bc} | K={K} | τ={tau}%")
            log(f"=======================================================")

            # Stage 1: Base Spherical K-Means on parent gradients
            np.random.seed(42)
            idx0 = np.random.choice(Np, K_hybrid, replace=False)
            centroids_p0 = g3d_norm_p[idx0].copy()
            for _ in range(20):
                sims = np.abs(g3d_norm_p @ centroids_p0.T)
                lbl = np.argmax(sims, axis=1)
                for k in range(K_hybrid):
                    m = (lbl == k); sg_v = np.zeros(3); mu = centroids_p0[k]
                    for i in np.where(m)[0]: sg_v += np.sign(np.dot(g3d_norm_p[i], mu)) * g3d_norm_p[i]
                    n = np.linalg.norm(sg_v)
                    if n > 1e-8: centroids_p0[k] = sg_v / n

            slp_init_p = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
            slw_init_p = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

            # Pass 1 Cost Matrix
            log("Pass 1: Building initial parent cost matrix...")
            t0 = time.time()
            cost_mat_pass1 = build_cost_matrix_hierarchical(
                blocks_p, blocks_c, parent_to_children,
                tangent_p, tangent_c, struct_map_c,
                centroids_p0, slp_init_p, slw_init_p,
                g3d_norm_p, g3d_raw_p,
                pc.V, pc.A, dm, lag, q, K)
            log(f"  Pass 1 cost matrix done in {time.time()-t0:.1f}s")

            struct_tot_pass1 = cost_mat_pass1[:, 0].sum()
            gains_pass1 = (cost_mat_pass1[:, 0] - cost_mat_pass1[:, 1:].min(axis=1)) / (cost_mat_pass1[:, 0] + 1e-12) * 100
            active_mask_m1 = gains_pass1 > tau
            n_act_m1 = active_mask_m1.sum()
            log(f"  Stage 1 Active Parents (M1): {n_act_m1}/{Np} ({100*n_act_m1/Np:.1f}%)")

            for em in em_steps_list:
                log(f"\n--- Executing EM={em} for K={K}, τ={tau}% ---")
                t_em_start = time.time()

                if em == 1:
                    # EM=1: Single-pass retrain on M1 active parents
                    if n_act_m1 < K_hybrid * 3:
                        log(f"  SKIP EM=1: fewer than {K_hybrid*3} active parents.")
                        continue
                    centroids_em1, _ = spherical_kmeans(g3d_norm_p[active_mask_m1], K_hybrid)

                    # Active children under active parents M1
                    child_act_m1 = np.zeros(Nc, dtype=bool)
                    for pi in np.where(active_mask_m1)[0]:
                        for ci in parent_to_children[pi]:
                            child_act_m1[ci] = True

                    log("  Grid search slp/slw on active children (EM=1)...")
                    slp_em1, slw_em1 = grid_search(
                        blocks_c, tangent_c, centroids_em1, g3d_raw_c, g3d_norm_c,
                        pc.V, pc.A, dm, lag, q, K, child_act_m1)

                    log("  Building final cost matrix (EM=1)...")
                    cost_mat_final = build_cost_matrix_hierarchical(
                        blocks_p, blocks_c, parent_to_children,
                        tangent_p, tangent_c, struct_map_c,
                        centroids_em1, slp_em1, slw_em1,
                        g3d_norm_p, g3d_raw_p,
                        pc.V, pc.A, dm, lag, q, K)
                    n_act_final = n_act_m1

                elif em == 2:
                    # EM=2: 2-stage EM loop (Re-evaluate -> Re-filter M2 -> Polish centroids C2 -> Grid search -> Final matrix)
                    if n_act_m1 < K_hybrid * 3:
                        log(f"  SKIP EM=2: Stage 1 active parents < {K_hybrid*3}.")
                        continue
                    centroids_em2_pass1, _ = spherical_kmeans(g3d_norm_p[active_mask_m1], K_hybrid)

                    log("  EM=2 Stage 2: Re-evaluating cost matrix to form M2 active mask...")
                    cost_mat_pass2 = build_cost_matrix_hierarchical(
                        blocks_p, blocks_c, parent_to_children,
                        tangent_p, tangent_c, struct_map_c,
                        centroids_em2_pass1, slp_init_p, slw_init_p,
                        g3d_norm_p, g3d_raw_p,
                        pc.V, pc.A, dm, lag, q, K)

                    gains_pass2 = (cost_mat_pass2[:, 0] - cost_mat_pass2[:, 1:].min(axis=1)) / (cost_mat_pass2[:, 0] + 1e-12) * 100
                    active_mask_m2 = gains_pass2 > tau
                    n_act_m2 = active_mask_m2.sum()
                    log(f"  Stage 2 Re-filtered Active Parents (M2): {n_act_m2}/{Np} ({100*n_act_m2/Np:.1f}%)")

                    if n_act_m2 < K_hybrid * 3:
                        log(f"  Fallback M2: using M1 mask as fallback.")
                        active_mask_m2 = active_mask_m1
                        n_act_m2 = n_act_m1

                    log("  Final Centroid Polishing (20 iterations spherical k-means on M2)...")
                    centroids_em2_final, _ = spherical_kmeans(g3d_norm_p[active_mask_m2], K_hybrid, max_iter=20)

                    child_act_m2 = np.zeros(Nc, dtype=bool)
                    for pi in np.where(active_mask_m2)[0]:
                        for ci in parent_to_children[pi]:
                            child_act_m2[ci] = True

                    log("  Grid search slp/slw on active children (EM=2)...")
                    slp_em2, slw_em2 = grid_search(
                        blocks_c, tangent_c, centroids_em2_final, g3d_raw_c, g3d_norm_c,
                        pc.V, pc.A, dm, lag, q, K, child_act_m2)

                    log("  Building final cost matrix (EM=2)...")
                    cost_mat_final = build_cost_matrix_hierarchical(
                        blocks_p, blocks_c, parent_to_children,
                        tangent_p, tangent_c, struct_map_c,
                        centroids_em2_final, slp_em2, slw_em2,
                        g3d_norm_p, g3d_raw_p,
                        pc.V, pc.A, dm, lag, q, K)
                    n_act_final = n_act_m2

                struct_tot_final = cost_mat_final[:, 0].sum()

                for gamma in gammas:
                    vl_p = run_viterbi(cost_mat_final, K, gamma, nw_p)
                    gft_c = sum(cost_mat_final[i, vl_p[i]] for i in range(Np))
                    sav_final = (struct_tot_final - gft_c) / struct_tot_final * 100
                    H_final = compute_entropy(vl_p, K)
                    oh_final = H_final * Np / total_v
                    net_final = base_rate * (sav_final / 100) - oh_final
                    sw_final = int((vl_p[1:] != vl_p[:-1]).sum())

                    standalone_ref = baselines_bchild[K].get(gamma, 0.0)
                    dnet_vs_bchild = net_final - standalone_ref

                    log(f"  Result -> EM={em} K={K} τ={tau}% γ={gamma:.0f} | "
                        f"GFT={sav_final:.3f}% OH={oh_final:.5f} Net={net_final:+.5f} "
                        f"ΔNet vs B{Bc}={dnet_vs_bchild:+.5f}")

                    results_sweep.append({
                        "child_b": Bc,
                        "k": K,
                        "em": em,
                        "tau": tau,
                        "gamma": gamma,
                        "n_active": n_act_final,
                        "n_active_pct": 100 * n_act_final / Np,
                        "gft_sav": sav_final,
                        "oh_bpv": oh_final,
                        "net_bpv": net_final,
                        "standalone_ref": standalone_ref,
                        "dnet": dnet_vs_bchild,
                        "switches": sw_final,
                        "time_s": time.time() - t_em_start
                    })

    return results_sweep, baselines_bchild


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--child-block-sizes", type=str, default="16,8", help="Child block sizes (comma-separated)")
    p.add_argument("--ks", type=str, default="10,12,16", help="K cluster capacities (comma-separated)")
    p.add_argument("--em-steps", type=str, default="1,2", help="EM steps to test (comma-separated)")
    p.add_argument("--taus", type=str, default="0.5,1.0", help="Tau gain thresholds %% (comma-separated)")
    p.add_argument("--gammas", type=str, default="1000,2000", help="Viterbi gammas (comma-separated)")
    p.add_argument("--q-step", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    child_b_list = [int(x) for x in args.child_block_sizes.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    em_steps_list = [int(x) for x in args.em_steps.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    q = args.q_step

    section("HIERARCHICAL EM-REFINED GFT SWEEP (Parent B=32)")
    log(f"Child block sizes: {child_b_list}")
    log(f"K capacities: {ks}")
    log(f"EM steps: {em_steps_list}")
    log(f"Tau thresholds: {taus}%")
    log(f"Gammas: {gammas}")

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    Bp = 32
    log(f"Partitioning parent blocks B={Bp}...")
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=Bp)
    Np = len(blocks_p)
    log(f"  Parent blocks: {Np}")

    tangent_p = extract_block_tangent_planes(blocks_p, pc.V, pc.A)

    g3d_norm_p, g3d_raw_p = [], []
    normals_p = []
    for i, pb in enumerate(blocks_p):
        pb.init_data(pc.V, pc.A)
        normals_p.append(tangent_p[i]["v3"])
        g, rn = fit_gradient_3d(pb.Vblock, pb.Ablock, tangent_p[i]["v1"], tangent_p[i]["v2"])
        g3d_norm_p.append(g); g3d_raw_p.append(rn)
        pb.clear_data()
    g3d_norm_p = np.array(g3d_norm_p); g3d_raw_p = np.array(g3d_raw_p)
    normals_p = np.array(normals_p)

    all_sweep_results = []

    for Bc in child_b_list:
        section(f"SWEEPING CHILD BLOCK SIZE B={Bc}")
        res_bc, base_bc = run_child_configuration(
            pc, blocks_p, tangent_p, g3d_norm_p, g3d_raw_p, normals_p,
            Bc, ks, em_steps_list, taus, gammas, dm, lag, q)
        all_sweep_results.extend(res_bc)

    # ---------------------------------------------------------------------------
    # Structured Summary Table
    # ---------------------------------------------------------------------------
    section("HIERARCHICAL EM-REFINED GFT SWEEP SUMMARY (Parent B=32)")
    print()
    header = f"{'Child B':>7} | {'K':>3} | {'EM':>2} | {'τ%':>5} | {'γ':>6} | {'Active Parents':>16} | {'GFT% Sav':>8} | {'Overhead BPV':>12} | {'Net BPV':>10} | {'ΔNet vs B_child':>15}"
    print(header)
    print("-" * len(header))

    for r in all_sweep_results:
        act_str = f"{r['n_active']} ({r['n_active_pct']:.1f}%)"
        print(f"B={r['child_b']:<5} | {r['k']:>3} | {r['em']:>2} | {r['tau']:>4.1f}% | {r['gamma']:>6.0f} | "
              f"{act_str:>16} | {r['gft_sav']:>7.3f}% | {r['oh_bpv']:>12.5f} | {r['net_bpv']:>+10.5f} | {r['dnet']:>+15.5f}")

    print("-" * len(header))


if __name__ == "__main__":
    main()
