"""
Experiment: Hierarchical Decoupled GFT
=======================================
Scientific Motivation:
  Two traps revealed by the architectural sweep:
  - Fine-scale overhead trap (B=8,16): Small block GFT savings eaten by per-block index entropy.
  - Coarse-scale blurring trap (B=32): Large blocks span multiple surfaces, causing gradient
    cancellation and blurring.

  Solution: Decouple MODE SIGNALING RESOLUTION from TRANSFORM GRAPH RESOLUTION.
  Signal a single cluster mode index at the B=32 parent level (N=834 blocks, low overhead),
  but execute independent GFT graphs on 8 B=16 child subblocks per parent.

Experimental Arms:
  Arm 0 (Control): Single B=32 GFT graph per parent block. (Existing best baseline.)
  Arm 1 (Top-Down): Parent B=32 gradient → signal at B=32 level → execute 8x B=16 child GFTs.
  Arm 2 (Bottom-Up Consensus): 8x B=16 child gradients → SVD consensus → signal at B=32 level
    → execute 8x B=16 child GFTs.

Overhead Equality Guarantee:
  All arms compute Viterbi and index entropy at the B=32 level (N=834), so the overhead BPV
  term is identical across arms. This isolates energy compaction gains as the sole variable.

SVD Consensus (Arm 2):
  S = sum_m(d_m @ d_m.T) for active child gradients d_m.
  u_parent = dominant eigenvector of S.
  Antipodal gradients (d and -d) contribute identically → no cancellation.

Runtime estimate: ~15-25 min for K=10, tau=0.5%, 3 gammas. Arm 2 takes longer due to
  per-child gradient extraction.
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
def section(t): print(f"\n{'='*62}\n  {t}\n{'='*62}", flush=True)


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


def svd_consensus(directions):
    """
    Sign-invariant SVD consensus. Given unit direction vectors d_m, build
    scatter matrix S = sum(d_m d_m^T) and return dominant eigenvector.
    Antipodal vectors contribute identically so no cancellation occurs.
    """
    S = np.zeros((3, 3))
    for d in directions:
        S += np.outer(d, d)
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs[:, np.argmax(eigvals)]  # unit vector, dominant axis


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
            sg = m.sum(0); n = np.linalg.norm(sg)
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
# Grid search for slp/slw per cluster — copied verbatim from exp_e1_gain_filter.py
# Operates on child-scale blocks (same block set that GFT is evaluated on).
# active_mask: boolean array over child blocks indicating high-gain blocks.
# centroids_3d: K_hybrid centroid directions (k=0 is structural, k=1..K_hybrid directional).
# ---------------------------------------------------------------------------

def grid_search(blocks, tangent_data, centroids_3d, g3d_raw, g3d_norm,
                pc_V, pc_A, dm, lag, q, K, active_mask):
    K_hybrid = K - 1
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
    p_grid = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    w_grid = [0.25, 0.50, 0.80, 1.00, 1.40, 1.80]

    # Assign active blocks to their closest directional centroid
    active_indices = np.where(active_mask)[0]
    if len(active_indices) > 0:
        active_grads = g3d_norm[active_indices]
        sims = np.abs(active_grads @ centroids_3d.T)  # (N_active, K_hybrid)
        assigned_cluster = np.argmax(sims, axis=1)     # (N_active,)
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
        log(f"    k={k}: slp={best_p:.2f} slw={best_w:.2f}")
    return slp, slw


# ---------------------------------------------------------------------------
# Parent-Child block mapping
# ---------------------------------------------------------------------------

def build_parent_child_map(blocks_p, blocks_c, pc_V, bsize_p):
    """
    Map each B=bsize_p parent block to its constituent B=bsize_p/2 child blocks.

    Strategy: For each child block, compute its coarse B=bsize_p spatial key
    (floor(voxel / bsize_p) * bsize_p) and assign it to the matching parent.
    Uses metadata index ranges directly (no init_data) to avoid requiring pc_A.

    Returns: dict { parent_index -> list of child_indices }
    """
    # Build parent spatial key map: coarse origin → parent index
    # Use the first voxel of each block (pc_V[start]) to determine spatial cell
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
# Structural GFT cost for a list of child blocks (summed) — parent mode k=0
# ---------------------------------------------------------------------------

def _struct_cost_children(child_indices, blocks_c, pc_V, pc_A, struct_map_c, dec):
    total = 0.0
    for ci in child_indices:
        coeffs = struct_map_c.get(ci)
        if coeffs is None: return np.inf
        try:
            c, _, _ = dec._RDcost(coeffs)
            total += c
        except: return np.inf
    return total


# ---------------------------------------------------------------------------
# GFT cost for child blocks under a given directional slope (Arms 1 & 2)
# ---------------------------------------------------------------------------

def _child_directional_cost(child_indices, blocks_c, tangent_c, pc_V, pc_A,
                             slope_3d, k_label, slp_k, slw_k, dec):
    """Sum GFT R-D costs for all child blocks under the given 3D slope direction."""
    gft = GFTStrategyWraper(); app = Approximator()
    total = 0.0
    for ci in child_indices:
        cb = blocks_c[ci]; cb.init_data(pc_V, pc_A)
        v1, v2 = tangent_c[ci]["v1"], tangent_c[ci]["v2"]
        # Project parent/consensus slope onto child's tangent plane
        s2 = np.dot(slope_3d, v1)*v1 + np.dot(slope_3d, v2)*v2
        rn = np.linalg.norm(s2)
        if rn < 1e-3: s2 = v1*1e-3; rn = 1e-3
        # Keep magnitude from child's own gradient regression
        Vc = cb.Vblock - cb.Vblock.mean(0)
        X = np.column_stack((Vc @ v1, Vc @ v2))
        Y = cb.Ablock[:, 0] - cb.Ablock[:, 0].mean()
        s2d_child, *_ = np.linalg.lstsq(X, Y, rcond=None)
        g_child = s2d_child[0]*v1 + s2d_child[1]*v2
        unc_norm = np.linalg.norm(g_child)
        s3d = (s2/rn) * (unc_norm if unc_norm > 1e-8 else rn)
        # Build child GFT graph with parent's directional mode
        sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
        ag = AttributeGraph(sg, s3d, k_label, slp_k, slw_k)
        Vr = app._spatial_norm(cb.Vblock); Aa = cb.Ablock.copy(); Aa[:, 0] = Vr @ s3d.T
        ag.set_data(cb.Vblock, Aa)
        b_tmp = Block(cb.metadata); b_tmp.Vblock = cb.Vblock; b_tmp.Ablock = cb.Ablock
        try:
            _, coeffs = gft(b_tmp, ag); c, _, _ = dec._RDcost(coeffs); total += c
        except: total += np.inf
        cb.clear_data()
    return total


# ---------------------------------------------------------------------------
# Parent cost matrix construction (parallelised over parents)
# ---------------------------------------------------------------------------

def _eval_parent_row(pi, child_indices, blocks_c_data, tangent_c,
                     struct_map_c, centroids, slp, slw,
                     parent_g3d, parent_v1, parent_v2, parent_unc,
                     mode, dm, lag, q, K):
    """
    Evaluate cost for one parent block under all K cluster modes.
    mode: 'arm0' | 'arm1' | 'arm2'
    parent_g3d: 3D unit gradient (Arm1: from parent regression; Arm2: SVD consensus)
    """
    dec = Decider(dm, lag); dec._set_vars(q)
    costs = np.full(K, np.inf)

    # Arm 0: single B=32 GFT for each child block separately (structural)
    # For Arm0 we pass struct_map_c costs directly for mode 0
    # and arm0 directional uses the B=32 slope projected per child

    # Mode k=0: structural — same for all arms
    sc = 0.0
    for ci in child_indices:
        coeffs_ci = struct_map_c.get(ci)
        if coeffs_ci is None: sc = np.inf; break
        try: c, _, _ = dec._RDcost(coeffs_ci); sc += c
        except: sc = np.inf; break
    costs[0] = sc

    # Directional modes k=1..K-1
    gft = GFTStrategyWraper(); app = Approximator()
    for k in range(1, K):
        slope_3d = project_slope(centroids[k-1], parent_v1, parent_v2, parent_unc)
        c_k = _child_directional_cost(
            child_indices, blocks_c_data, tangent_c, None, None,
            slope_3d, k, slp[k], slw[k], dec)
        costs[k] = c_k
    return costs


def build_cost_matrix_hierarchical(
        parent_blocks, child_blocks, parent_to_children,
        tangent_p, tangent_c, struct_map_c,
        centroids, slp, slw,
        g3d_parent, g3d_raw_parent,
        pc_V, pc_A, dm, lag, q, K):
    """Build the N_parent x K cost matrix using parallel workers."""
    tasks = []
    for pi, pb in enumerate(parent_blocks):
        child_idxs = parent_to_children[pi]
        # Pre-load child block data for this parent
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
# Arm 0: Standalone B=16 baseline (the overhead-limited target to beat)
# ---------------------------------------------------------------------------

def build_cost_matrix_standalone_b16(child_blocks, tangent_c, g3d_norm_c, g3d_raw_c,
                                      centroids, slp, slw,
                                      pc_V, pc_A, dm, lag, q, K):
    """
    Build N_child x K cost matrix: one B=16 GFT graph per child block.
    k=0: StructuralGraph on child voxels.
    k>0: AttributeGraph on child voxels with child-scale centroid slope.
    Viterbi runs at B=16 level (N~3480) → high overhead tax.
    This is the standalone B=16 baseline that Arms 1 & 2 aim to beat.
    """
    def _worker_b16(ci, Vc, Ac, meta, g3d, v1, v2, unc, centroids, slp, slw, dm, lag, q, K):
        dec = Decider(dm, lag); dec._set_vars(q)
        gft = GFTStrategyWraper(); app = Approximator()
        costs = np.full(K, np.inf)
        b_tmp = Block(meta); b_tmp.Vblock = Vc; b_tmp.Ablock = Ac
        # k=0: structural
        try:
            sg = StructuralGraph(meta); sg.set_data(Vc)
            _, coeffs = gft(b_tmp, sg); c, _, _ = dec._RDcost(coeffs); costs[0] = c
        except: costs[0] = np.inf
        # k>0: directional (child-scale slope from child centroid)
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

    rows = Parallel(n_jobs=-1)(delayed(_worker_b16)(*t) for t in tasks)
    return np.array(rows)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size-parent", type=int, default=32)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--q-step", type=int, default=24)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Gain filter threshold %% for centroid retraining")
    p.add_argument("--gammas", type=str, default="1000,2000,4000")
    p.add_argument("--arms", type=str, default="0,1,2",
                   help="Arms to run (comma-separated subset of 0,1,2)")
    return p.parse_args()


def main():
    args = parse_args()
    Bp = args.block_size_parent
    Bc = Bp // 2
    K, q = args.k, args.q_step
    K_hybrid = K - 1
    tau = args.tau
    gammas = [float(x) for x in args.gammas.split(",")]
    arms = [int(x) for x in args.arms.split(",")]

    section(f"Hierarchical Decoupled GFT  (Parent B={Bp}, Child B={Bc}, K={K}, τ={tau}%)")
    log(f"Arms: {arms}")
    log(f"Mode signaling: B={Bp} (N~834 blocks) | GFT execution: B={Bc} (per-child)")

    params = load_experiment_config(project_root / "config/base_config.yaml")
    pc = PointCloud.from_file(project_root / params.sequential_params.point_cloud_path,
                              "ply", params.pointcloud)
    pc.transform_attributes(Colourist()._RGBtoYUV)
    dm = params.sequential_params.decider_mode
    lag = params.sequential_params.lagrange_proportional

    # Partition at both scales
    log(f"Partitioning at parent B={Bp} and child B={Bc}...")
    _, blocks_p = MortonBlockPartition().partition(pc, bsize=Bp)
    _, blocks_c = MortonBlockPartition().partition(pc, bsize=Bc)
    Np, Nc = len(blocks_p), len(blocks_c)
    total_v = sum(b.metadata.end - b.metadata.start + 1 for b in blocks_p)
    log(f"  Parent blocks: {Np} | Child blocks: {Nc} | Voxels: {total_v}")
    log(f"  Expected children/parent: ~{Nc/Np:.1f}")

    # Build parent → children mapping
    log("Building parent→child spatial mapping...")
    parent_to_children = build_parent_child_map(blocks_p, blocks_c, pc.V, Bp)
    child_counts = [len(v) for v in parent_to_children.values()]
    log(f"  Children per parent: min={min(child_counts)}, max={max(child_counts)}, "
        f"mean={np.mean(child_counts):.1f}")

    # Extract tangent planes at both scales
    log("Extracting tangent planes (parent scale)...")
    tangent_p = extract_block_tangent_planes(blocks_p, pc.V, pc.A)
    log("Extracting tangent planes (child scale)...")
    tangent_c = extract_block_tangent_planes(blocks_c, pc.V, pc.A)

    # Compute parent-scale gradients (for Arm 0 and Arm 1)
    log("Computing parent-scale gradients (Arm 0 / Arm 1)...")
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

    # Compute SVD consensus parent-scale gradients (for Arm 2)
    log("Computing SVD consensus gradients (Arm 2)...")
    g3d_norm_consensus = []
    for i, pb in enumerate(blocks_p):
        child_idxs = parent_to_children[i]
        child_dirs = []
        for ci in child_idxs:
            blocks_c[ci].init_data(pc.V, pc.A)
            g_c, rn_c = fit_gradient_3d(blocks_c[ci].Vblock, blocks_c[ci].Ablock,
                                         tangent_c[ci]["v1"], tangent_c[ci]["v2"])
            if rn_c > 1e-6: child_dirs.append(g_c)
            blocks_c[ci].clear_data()
        if len(child_dirs) >= 2:
            g3d_norm_consensus.append(svd_consensus(np.array(child_dirs)))
        elif len(child_dirs) == 1:
            g3d_norm_consensus.append(child_dirs[0])
        else:
            g3d_norm_consensus.append(g3d_norm_p[i])  # fallback to parent
    g3d_norm_consensus = np.array(g3d_norm_consensus)

    # Transition weights for Viterbi (normal angle similarity)
    nw = np.ones(Np)
    for i in range(1, Np): nw[i] = abs(np.dot(normals_p[i], normals_p[i-1]))

    # Structural GFT at child scale
    gft_comp = GFTStrategyWraper()
    struct_map_c = {}  # child_index → coefficients
    log("Computing structural GFT at child scale...")
    for ci, cb in enumerate(tqdm(blocks_c, desc="Structural GFT (child)", ncols=70)):
        cb.init_data(pc.V, pc.A)
        sg = StructuralGraph(cb.metadata); sg.set_data(cb.Vblock)
        _, coeffs = gft_comp(cb, sg)
        struct_map_c[ci] = coeffs
        cb.clear_data()

    # Baseline rate (at child scale for structural)
    from pcadc.pointcloud import Sampler
    sampled = Sampler(ratio=0.01, n_strata=5)(pc.V, pc.A, blocks_c)
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    tr, tv2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tv2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs); tr += r; b.clear_data()
    base_rate = tr / tv2
    log(f"Baseline rate: {base_rate:.5f} bpv")

    # Child-scale gradients (needed by Arm 0 centroid training and by Arms 1&2 grid search)
    log("Computing child-scale gradients (for Arm 0 centroids / Arms 1&2 grid search)...")
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
    nw_c = np.ones(Nc)
    for i in range(1, Nc): nw_c[i] = abs(np.dot(normals_c[i], normals_c[i-1]))

    results = {}  # arm → list of (gamma, gft%, overhead, net, switches)

    # ===========================================================================
    # ARM 0: Standalone B=16 Baseline — the overhead-limited target to beat
    #   Viterbi at B=16 level (Nc=3480) → overhead = H * Nc / N_voxels
    #   Each B=16 block evaluated independently against child-scale centroids
    # ===========================================================================
    if 0 in arms:
        section("ARM 0 — Standalone B=16 Baseline (Overhead-Limited Target)")
        log(f"N_child={Nc} blocks (signaling {Nc} mode decisions, high overhead tax)")

        # Train centroids at B=16 scale
        np.random.seed(42)
        idx0 = np.random.choice(Nc, K_hybrid, replace=False)
        centroids_arm0 = g3d_norm_c[idx0].copy()
        for _ in range(20):
            sims = np.abs(g3d_norm_c @ centroids_arm0.T)
            lbl = np.argmax(sims, axis=1)
            for k in range(K_hybrid):
                m = (lbl == k); sg_v = np.zeros(3); mu = centroids_arm0[k]
                for i in np.where(m)[0]: sg_v += np.sign(np.dot(g3d_norm_c[i], mu)) * g3d_norm_c[i]
                n = np.linalg.norm(sg_v)
                if n > 1e-8: centroids_arm0[k] = sg_v / n

        # Build B=16 cost matrix with initial slp/slw defaults (same as E1 baseline phase)
        slp_init = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
        slw_init = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
        log("Building Arm 0 initial cost matrix (slp/slw defaults, one B=16 GFT per child)...")
        t0 = time.time()
        cost_matrix_arm0 = build_cost_matrix_standalone_b16(
            blocks_c, tangent_c, g3d_norm_c, g3d_raw_c,
            centroids_arm0, slp_init, slw_init,
            pc.V, pc.A, dm, lag, q, K)
        log(f"  Initial cost matrix done in {time.time()-t0:.1f}s")

        struct_total_arm0 = cost_matrix_arm0[:, 0].sum()
        gains_arm0 = (cost_matrix_arm0[:, 0] - cost_matrix_arm0[:, 1:].min(axis=1)) / (cost_matrix_arm0[:, 0] + 1e-12) * 100
        log(f"  Gain dist: P25={np.percentile(gains_arm0,25):+.2f}%  "
            f"P50={np.percentile(gains_arm0,50):+.2f}%  P75={np.percentile(gains_arm0,75):+.2f}%")

        active_mask_arm0 = gains_arm0 > tau
        n_active = active_mask_arm0.sum()
        log(f"  Gain-filter τ={tau}%: Active={n_active} ({100*n_active/Nc:.1f}%) children")

        if n_active >= K_hybrid * 3:
            active_grads = g3d_norm_c[active_mask_arm0]
            centroids_arm0, _ = spherical_kmeans(active_grads, K_hybrid)

            log("  Grid search slp/slw on active children (matching E1 rigour)...")
            slp_arm0, slw_arm0 = grid_search(
                blocks_c, tangent_c, centroids_arm0, g3d_raw_c, g3d_norm_c,
                pc.V, pc.A, dm, lag, q, K, active_mask_arm0)

            log("  Rebuilding Arm 0 cost matrix with grid-searched slp/slw...")
            t0 = time.time()
            cost_matrix_arm0 = build_cost_matrix_standalone_b16(
                blocks_c, tangent_c, g3d_norm_c, g3d_raw_c,
                centroids_arm0, slp_arm0, slw_arm0,
                pc.V, pc.A, dm, lag, q, K)
            struct_total_arm0 = cost_matrix_arm0[:, 0].sum()
            log(f"  Arm 0 retrain done in {time.time()-t0:.1f}s")
        else:
            slp_arm0, slw_arm0 = slp_init, slw_init
            log(f"  SKIP retrain: fewer than {K_hybrid*3} active children.")


        results[0] = []
        for gamma in gammas:
            # Viterbi at B=16 level — pays full child-scale overhead
            vl_c = run_viterbi(cost_matrix_arm0, K, gamma, nw_c)
            gft_c = sum(cost_matrix_arm0[i, vl_c[i]] for i in range(Nc))
            sav = (struct_total_arm0 - gft_c) / struct_total_arm0 * 100
            H = compute_entropy(vl_c, K)
            oh = H * Nc / total_v          # overhead at B=16 scale (high)
            net = base_rate * (sav / 100) - oh
            sw = int((vl_c[1:] != vl_c[:-1]).sum())
            log(f"  Arm 0 (B=16 standalone) | γ={gamma:.0f}: GFT={sav:.3f}% OH={oh:.5f} Net={net:+.5f} sw={sw}")
            results[0].append((gamma, sav, oh, net, sw))
        log("  ★ This is the target to beat. Arms 1 & 2 use B=32 overhead (~%.5f) with B=16 GFTs." %
            (compute_entropy(np.zeros(Np, dtype=int), K) * Np / total_v))

    # ===========================================================================
    # ARM 1: Top-Down — Parent B=32 gradient → B=16 child GFTs
    # ===========================================================================
    if 1 in arms:
        section("ARM 1 — Top-Down Hierarchical (B=32 gradient → B=16 child GFTs)")
        log("Training centroids on parent-scale gradients (B=32)...")
        np.random.seed(42)
        idx1 = np.random.choice(Np, K_hybrid, replace=False)
        centroids_arm1 = g3d_norm_p[idx1].copy()
        for _ in range(20):
            sims = np.abs(g3d_norm_p @ centroids_arm1.T)
            lbl = np.argmax(sims, axis=1)
            for k in range(K_hybrid):
                m = (lbl==k); sg = np.zeros(3); mu = centroids_arm1[k]
                for i in np.where(m)[0]: sg += np.sign(np.dot(g3d_norm_p[i], mu))*g3d_norm_p[i]
                n = np.linalg.norm(sg)
                if n > 1e-8: centroids_arm1[k] = sg/n

        # Build initial Arm 1 cost matrix with slp/slw defaults
        slp_init1 = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
        slw_init1 = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
        log("Building Arm 1 initial cost matrix (slp/slw defaults)...")
        t0 = time.time()
        cost_matrix_arm1 = build_cost_matrix_hierarchical(
            blocks_p, blocks_c, parent_to_children,
            tangent_p, tangent_c, struct_map_c,
            centroids_arm1, slp_init1, slw_init1,
            g3d_norm_p, g3d_raw_p,
            pc.V, pc.A, dm, lag, q, K)
        log(f"  Arm 1 initial cost matrix done in {time.time()-t0:.1f}s")
        struct_total_arm1 = cost_matrix_arm1[:, 0].sum()

        gains_arm1 = (cost_matrix_arm1[:, 0] - cost_matrix_arm1[:, 1:].min(axis=1)) / (cost_matrix_arm1[:, 0] + 1e-12) * 100
        active_mask_arm1 = gains_arm1 > tau
        n_active = active_mask_arm1.sum()
        log(f"  Gain-filter τ={tau}%: Active={n_active} ({100*n_active/Np:.1f}%) parents")

        if n_active >= K_hybrid * 3:
            active_grads_p = g3d_norm_p[active_mask_arm1]
            centroids_arm1, _ = spherical_kmeans(active_grads_p, K_hybrid)

            # Grid search on child blocks that belong to active parents
            # Build the child-level active mask: a child is active if its parent is active
            child_active_mask_arm1 = np.zeros(Nc, dtype=bool)
            for pi in np.where(active_mask_arm1)[0]:
                for ci in parent_to_children[pi]:
                    child_active_mask_arm1[ci] = True
            log(f"  Active children under active parents: {child_active_mask_arm1.sum()}")

            log("  Grid search slp/slw on active children (matching E1 rigour)...")
            slp_arm1, slw_arm1 = grid_search(
                blocks_c, tangent_c, centroids_arm1, g3d_raw_c, g3d_norm_c,
                pc.V, pc.A, dm, lag, q, K, child_active_mask_arm1)

            log("  Rebuilding Arm 1 cost matrix with grid-searched slp/slw...")
            t0 = time.time()
            cost_matrix_arm1 = build_cost_matrix_hierarchical(
                blocks_p, blocks_c, parent_to_children,
                tangent_p, tangent_c, struct_map_c,
                centroids_arm1, slp_arm1, slw_arm1,
                g3d_norm_p, g3d_raw_p,
                pc.V, pc.A, dm, lag, q, K)
            struct_total_arm1 = cost_matrix_arm1[:, 0].sum()
            log(f"  Arm 1 retrain done in {time.time()-t0:.1f}s")
        else:
            slp_arm1, slw_arm1 = slp_init1, slw_init1
            log(f"  SKIP retrain: fewer than {K_hybrid*3} active parents.")


        results[1] = []
        for gamma in gammas:
            vl = run_viterbi(cost_matrix_arm1, K, gamma, nw)
            gft_c = sum(cost_matrix_arm1[i, vl[i]] for i in range(Np))
            sav = (struct_total_arm1 - gft_c)/struct_total_arm1*100
            H = compute_entropy(vl, K)
            oh = H * Np / total_v
            net = base_rate*(sav/100) - oh
            sw = int((vl[1:] != vl[:-1]).sum())
            log(f"  Arm 1 | γ={gamma:.0f}: GFT={sav:.3f}% OH={oh:.5f} Net={net:+.5f} sw={sw}")
            results[1].append((gamma, sav, oh, net, sw))

    # ===========================================================================
    # ARM 2: Bottom-Up SVD Consensus — Child gradients → consensus parent → B=16 GFTs
    # ===========================================================================
    if 2 in arms:
        section("ARM 2 — Bottom-Up SVD Consensus (Child gradients → B=16 child GFTs)")
        log("Training centroids on SVD consensus gradients...")
        np.random.seed(42)
        idx2 = np.random.choice(Np, K_hybrid, replace=False)
        centroids_arm2 = g3d_norm_consensus[idx2].copy()
        for _ in range(20):
            sims = np.abs(g3d_norm_consensus @ centroids_arm2.T)
            lbl = np.argmax(sims, axis=1)
            for k in range(K_hybrid):
                m = (lbl==k); sg = np.zeros(3); mu = centroids_arm2[k]
                for i in np.where(m)[0]: sg += np.sign(np.dot(g3d_norm_consensus[i], mu))*g3d_norm_consensus[i]
                n = np.linalg.norm(sg)
                if n > 1e-8: centroids_arm2[k] = sg/n

        # Build initial Arm 2 cost matrix with slp/slw defaults
        slp_init2 = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
        slw_init2 = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])
        log("Building Arm 2 initial cost matrix (slp/slw defaults)...")
        t0 = time.time()
        cost_matrix_arm2 = build_cost_matrix_hierarchical(
            blocks_p, blocks_c, parent_to_children,
            tangent_p, tangent_c, struct_map_c,
            centroids_arm2, slp_init2, slw_init2,
            g3d_norm_consensus, g3d_raw_p,
            pc.V, pc.A, dm, lag, q, K)
        log(f"  Arm 2 initial cost matrix done in {time.time()-t0:.1f}s")
        struct_total_arm2 = cost_matrix_arm2[:, 0].sum()

        gains_arm2 = (cost_matrix_arm2[:, 0] - cost_matrix_arm2[:, 1:].min(axis=1)) / (cost_matrix_arm2[:, 0] + 1e-12) * 100
        active_mask_arm2 = gains_arm2 > tau
        n_active = active_mask_arm2.sum()
        log(f"  Gain-filter τ={tau}%: Active={n_active} ({100*n_active/Np:.1f}%) parents")

        if n_active >= K_hybrid * 3:
            active_grads_c2 = g3d_norm_consensus[active_mask_arm2]
            centroids_arm2, _ = spherical_kmeans(active_grads_c2, K_hybrid)

            # Grid search on child blocks under active parents
            child_active_mask_arm2 = np.zeros(Nc, dtype=bool)
            for pi in np.where(active_mask_arm2)[0]:
                for ci in parent_to_children[pi]:
                    child_active_mask_arm2[ci] = True
            log(f"  Active children under active parents: {child_active_mask_arm2.sum()}")

            log("  Grid search slp/slw on active children (matching E1 rigour)...")
            slp_arm2, slw_arm2 = grid_search(
                blocks_c, tangent_c, centroids_arm2, g3d_raw_c, g3d_norm_c,
                pc.V, pc.A, dm, lag, q, K, child_active_mask_arm2)

            log("  Rebuilding Arm 2 cost matrix with grid-searched slp/slw...")
            t0 = time.time()
            cost_matrix_arm2 = build_cost_matrix_hierarchical(
                blocks_p, blocks_c, parent_to_children,
                tangent_p, tangent_c, struct_map_c,
                centroids_arm2, slp_arm2, slw_arm2,
                g3d_norm_consensus, g3d_raw_p,
                pc.V, pc.A, dm, lag, q, K)
            struct_total_arm2 = cost_matrix_arm2[:, 0].sum()
            log(f"  Arm 2 retrain done in {time.time()-t0:.1f}s")
        else:
            slp_arm2, slw_arm2 = slp_init2, slw_init2
            log(f"  SKIP retrain: fewer than {K_hybrid*3} active parents.")


        results[2] = []
        for gamma in gammas:
            vl = run_viterbi(cost_matrix_arm2, K, gamma, nw)
            gft_c = sum(cost_matrix_arm2[i, vl[i]] for i in range(Np))
            sav = (struct_total_arm2 - gft_c)/struct_total_arm2*100
            H = compute_entropy(vl, K)
            oh = H * Np / total_v
            net = base_rate*(sav/100) - oh
            sw = int((vl[1:] != vl[:-1]).sum())
            log(f"  Arm 2 | γ={gamma:.0f}: GFT={sav:.3f}% OH={oh:.5f} Net={net:+.5f} sw={sw}")
            results[2].append((gamma, sav, oh, net, sw))

    # ===========================================================================
    # Summary Table
    # ===========================================================================
    section("COMPARATIVE RESULTS TABLE")
    print()
    print("  Objective: Arms 1 & 2 use B=32 grouped signaling (N=834) with B=16 child GFTs.")
    print("  Target:    Beat B=16 standalone baseline (Arm 0, N=3480, overhead-limited).")
    print()

    arm_labels = {
        0: "Arm0 B=16 Standalone",
        1: "Arm1 TopDown B=32→16",
        2: "Arm2 SVD Consensus  "
    }

    header = f"{'Arm':>22} | {'γ':>6} | {'N signals':>9} | {'GFT%':>7} | {'Overhead':>10} | {'Net BPV':>10} | {'ΔNet vs B16':>12}"
    print(header)
    print("-" * len(header))

    # Arm 0 reference
    ref_net = {}  # gamma → B=16 standalone net
    if 0 in results:
        for gamma, sav, oh, net, sw in results[0]:
            ref_net[gamma] = net
            print(f"{'Arm0 B=16 Standalone':>22} | {gamma:>6.0f} | {Nc:>9} | "
                  f"{sav:>6.3f}% | {oh:>10.5f} | {net:>+10.5f} | {'[BASELINE]':>12}")

    for arm in [1, 2]:
        if arm not in results: continue
        for gamma, sav, oh, net, sw in results[arm]:
            dn = net - ref_net.get(gamma, 0.0)
            verdict = "✓" if dn > 0.00010 else ("~" if abs(dn) <= 0.00010 else "✗")
            label = arm_labels[arm]
            print(f"{label:>22} | {gamma:>6.0f} | {Np:>9} | "
                  f"{sav:>6.3f}% | {oh:>10.5f} | {net:>+10.5f} | {dn:>+12.5f} {verdict}")

    section("HYPOTHESIS EVALUATION")
    print("  Hypothesis: B=32-level signaling + B=16 GFT execution > B=16 standalone")
    for gamma in gammas:
        rn = ref_net.get(gamma)
        if rn is None: continue
        print(f"\n  γ={gamma:.0f} (B=16 standalone net = {rn:+.5f}):")
        for arm in [1, 2]:
            if arm not in results: continue
            row = next((r for r in results[arm] if r[0] == gamma), None)
            if row is None: continue
            dn = row[3] - rn
            if dn > 0.00050:
                verdict = "✓✓ STRONG SUCCESS (overhead rescue works)"
            elif dn > 0.00010:
                verdict = "✓  MODERATE SUCCESS"
            elif abs(dn) <= 0.00010:
                verdict = "~  NEUTRAL"
            else:
                verdict = "✗  NOT SUPPORTED"
            print(f"    Arm{arm}: Net={row[3]:+.5f}  ΔNet vs B=16={dn:+.5f}  {verdict}")



if __name__ == "__main__":
    main()
