"""
Geometry-Defined Segment Feasibility Test
==========================================
Tests whether deterministic geometry rules can define Morton-order segments
that are decodable with ZERO extra overhead (no transmitted boundaries).

Decoder reproducibility guarantee:
  Both encoder and decoder have: geometry V, block size B, Morton order.
  Any fixed function of (V, B, Morton) needs no extra bits.
  We test two families of rules:

  RULE 1 — Normal-angle break (Morton-sequential):
    A new segment starts at block i when:
       angle(normal[i], normal[i-1]) > theta_break
    Threshold theta_break ∈ {10°, 20°, 30°, 45°, 60°, 75°}
    → low theta = many breaks (many short segments, many labels to transmit)
    → high theta = few breaks (fewer longer segments, fewer labels to transmit)
    This is purely from PCA normals of point positions. No colour data.

  RULE 2 — Gradient-energy gate (per-block, independent):
    Blocks with gradient energy below a fixed percentile threshold are FORCED
    to Cluster 0 (structural/flat), excluded from Viterbi.
    The remaining blocks are Viterbi-decoded normally (or with segments).
    Threshold: percentile ∈ {20, 30, 40, 50}
    → A block with near-zero colour gradient gains nothing from an attribute
      transform — forcing it flat is safe and reduces the sequence to decode.

  RULE 3 — Combined: energy gate + normal-angle segments
    Apply Rule 2 first, then Rule 1 on the remaining non-flat blocks.

For each (rule, threshold):
  A. Compute M = number of segments and mean segment size
  B. For each segment, assign the MAJORITY Viterbi label from the block-level
     Viterbi (baseline). Report segment PURITY = fraction of blocks in segment
     that agree with the majority — measures how well the geometry rule
     predicts transform assignment without running Viterbi on segments.
  C. Compute GFT cost using majority-label assignment (cost from Phase 2 matrix)
  D. Compute overhead = H(segment_labels) × M / total_v
  E. Report net BPV vs block-level baseline

The test does NOT require re-running the GFT evaluation per rule variation.
We reuse the Phase 2 cost matrix computed once at the start.
All rule comparisons are analytical. Runtime: ~5 min total.
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
# Geometry helpers (pure — no colour data)
# ---------------------------------------------------------------------------

def normal_angle_deg(n1, n2):
    return float(np.degrees(np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0, 1))))


def compute_entropy(labels, K):
    if len(labels) == 0: return 0.0
    cs = np.bincount(labels, minlength=K)
    tr = np.ones((K, K))
    for i in range(1, len(labels)):
        tr[labels[i-1], labels[i]] += 1
    tp = tr / tr.sum(axis=1, keepdims=True)
    h = 0.0
    for p in range(K):
        pp = cs[p] / len(labels)
        for c in range(K):
            v = tp[p, c]
            if v > 0 and pp > 0: h -= pp * v * np.log2(v)
    return h


# ---------------------------------------------------------------------------
# Segmentation rules (all pure geometry, no colour)
# ---------------------------------------------------------------------------

def segment_by_normal_angle(normals, theta_break_deg, forced_flat=None):
    """
    Break the Morton sequence when consecutive normal angle > theta_break_deg.
    forced_flat: boolean mask — blocks forced to Cluster 0, always a segment break.
    Returns: segment_id per block (int array), list of (start, end_excl) tuples.
    """
    N = len(normals)
    seg_id = np.zeros(N, dtype=int)
    segments = []
    cur_seg = 0
    seg_start = 0
    for i in range(1, N):
        forced_break = (forced_flat is not None) and (forced_flat[i] or forced_flat[i-1])
        angle = normal_angle_deg(normals[i], normals[i-1])
        if angle > theta_break_deg or forced_break:
            segments.append((seg_start, i))
            cur_seg += 1
            seg_start = i
        seg_id[i] = cur_seg
    segments.append((seg_start, N))
    return seg_id, segments


def gradient_energy_gate(g3d_raw_norms, percentile_threshold):
    """
    Returns a boolean mask: True = block is forced flat (low energy).
    Purely from point positions (gradient from geometry, not from colour).
    NOTE: gradient is computed from GEOMETRY slope direction — uses colour implicitly.
    
    For a truly geometry-only gate we use PLANARITY instead (below).
    Both are tested.
    """
    thresh = np.percentile(g3d_raw_norms, percentile_threshold)
    return g3d_raw_norms < thresh


def planarity_gate(planarities, percentile_threshold):
    """
    Force flat blocks with low planarity (< percentile threshold).
    Planarity is PURELY geometric (PCA of point positions, no colour).
    """
    thresh = np.percentile(planarities, percentile_threshold)
    return planarities < thresh


def compute_planarity(Vblock):
    V_c = Vblock - Vblock.mean(0)
    cov = V_c.T @ V_c / len(Vblock)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    l1, l2, l3 = eigvals
    return float(np.clip((l2 - l3) / (l1 + 1e-12), 0, 1))


# ---------------------------------------------------------------------------
# Cost evaluation (one-time Phase 2 build)
# ---------------------------------------------------------------------------

def project_slope(mu_k, v1, v2, unc_norm, degen=1e-3):
    s3d = np.dot(mu_k, v1)*v1 + np.dot(mu_k, v2)*v2
    rn = np.linalg.norm(s3d)
    if rn < degen: s3d = v1*degen; rn = degen
    t = unc_norm if unc_norm > 1e-8 else rn
    return (s3d/rn)*t


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


def spherical_em(g3d_norm, K_hybrid, n_iter=20):
    np.random.seed(42)
    idx0 = np.random.choice(len(g3d_norm), K_hybrid, replace=False)
    centroids = g3d_norm[idx0].copy()
    labels = np.zeros(len(g3d_norm), dtype=int)
    for _ in range(n_iter):
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


# ---------------------------------------------------------------------------
# Segment analytics
# ---------------------------------------------------------------------------

def analyse_segments(seg_id, segments, vit_labels, cost_matrix, K,
                     N, total_v, label_name=""):
    """
    Given segments (list of (start,end) tuples) and block-level Viterbi labels,
    compute:
      - purity: fraction of blocks agreeing with segment majority label
      - GFT cost using majority label per segment
      - overhead using segment-level Markov entropy
      - net BPV
    """
    M = len(segments)
    seg_labels = np.zeros(M, dtype=int)  # majority label per segment
    purities = []
    seg_gft_cost = 0.0

    for s_idx, (start, end) in enumerate(segments):
        block_lbls = vit_labels[start:end]
        counts = np.bincount(block_lbls, minlength=K)
        majority = np.argmax(counts)
        purity = counts[majority] / max(1, end - start)
        purities.append(purity)
        seg_labels[s_idx] = majority
        # GFT cost: use majority label for all blocks in this segment
        for i in range(start, end):
            seg_gft_cost += cost_matrix[i, majority]

    mean_purity = np.mean(purities)
    # Overhead: H(segment_label_sequence) × M / total_v
    H_seg = compute_entropy(seg_labels, K)
    overhead = H_seg * M / total_v
    # Structural cost
    structural_cost = cost_matrix[:, 0].sum()
    savings_pct = (structural_cost - seg_gft_cost) / structural_cost * 100
    net_bpv = None  # caller sets this relative to base_rate

    seg_sizes = [end - start for start, end in segments]
    return {
        "label": label_name,
        "M": M,
        "mean_size": np.mean(seg_sizes),
        "median_size": int(np.median(seg_sizes)),
        "min_size": min(seg_sizes),
        "max_size": max(seg_sizes),
        "mean_purity": mean_purity,
        "H_seg": H_seg,
        "overhead": overhead,
        "savings_pct": savings_pct,
        "gft_cost": seg_gft_cost,
    }


# ---------------------------------------------------------------------------
# Viterbi (block-level, for baseline)
# ---------------------------------------------------------------------------

def run_viterbi(cost_matrix, K, gamma, normal_weights):
    N = len(cost_matrix)
    dp = np.full((N, K), np.inf); paths = np.zeros((N,K), dtype=int)
    dp[0] = cost_matrix[0]
    for i in range(1, N):
        w = normal_weights[i]
        for k in range(K):
            tc = dp[i-1] + gamma*w*(np.arange(K)!=k)
            bp = np.argmin(tc)
            dp[i,k] = cost_matrix[i,k]+tc[bp]; paths[i,k] = bp
    vl = np.zeros(N, dtype=int); vl[N-1] = np.argmin(dp[N-1])
    for i in range(N-2,-1,-1): vl[i] = paths[i+1,vl[i+1]]
    return vl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    section(f"Geometry Segment Feasibility Test  (B={B}, K={K}, γ={gamma})")
    log("All segment rules use ONLY point positions (geometry). No colour data.")
    log("Zero extra bits needed at decoder — rules are fixed deterministic functions.")

    # Setup
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

    # --- Extract all geometry features (NO colour used after this point) ---
    log("Extracting geometry features (normals, planarity, gradient norms)...")
    normals, planarities, g3d_norm_all, g3d_raw = [], [], [], []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        normals.append(tangent_data[i]["v3"])
        planarities.append(compute_planarity(b.Vblock))
        # Gradient uses COLOUR — this is the non-pure feature
        Vc = b.Vblock - b.Vblock.mean(0)
        X = np.column_stack((Vc @ tangent_data[i]["v1"], Vc @ tangent_data[i]["v2"]))
        Y = b.Ablock[:,0] - b.Ablock[:,0].mean()
        s2d, *_ = np.linalg.lstsq(X, Y, rcond=None)
        g = s2d[0]*tangent_data[i]["v1"] + s2d[1]*tangent_data[i]["v2"]
        rn = np.linalg.norm(g)
        g3d_raw.append(rn)
        g3d_norm_all.append(g/rn if rn>1e-8 else np.array([0.,0.,1.]))
        b.clear_data()
    normals = np.array(normals)
    planarities = np.array(planarities)
    g3d_raw = np.array(g3d_raw)
    g3d_norm_all = np.array(g3d_norm_all)

    # Normal angles along Morton path (geometry-only)
    normal_angles = np.array([normal_angle_deg(normals[i], normals[i-1])
                               for i in range(1, N)])
    log(f"  Morton normal-angle stats: "
        f"mean={normal_angles.mean():.1f}° median={np.median(normal_angles):.1f}° "
        f"p25={np.percentile(normal_angles,25):.1f}° "
        f"p75={np.percentile(normal_angles,75):.1f}° "
        f"p90={np.percentile(normal_angles,90):.1f}°")
    log(f"  Planarity: mean={planarities.mean():.3f} "
        f"p25={np.percentile(planarities,25):.3f} "
        f"p75={np.percentile(planarities,75):.3f}")

    # Structural GFT + Phase 2 cost matrix
    log("Structural GFT pre-computation...")
    gft_comp = GFTStrategyWraper(); struct_map = {}
    nw = np.ones(N)
    for i in range(1, N): nw[i] = abs(np.dot(normals[i], normals[i-1]))
    for b in tqdm(blocks, desc="Structural GFT", ncols=70):
        b.init_data(pc.V, pc.A)
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, coeffs = gft_comp(b, sg); struct_map[b.block_id] = coeffs; b.clear_data()

    log("Spherical EM + Phase 2 cost matrix (computed once, shared by all tests)...")
    centroids_3d, _ = spherical_em(g3d_norm_all, K_hybrid)
    slp = np.concatenate([[0.0], np.full(K_hybrid, 0.30)])
    slw = np.concatenate([[1.0], np.full(K_hybrid, 0.80)])

    tasks = []
    for i, b in enumerate(blocks):
        b.init_data(pc.V, pc.A)
        tasks.append((b.block_id, b.Vblock.copy(), b.Ablock.copy(), b.metadata,
                      struct_map[b.block_id],
                      tangent_data[i]["v1"], tangent_data[i]["v2"],
                      centroids_3d, slp, slw, g3d_raw[i], dm, lag, q, K))
        b.clear_data()
    t0 = time.time()
    rows = Parallel(n_jobs=-1)(delayed(_eval_row)(*t) for t in tasks)
    cost_matrix = np.array(rows)
    log(f"  Cost matrix done in {time.time()-t0:.1f}s.")

    structural_cost = cost_matrix[:, 0].sum()

    # Baseline rate
    dec0 = Decider(mode="0", lagrange_proportional=0.8); dec0._set_vars(q)
    from pcadc.pointcloud import Sampler
    sampled = Sampler(ratio=0.01, n_strata=5)(pc.V, pc.A, blocks)
    tr, tv2 = 0.0, 0
    for b in sampled:
        b.init_data(pc.V, pc.A); tv2 += b.Vblock.shape[0]
        sg = StructuralGraph(b.metadata); sg.set_data(b.Vblock)
        _, cs = gft_comp(b, sg); _, r, _ = dec0._RDcost(cs); tr += r; b.clear_data()
    base_rate = tr / tv2
    log(f"  Baseline rate: {base_rate:.5f} bpv")

    # Block-level Viterbi baseline
    vit_labels = run_viterbi(cost_matrix, K, gamma, nw)
    vit_cost = sum(cost_matrix[i, vit_labels[i]] for i in range(N))
    vit_sav = (structural_cost - vit_cost)/structural_cost*100
    vit_H = compute_entropy(vit_labels, K)
    vit_oh = vit_H * N / total_v
    vit_net = base_rate*(vit_sav/100) - vit_oh
    vit_sw = int((vit_labels[1:]!=vit_labels[:-1]).sum())

    section("BLOCK-LEVEL BASELINE (current pipeline)")
    log(f"  γ={gamma:.0f}: GFT={vit_sav:.3f}% H={vit_H:.4f} "
        f"overhead={vit_oh:.5f} net={vit_net:+.5f} switches={vit_sw}")
    log(f"  N_sequence={N}  (labels transmitted: {N})")

    # =========================================================================
    # TEST GEOMETRY RULES
    # =========================================================================
    results = []  # list of dicts

    section("RULE 1 — Normal-Angle Break (pure geometry, no colour)")
    log("  Rule: start new segment when angle(normal[i], normal[i-1]) > theta")
    log("  Decoder reproducibility: YES — normals from PCA of point positions")
    log("  Extra overhead: ZERO — threshold is a fixed constant\n")

    theta_breaks = [10, 20, 30, 45, 60, 75]
    for theta in theta_breaks:
        seg_id, segs = segment_by_normal_angle(normals, theta)
        res = analyse_segments(seg_id, segs, vit_labels, cost_matrix, K, N, total_v,
                               label_name=f"normal_break_{theta}°")
        res["net_bpv"] = base_rate*(res["savings_pct"]/100) - res["overhead"]
        res["delta_net"] = res["net_bpv"] - vit_net
        # How many Morton transitions ARE breaks at this threshold?
        frac_breaks = (normal_angles > theta).mean()
        res["frac_breaks"] = frac_breaks
        results.append(res)

        log(f"  θ={theta:>3}°: M={res['M']:>4} segs | "
            f"mean_size={res['mean_size']:.1f} | "
            f"purity={res['mean_purity']:.3f} | "
            f"overhead={res['overhead']:.5f} | "
            f"GFT={res['savings_pct']:.3f}% | "
            f"net={res['net_bpv']:+.5f} | "
            f"Δnet={res['delta_net']:+.5f} | "
            f"break_frac={frac_breaks:.0%}")

    section("RULE 2a — Planarity Gate (PURE geometry, zero colour data)")
    log("  Rule: force low-planarity blocks to Cluster 0 before Viterbi")
    log("  Decoder reproducibility: YES — planarity = PCA eigenvalue ratio of V")
    log("  Extra overhead: ZERO — threshold is a fixed constant")
    log("  NOTE: the non-flat blocks still use block-level Viterbi (no segmenting)\n")

    for pct in [20, 30, 40, 50]:
        forced = planarity_gate(planarities, pct)
        # Re-run effective Viterbi: forced blocks always cluster 0
        effective_labels = vit_labels.copy()
        effective_labels[forced] = 0
        eff_cost = sum(cost_matrix[i, effective_labels[i]] for i in range(N))
        eff_sav = (structural_cost - eff_cost)/structural_cost*100
        eff_H = compute_entropy(effective_labels, K)
        eff_oh = eff_H * N / total_v
        eff_net = base_rate*(eff_sav/100) - eff_oh
        n_forced = forced.sum()
        # purity: fraction of forced blocks that Viterbi ALSO assigned to 0
        purity_agree = (vit_labels[forced] == 0).mean() if n_forced > 0 else 0.0
        log(f"  p{pct:>2}th planarity gate: "
            f"forced={n_forced}({100*n_forced/N:.0f}%) | "
            f"Viterbi-agreement={purity_agree:.0%} | "
            f"overhead={eff_oh:.5f} | "
            f"GFT={eff_sav:.3f}% | "
            f"net={eff_net:+.5f} | "
            f"Δnet={eff_net-vit_net:+.5f}")

    section("RULE 2b — Gradient Energy Gate (uses colour implicitly — flagged)")
    log("  Rule: force low-colour-gradient blocks to Cluster 0")
    log("  Decoder reproducibility: PARTIAL — gradient norm depends on colour")
    log("  This rule is flagged as colour-dependent, included only for comparison\n")

    for pct in [20, 30, 40, 50]:
        forced = gradient_energy_gate(g3d_raw, pct)
        effective_labels = vit_labels.copy()
        effective_labels[forced] = 0
        eff_cost = sum(cost_matrix[i, effective_labels[i]] for i in range(N))
        eff_sav = (structural_cost - eff_cost)/structural_cost*100
        eff_H = compute_entropy(effective_labels, K)
        eff_oh = eff_H * N / total_v
        eff_net = base_rate*(eff_sav/100) - eff_oh
        n_forced = forced.sum()
        purity_agree = (vit_labels[forced] == 0).mean() if n_forced > 0 else 0.0
        log(f"  p{pct:>2}th gradient gate [COLOUR-DEPENDENT]: "
            f"forced={n_forced}({100*n_forced/N:.0f}%) | "
            f"Viterbi-agreement={purity_agree:.0%} | "
            f"overhead={eff_oh:.5f} | "
            f"GFT={eff_sav:.3f}% | "
            f"net={eff_net:+.5f} | "
            f"Δnet={eff_net-vit_net:+.5f}")

    section("RULE 3 — Combined: Planarity Gate + Normal-Angle Segments")
    log("  Fully geometry-only, fully decodable, zero extra bits")
    log("  Step 1: force low-planarity blocks to Cluster 0")
    log("  Step 2: segment remaining blocks by normal-angle break\n")

    for p_pct in [30, 40]:
        forced = planarity_gate(planarities, p_pct)
        for theta in [30, 45]:
            seg_id, segs = segment_by_normal_angle(normals, theta, forced_flat=forced)
            # For forced-flat blocks, majority must be Cluster 0
            adjusted_vit = vit_labels.copy()
            adjusted_vit[forced] = 0
            res = analyse_segments(seg_id, segs, adjusted_vit, cost_matrix, K,
                                   N, total_v,
                                   label_name=f"plan_p{p_pct}_normal_{theta}°")
            res["net_bpv"] = base_rate*(res["savings_pct"]/100) - res["overhead"]
            res["delta_net"] = res["net_bpv"] - vit_net
            log(f"  planarity_p{p_pct} + normal_{theta}°: "
                f"M={res['M']:>4} segs | "
                f"mean_size={res['mean_size']:.1f} | "
                f"purity={res['mean_purity']:.3f} | "
                f"overhead={res['overhead']:.5f} | "
                f"GFT={res['savings_pct']:.3f}% | "
                f"net={res['net_bpv']:+.5f} | "
                f"Δnet={res['delta_net']:+.5f}")

    # =========================================================================
    # Summary
    # =========================================================================
    section("SUMMARY TABLE — Rule 1 (Normal-Angle Break, pure geometry)")
    print(f"\n{'θ':>5} | {'M segs':>6} | {'mean sz':>7} | {'purity':>7} | "
          f"{'overhead':>10} | {'GFT%':>7} | {'net BPV':>10} | {'Δnet':>10} | {'verdict'}")
    print("-" * 90)
    for r in results:
        theta = r["label"].replace("normal_break_","").replace("°","")
        verdict = ("✓ WIN" if r["delta_net"] > 0.00003 else
                   "✗ LOSS" if r["delta_net"] < -0.00003 else "~ DRAW")
        print(f"{theta:>5}° | {r['M']:>6} | {r['mean_size']:>7.1f} | "
              f"{r['mean_purity']:>7.3f} | {r['overhead']:>10.5f} | "
              f"{r['savings_pct']:>6.3f}% | {r['net_bpv']:>+10.5f} | "
              f"{r['delta_net']:>+10.5f} | {verdict}")

    print(f"\nBlock baseline: overhead={vit_oh:.5f} GFT={vit_sav:.3f}% "
          f"net={vit_net:+.5f}")

    section("VERDICT")
    best = max(results, key=lambda r: r["net_bpv"])
    log(f"Best geometry-defined segmentation: {best['label']}")
    log(f"  M={best['M']} segments (vs N={N} blocks = {N/best['M']:.1f}× reduction)")
    log(f"  Purity={best['mean_purity']:.3f} — "
        f"{'HIGH: geometry predicts transform well' if best['mean_purity']>0.75 else 'LOW: geometry is a weak predictor'}")
    log(f"  Net BPV improvement: {best['delta_net']:+.5f}")
    if best["delta_net"] > 0.0001:
        log("  → GEOMETRY RULE ADDS VALUE. Proceed to production implementation.")
    elif best["delta_net"] > 0:
        log("  → MARGINAL GAIN. Geometry rule marginally helps but not compelling.")
    else:
        log("  → NO BENEFIT. Geometry-defined segments do not improve over block-level.")
        log("    The overhead model is not the bottleneck. Focus on better transforms.")


if __name__ == "__main__":
    main()
