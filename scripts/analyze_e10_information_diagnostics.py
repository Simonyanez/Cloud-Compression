#!/usr/bin/env python3
"""
E10 information-theoretic diagnostics (post-hoc, read-only against the production algorithm).

Both analyses are built by CALLING the existing, unmodified encode-time primitives from
exp_e10_multisequence_benchmark.py / exp_e9_multi_entropy_suite.py / src/pcadc/* -- this script
never edits them, it only observes intermediate values (the self-loop sink vector, per-child
per-mode RD costs) that encode_frame() already computes and then discards.

1. MI(sink-vector selection, coefficient compaction): does the self-loop mechanism's own
   internal signal (which nodes get selected as sinks, Sec. Selección de self-loops) track the
   spectral compaction it's meant to produce?
2. H(child's independently-best mode | parent's inherited mode): how much does forcing every
   child block to inherit its parent's Viterbi-chosen mode (instead of choosing independently)
   cost, in bits, beyond what the aggregate bpv number already shows?

Scope (deliberately small -- a diagnostic sample, not a benchmark): 3 representative sequences
(longdress, ricardo9 -- both already used as the worked example in the existing
training-diagnostics figure; soldier -- a different 8iVFB subject for contrast), each at its own
best (B_c*, K=24) from tab:e10_bd_summary_multiframe, over a handful of frames spread across the
sequence.
"""
import sys
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from pcadc.pointcloud import MortonBlockPartition                                  # noqa: E402
from exp_e9_multi_entropy_suite import (                                            # noqa: E402
    RLGRDecider, extract_block_tangent, build_parent_child_map, run_viterbi,
)
from exp_e10_multisequence_benchmark import (                                       # noqa: E402
    load_ply_from_zip, make_pc, _worker_gft, GAMMA_VAL, ALPHA_GATE, ALPHA_RLGR,
)

DB = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
OUT_DIR = REPO_ROOT / "docs" / "thesis" / "img" / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best (B_c, K=24) per target sequence, taken from the multi-frame BD-Rate tables
# (Sección Ganancia de Codificación). "label" lets one real sequence appear twice under two
# different B_c (the item-14 supplementary larger-block-mesh run) without record collisions.
TARGETS = {
    "longdress":    {"name": "longdress", "B_c": 16, "K": 24},
    "ricardo9":     {"name": "ricardo9",  "B_c": 16, "K": 24},
    "soldier":      {"name": "soldier",   "B_c": 16, "K": 24},
    # Supplementary run (item 14): same sequence, coarser child mesh (B_c=32 instead of 16) --
    # tests whether the s_lp/compaction relationship sharpens or stays flat at a larger block,
    # where each child covers more surface and self-loop node count varies over a wider range.
    "longdress_B32": {"name": "longdress", "B_c": 32, "K": 24},
}
N_FRAMES_PER_SEQ = 3     # spread across the sequence, not all of it -- a diagnostic sample
Q_DIAG = 32              # one representative mid-range operating point


def conn():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def get_frames(c, name, n):
    """Pick n frame_ply values spread across the sequence, using rows already in frame_results
    (so we reuse exactly the frames the real benchmark already encoded)."""
    rows = c.execute("""
        SELECT DISTINCT fr.frame_idx, fr.frame_ply FROM frame_results fr
        JOIN sequences se ON se.seq_id = fr.seq_id
        WHERE se.name=? ORDER BY fr.frame_idx
    """, (name,)).fetchall()
    if not rows:
        return []
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    return [rows[i] for i in idx]


def load_codebook_and_adapters(c, seq_id, K, Q):
    centroids = np.array(c.execute("""
        SELECT cx, cy, cz FROM cluster_centroids WHERE seq_id=? AND K=? ORDER BY mode_idx
    """, (seq_id, K)).fetchall())
    slp = {0: 0.0}
    slw = {0: 0.0}
    for mode_idx, s_lp, s_lw in c.execute("""
        SELECT mode_idx, s_lp, s_lw FROM mode_adapters WHERE seq_id=? AND K=? AND Q=?
    """, (seq_id, K, Q)):
        slp[mode_idx] = s_lp
        slw[mode_idx] = s_lw
    slp_arr = np.array([slp.get(k, 0.0) for k in range(K)])
    slw_arr = np.array([slw.get(k, 0.0) for k in range(K)])
    return centroids, slp_arr, slw_arr


def analyze_frame(c, name, seq_id, depth, zip_path, frame_idx, frame_ply, B_p, B_c, K, Q):
    """Runs the same setup encode_frame() does for a single (frame, B_c, K, Q) point, but keeps
    the intermediate per-child data instead of discarding it. Returns per-child records with:
    sink-vector summary, coefficient energy compaction, own-best mode, and (after Viterbi) the
    actually-assigned parent mode."""
    V_raw, A_raw = load_ply_from_zip(zip_path, frame_ply)
    pc = make_pc(V_raw, A_raw, name, depth, frame_idx)
    from pcadc.color import Colourist
    pc.transform_attributes(Colourist()._RGBtoYUV)
    pc_V, pc_A = pc.V.copy(), pc.A.copy()

    _, blocks_p = MortonBlockPartition().partition(pc, bsize=B_p)
    N_p = len(blocks_p)
    frame_tang_p = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_p]
    nw_frame = np.zeros(N_p)
    for i in range(1, N_p):
        nw_frame[i] = abs(float(np.dot(frame_tang_p[i]["v3"], frame_tang_p[i - 1]["v3"])))
    parent_var = np.array([
        float(np.var(pc_A[b.metadata.start:b.metadata.end, 0])) if b.metadata.end > b.metadata.start else 0.0
        for b in blocks_p
    ])

    _, blocks_c = MortonBlockPartition().partition(pc, bsize=B_c)
    N_c = len(blocks_c)
    p2c = build_parent_child_map(blocks_p, blocks_c, pc_V, B_p)
    child_tang = [extract_block_tangent(b, pc_V, pc_A) for b in blocks_c]
    child_data = []
    for b in blocks_c:
        b.init_data(pc_V, pc_A)
        child_data.append((b.Vblock.copy(), b.Ablock.copy()))
        b.clear_data()

    centroids, slp, slw = load_codebook_and_adapters(c, seq_id, K, Q)

    dec_master = RLGRDecider("0", 0.8, alpha_rlgr=ALPHA_RLGR)
    dec_master._set_vars(Q)

    # Same call encode_frame() makes for every child block, dispatched in parallel the same
    # way (loky backend) -- unmodified, just kept here instead of discarded.
    cr = Parallel(n_jobs=-1, backend="loky", batch_size=1)(
        delayed(_worker_gft)(
            ci, child_data[ci][0], child_data[ci][1], blocks_c[ci].metadata,
            child_tang[ci]["v1"], child_tang[ci]["v2"], child_tang[ci]["g_raw"],
            centroids, slp, slw, K,
        )
        for ci in range(N_c)
    )
    ccd = {ci: cmap for ci, cmap in cr}

    # Per-child, per-mode cost (own-best mode if evaluated independently of its parent).
    child_cost = np.full((N_c, K), np.nan)
    child_compaction = {}   # (ci, k) -> fraction of Y-channel coefficient energy in first 4 coeffs
    for ci in range(N_c):
        for k in range(K):
            coeffs = ccd[ci].get(k)
            if coeffs is None:
                continue
            cv, _, _ = dec_master._RDcost(coeffs)
            child_cost[ci, k] = cv
            y = np.asarray(coeffs)[:, 0] if coeffs.ndim > 1 else np.asarray(coeffs)
            e = y ** 2
            total = e.sum()
            child_compaction[(ci, k)] = float(e[:4].sum() / total) if total > 1e-12 else np.nan

    child_own_best = np.nanargmin(child_cost, axis=1)

    # Parent-level aggregation + gating + Viterbi -- identical to encode_frame().
    pcm = np.zeros((N_p, K))
    for pi, cl in p2c.items():
        for ci in cl:
            for k in range(K):
                if not np.isnan(child_cost[ci, k]):
                    pcm[pi, k] += child_cost[ci, k]
    tau_gate = ALPHA_GATE * ((Q / 32.0) ** 2)
    active = parent_var >= tau_gate
    gated_cost = pcm.copy()
    for pi in range(N_p):
        if not active[pi]:
            gated_cost[pi, 1:] = 1e18
    gamma_q = GAMMA_VAL * ((Q / 24.0) ** 2)
    mode_seq = run_viterbi(gated_cost, K, gamma_q, nw_frame)

    c2p = {ci: pi for pi, cl in p2c.items() for ci in cl}
    records = []
    for ci in range(N_c):
        pi = c2p.get(ci)
        if pi is None:
            continue
        assigned = int(mode_seq[pi])
        own_best = int(child_own_best[ci])
        # Sink-vector summary and compaction, evaluated at the mode this child was ACTUALLY
        # assigned (the mode whose self-loop graph really shaped its transform).
        n_sel = None
        if assigned >= 1 and slp[assigned] > 0:
            n_sel = int(slp[assigned] * len(pc_V[blocks_c[ci].metadata.start:blocks_c[ci].metadata.end]))
        compaction = child_compaction.get((ci, assigned))
        # Item 14: Y_std per child block (signal variable, captured regardless of gating so
        # gated/structural-only blocks -- assigned_mode==0 -- stay in the record too).
        y_block = child_data[ci][1][:, 0]
        y_std = float(np.std(y_block)) if len(y_block) > 0 else float("nan")
        if compaction is not None and not np.isnan(compaction):
            records.append({
                "n_selfloop_frac": float(slp[assigned]) if assigned >= 1 else 0.0,
                "n_selfloop_nodes": n_sel if n_sel is not None else 0,
                "compaction": compaction,
                "own_best_mode": own_best,
                "assigned_mode": assigned,
                "y_std": y_std,
                "n_points": int(len(y_block)),
            })
    return records


def main():
    c = conn().cursor()
    all_records = []
    for label, cfg in TARGETS.items():
        name = cfg["name"]
        row = c.execute("SELECT seq_id, depth, zip_path FROM sequences WHERE name=?", (name,)).fetchone()
        seq_id, depth, zip_path = row
        B_p = 32 if depth == 10 else 16
        frames = get_frames(c, name, N_FRAMES_PER_SEQ)
        print(f"[{label}] seq_id={seq_id} depth={depth} B_c={cfg['B_c']} K={cfg['K']} "
              f"frames={[f[0] for f in frames]}")
        for frame_idx, frame_ply in frames:
            recs = analyze_frame(c, name, seq_id, depth, zip_path, frame_idx, frame_ply,
                                  B_p, cfg["B_c"], cfg["K"], Q_DIAG)
            for r in recs:
                r["sequence"] = label
                r["frame_idx"] = frame_idx
            all_records.extend(recs)
            print(f"    frame {frame_idx}: {len(recs)} child blocks captured")

    print(f"\nTotal child-block records: {len(all_records)}")
    with open(REPO_ROOT / "scripts" / "_e10_info_diag_records.json", "w") as f:
        json.dump(all_records, f, indent=1)

    # -----------------------------------------------------------------
    # Item 1 (reformulated, item 14): MI(sink-vector fraction s_lp of the assigned mode,
    # coefficient compaction), now as box plots per s_lp bucket PER SEQUENCE-LABEL (the raw
    # scatter was unreadable at n>20k) and with Y_std correlation alongside s_lp.
    # -----------------------------------------------------------------
    from sklearn.feature_selection import mutual_info_regression
    directional = [r for r in all_records if r["assigned_mode"] >= 1]
    X = np.array([[r["n_selfloop_frac"]] for r in directional])
    y = np.array([r["compaction"] for r in directional])
    mi = mutual_info_regression(X, y, random_state=0)[0]
    pearson_r = float(np.corrcoef(X[:, 0], y)[0, 1]) if len(y) > 2 else float("nan")
    print(f"\nITEM 1 (pooled): MI(s_lp, compaction) = {mi:.4f} nats "
          f"(n={len(directional)} directional child blocks, Pearson r={pearson_r:+.3f})")

    SLP_GRID = [0.05, 0.25, 0.60]  # the coarse-grid values s_lp actually takes (Sec. adapter_rdo_real)

    def nearest_slp_bucket(v):
        return min(SLP_GRID, key=lambda g: abs(g - v))

    labels = ["longdress", "ricardo9", "soldier", "longdress_B32"]
    labels = [l for l in labels if any(r["sequence"] == l for r in directional)]
    per_seq_item1 = {}
    for lname in labels:
        pts = [r for r in directional if r["sequence"] == lname]
        Xs = np.array([[r["n_selfloop_frac"]] for r in pts])
        ys = np.array([r["compaction"] for r in pts])
        ystds = np.array([r["y_std"] for r in pts])
        mi_s = float(mutual_info_regression(Xs, ys, random_state=0)[0]) if len(pts) > 2 else float("nan")
        r_s = float(np.corrcoef(Xs[:, 0], ys)[0, 1]) if len(pts) > 2 else float("nan")
        r_ystd = float(np.corrcoef(ystds, ys)[0, 1]) if len(pts) > 2 else float("nan")
        per_seq_item1[lname] = {"mi_nats": mi_s, "pearson_r_slp": r_s,
                                 "pearson_r_ystd": r_ystd, "n": len(pts)}
        print(f"  [{lname:15s}] MI(s_lp,C)={mi_s:.4f} nats  r(s_lp,C)={r_s:+.3f}  "
              f"r(Y_std,C)={r_ystd:+.3f}  n={len(pts)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(labels), figsize=(3.6 * len(labels), 4.2), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, lname in zip(axes, labels):
        pts = [r for r in directional if r["sequence"] == lname]
        by_bucket = defaultdict(list)
        for r in pts:
            by_bucket[nearest_slp_bucket(r["n_selfloop_frac"])].append(r["compaction"])
        data = [by_bucket[g] for g in SLP_GRID]
        ax.boxplot(data, positions=range(len(SLP_GRID)), widths=0.55, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#4C78A833"))
        ax.set_xticks(range(len(SLP_GRID)))
        ax.set_xticklabels([str(g) for g in SLP_GRID])
        ax.set_title(f"{lname}\nMI={per_seq_item1[lname]['mi_nats']:.3f} nats", fontsize=9)
        ax.set_xlabel("$s_{lp}$ del modo asignado")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("compactación espectral (energía en 4 primeros coef.)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_selfloop_compaction_mi.png", dpi=300)
    plt.close(fig)
    print("  [+] e10_selfloop_compaction_mi.png (box plots by s_lp bucket, per sequence-label)")

    # -----------------------------------------------------------------
    # Item 2 (reformulated, item 15): H(child's own-best mode | parent's assigned mode),
    # now PER SEQUENCE-LABEL (not just pooled), plus normalized MI (symmetric uncertainty) and
    # a correlation of agreement rate with Y_std.
    # -----------------------------------------------------------------
    K = TARGETS["longdress"]["K"]

    def entropy_from_counts(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def h_hc_mi_nmi(recs, K):
        own = np.array([r["own_best_mode"] for r in recs])
        assigned = np.array([r["assigned_mode"] for r in recs])
        h_own = entropy_from_counts(np.bincount(own, minlength=K).astype(float))
        h_assigned = entropy_from_counts(np.bincount(assigned, minlength=K).astype(float))
        joint = np.zeros((K, K))
        for a, b in zip(assigned, own):
            joint[a, b] += 1
        h_joint = entropy_from_counts(joint.flatten())
        mi_modes = h_own + h_assigned - h_joint
        h_own_given_assigned = h_own - mi_modes
        # Normalized MI (symmetric uncertainty): 2*I / (H(own)+H(assigned)), in [0,1].
        nmi = float(2 * mi_modes / (h_own + h_assigned)) if (h_own + h_assigned) > 0 else float("nan")
        agree = float(np.mean(own == assigned))
        ystds = np.array([r["y_std"] for r in recs])
        r_ystd_agree = float(np.corrcoef(ystds, (own == assigned).astype(float))[0, 1]) if len(recs) > 2 else float("nan")
        return {"h_own_bits": h_own, "h_own_given_assigned_bits": h_own_given_assigned,
                "mi_bits": mi_modes, "nmi": nmi, "agreement_rate": agree,
                "y_std_mean": float(np.mean(ystds)), "r_ystd_vs_agreement": r_ystd_agree, "n": len(recs)}

    pooled = h_hc_mi_nmi(all_records, K)
    print(f"\nITEM 2 (pooled): H(own)={pooled['h_own_bits']:.3f}  "
          f"H(own|assigned)={pooled['h_own_given_assigned_bits']:.3f}  "
          f"MI={pooled['mi_bits']:.3f} bits  NMI={pooled['nmi']:.3f}  "
          f"agreement={pooled['agreement_rate']:.1%}  (n={pooled['n']})")

    per_seq_item2 = {}
    for lname in labels:
        recs = [r for r in all_records if r["sequence"] == lname]
        per_seq_item2[lname] = h_hc_mi_nmi(recs, K)
        s = per_seq_item2[lname]
        print(f"  [{lname:15s}] H(own)={s['h_own_bits']:.3f} H(own|assigned)="
              f"{s['h_own_given_assigned_bits']:.3f} MI={s['mi_bits']:.3f} NMI={s['nmi']:.3f} "
              f"agree={s['agreement_rate']:.1%} Y_std_mean={s['y_std_mean']:.2f} "
              f"r(Y_std,agree)={s['r_ystd_vs_agreement']:+.3f} n={s['n']}")

    with open(REPO_ROOT / "scripts" / "_e10_info_diag_summary.json", "w") as f:
        json.dump({
            "item1_pooled": {"mi_nats": mi, "pearson_r": pearson_r, "n": len(directional)},
            "item1_per_seq": per_seq_item1,
            "item2_pooled": pooled,
            "item2_per_seq": per_seq_item2,
        }, f, indent=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
