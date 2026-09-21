#!/usr/bin/env python3
"""
Part 0 of the real-RLGR / sensitivity-analysis plan
(docs/thesis observations: "the Oracle rate is a proxy, check the real BPV" + "Figura 5.2 doesn't
convey a clear message, replace with a sensitivity analysis").

Recomputes real per-block GFT coefficients at the LINEAR ORACLE's ALREADY-CHOSEN (best_p, best_w)
per (block_id, bsize, q_step) -- reusing linear_oracle_upper_bound_full.py's own graph-construction
code (compute_sink_vector, the same structural-adjacency/self-loop/eigh math) for self-consistency
with the stored best_p/best_w/oracle_dist, rather than the general src/pcadc encoder classes (whose
disconnected-block routing and int()-truncated node count would silently diverge from what the
oracle search actually assumed). Also recomputes the STRUCTURAL BASELINE (p=0, no self-loops) the
same way, for a real (not proxy) baseline curve to compare the oracle against. From those real
coefficients:

  (a) quantizes + RLGR-encodes the Y-channel coefficients for real bits -- explicitly WITHOUT any
      CABAC/mode-index signaling overhead (the idealized "decoder already knows (p,w)" case) --
      for both the oracle and the baseline configuration, feeding the real-only RD curve + real
      BD-Rate/BD-PSNR (Part 1), and
  (b) computes, per block at one representative Q, real spectral-compaction (energy in the first
      4 GFT coefficients), real sparsity (nonzero quantized coefficients), a real per-block RLGR
      bpv (individual cold-start encode -- see note in the driver), and a geometric planarity
      driver (smallest/largest eigenvalue ratio of the block's point covariance -- how flat the
      point geometry itself is, independent of the luminance signal), feeding the sensitivity
      analysis that replaces Figura 5.2 (Part 2).

Data source (read-only): results/linear_oracle_upper_bound.db, and (mode=absolute, sensitivity
records skipped -- see below) results/oracle_upper_bound.db, both 291,588 rows, one fixed point
cloud -- res/longdress_vox10_1051.ply, frame 1051, depth 10. Never re-runs the (p,w) search --
that's already done; this only re-derives what those already-chosen configurations really cost.

Usage: `python3 oracle_real_rlgr_and_compaction.py [linear|absolute]` (default linear). The two
oracles differ only in what signal drives self-loop node SELECTION (Linear: the plane fit to Y;
Absolute: the real Y directly) -- everything downstream (structural graph, self-loop injection,
eigh, RLGR encoding) is identical code, reused as-is for both. Absolute mode skips the plane-fit
(fit_rmse/slopes) and the per-block sensitivity records: only the real oracle-vs-baseline RD
curve was requested for Absolute Oracle, not a repeat of the Linear-only sensitivity analysis.
"""
import sys
import json
import time
import sqlite3
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcadc.pointcloud import PointCloud, MortonBlockPartition          # noqa: E402
from pcadc.color import Colourist                                      # noqa: E402
from pcadc.parameters import load_experiment_config                    # noqa: E402
from pcadc.decider import Decider                                      # noqa: E402

from exp_e10_multisequence_benchmark import compress_to_bytes          # noqa: E402  (real RLGR)

# `compute_sink_vector` is copied verbatim from linear_oracle_upper_bound_full.py (byte-for-byte,
# not reimplemented) rather than imported -- that module imports `pandas` at the top level purely
# for an unused convenience import, and pandas isn't installed in this venv (confirmed elsewhere
# this session). This one function has no pandas dependency itself.
def compute_sink_vector(W, Y):
    M = W * np.subtract.outer(Y, Y)
    M_masked = np.copy(M); M_masked[W == 0] = np.inf; np.fill_diagonal(M_masked, np.inf)
    dec_i = np.argmin(M_masked, axis=1)
    unique, counts = np.unique(dec_i, return_counts=True)
    S = np.zeros(W.shape[0]); S[unique] = counts
    max_val = np.max(S)
    return S / max_val if max_val > 0 else S

SCRATCH_DIR = REPO_ROOT / "experiments" / "oracle_rlgr_scratch"

MODE = sys.argv[1] if len(sys.argv) > 1 else "linear"
assert MODE in ("linear", "absolute"), f"unknown mode {MODE!r}, expected linear|absolute"
DB_PATH = REPO_ROOT / "results" / (
    "linear_oracle_upper_bound.db" if MODE == "linear" else "oracle_upper_bound.db")
OUT_RD = REPO_ROOT / "scripts" / (
    "_oracle_real_rlgr.json" if MODE == "linear" else "_oracle_real_rlgr_absolute.json")
OUT_RECORDS = REPO_ROOT / "scripts" / "_oracle_compaction_records.json"  # linear mode only

BLOCK_SIZES = [16, 8, 4]  # smallest block-count first (16: 3,480 blocks) to validate fast
Q_STEPS = [24, 32, 48, 64]
SENSITIVITY_Q = 32  # representative Q for the per-block sensitivity records (Part 2)
TOTAL_POINTS_EXPECTED = 765821  # longdress frame 1051, full cloud -- cross-checked, not assumed


# ---------------------------------------------------------------------------
# Per-block math -- mirrors linear_oracle_upper_bound_full.py's linear_oracle_worker exactly,
# but evaluates ONE already-known (best_p, best_w) per q_step instead of the 2500-point grid.
# ---------------------------------------------------------------------------

def block_static(Vblock, Ablock, mode=MODE):
    """Everything independent of q_step: plane fit (linear mode only), structural adjacency,
    sink-vector ranking, structural (p=0) basis+coeffs. Byte-for-byte the same operations as
    linear_oracle_worker's / oracle_worker's first half (the two differ only in what signal
    drives the sink vector), computed ONCE per block and reused across all 4 Q values."""
    N = Vblock.shape[0]
    if mode == "linear":
        poly = PolynomialFeatures(degree=1, include_bias=True)
        V_poly = poly.fit_transform(Vblock)
        model = LinearRegression(fit_intercept=False).fit(V_poly, Ablock[:, 0])
        Y_fitted = model.predict(V_poly)
        fit_rmse = float(np.sqrt(np.mean((Ablock[:, 0] - Y_fitted) ** 2)))
        slopes = model.coef_[1:]
        Y_guide = Y_fitted
    else:  # absolute: sink vector driven directly by the real Y channel, no plane fit at all
        fit_rmse = 0.0
        slopes = np.zeros(3)
        Y_guide = Ablock[:, 0]

    D = cdist(Vblock, Vblock)
    W_s = np.zeros_like(D)
    mask = (D > 0) & (D <= np.sqrt(3) + 1e-5)
    W_s[mask] = 1.0 / D[mask]
    W_s += W_s.T

    L_s = np.diag(np.sum(W_s, axis=1)) - W_s
    evals_s, evecs_s = eigh(L_s)
    for i in range(evecs_s.shape[1]):
        if evecs_s[0, i] < 0:
            evecs_s[:, i] *= -1
    coeffs_s = evecs_s.T @ Ablock

    S = compute_sink_vector(W_s, Y_guide)
    sorted_nodes = np.argsort(S)[::-1]

    # Geometric planarity of the POINTS themselves (independent of the luminance signal) --
    # smallest/largest eigenvalue ratio of the point covariance: ~0 for a flat/planar patch,
    # ~1 for an isotropic/volumetric one. Complements fit_rmse (which measures how well the
    # *signal* fits a plane) with how well the *geometry* itself fits one.
    if N >= 3:
        Vc = Vblock - Vblock.mean(axis=0)
        cov = (Vc.T @ Vc) / N
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending: ev[0] >= ev[1] >= ev[2]
        planarity = float(ev[2] / ev[0]) if ev[0] > 1e-12 else 0.0
    else:
        planarity = 0.0

    return dict(N=N, W_s=W_s, coeffs_s=coeffs_s, sorted_nodes=sorted_nodes,
                fit_rmse=fit_rmse, slopes=slopes, y_var=float(np.var(Ablock[:, 0])),
                planarity=planarity)


def coeffs_at(static, Ablock, best_p, best_w):
    """Reproduce the oracle's winning coefficients for one (block, q_step)."""
    N = static["N"]
    if N <= 1 or best_p <= 0.0:
        return static["coeffs_s"]
    n_sl = max(1, int(round(N * best_p)))
    sel = static["sorted_nodes"][:n_sl]
    W_h = static["W_s"].copy()
    W_h[sel, sel] = best_w
    L_h = np.diag(np.sum(W_h, axis=1)) - W_h + np.diag(np.diag(W_h))
    evals_h, evecs_h = eigh(L_h)
    for i in range(evecs_h.shape[1]):
        if evecs_h[0, i] < 0:
            evecs_h[:, i] *= -1
    return evecs_h.T @ Ablock


def _quant_metrics(y, q_step, decider):
    decider._set_vars(q_step)
    y_quant = decider._quantize(y)
    qerror = float(decider._qError(y, y_quant))
    n = min(4, len(y))
    energy_total = float(np.sum(y ** 2))
    compaction = float(np.sum(y[:n] ** 2) / energy_total) if energy_total > 1e-12 else 1.0
    quant_int = np.round(y / q_step).astype(int)
    sparsity = float(np.count_nonzero(quant_int)) / max(len(quant_int), 1)
    return dict(quant=quant_int, qerror=qerror, qerror_sq=qerror ** 2,
                compaction=compaction, sparsity=sparsity)


def process_block(Vblock, Ablock, block_id, rows_by_q, lagrange_proportional,
                   sensitivity_q, scratch_dir):
    """Runs once per block (parallel-dispatched). Returns, per q_step, real bits/compaction for
    BOTH the oracle's chosen config and the structural (p=0) baseline, plus the static
    plane-fit/geometry values (for the sanity check and the sensitivity analysis). At
    `sensitivity_q` only, also RLGR-encodes this block on its own (cold-start, not part of the
    cross-block adaptive stream used for the aggregate rate in Part 1) to get a real per-block
    bpv -- the aggregate figure's rate isn't decomposable per block since RLGR's adaptive state
    carries across the whole concatenated stream, so a per-block sensitivity metric needs its
    own, deliberately-isolated real encode instead."""
    static = block_static(Vblock, Ablock, mode=MODE)
    decider = Decider(mode="0", lagrange_proportional=lagrange_proportional)
    out = {}
    for q_step, row in rows_by_q.items():
        coeffs_oracle = coeffs_at(static, Ablock, row["best_p"], row["best_w"])
        m_oracle = _quant_metrics(coeffs_oracle[:, 0], q_step, decider)
        m_base = _quant_metrics(static["coeffs_s"][:, 0], q_step, decider)

        if q_step == sensitivity_q:
            n_bytes, _ = compress_to_bytes([m_oracle["quant"].reshape(-1, 1)], scratch_dir)
            m_oracle["real_bpv_isolated"] = (n_bytes * 8) / max(static["N"], 1)

        out[q_step] = dict(oracle=m_oracle, base=m_base)
    # Only the small scalar plane-fit/geometry facts travel back to the driver -- W_s/coeffs_s/
    # sorted_nodes are block-local working state, not needed once this block's q_step loop is
    # done, and returning them for every block would keep O(bsize count x N^2) matrices alive
    # simultaneously in the driver process.
    static_light = dict(y_var=static["y_var"], fit_rmse=static["fit_rmse"],
                         slope_mag=float(np.linalg.norm(static["slopes"])),
                         slopes=static["slopes"], planarity=static["planarity"])
    return block_id, static_light, out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_rows_by_block(conn, bsize):
    """block_id -> {q_step: {best_p, best_w, oracle_dist, gain, oracle_rate, point_count}}"""
    rows = conn.execute("""
        SELECT block_id, q_step, best_p, best_w, oracle_dist, gain, oracle_rate, point_count
        FROM results WHERE bsize=?
    """, (bsize,)).fetchall()
    out = defaultdict(dict)
    for block_id, q_step, best_p, best_w, oracle_dist, gain, oracle_rate, point_count in rows:
        out[block_id][q_step] = dict(best_p=best_p, best_w=best_w, oracle_dist=oracle_dist,
                                      gain=gain, oracle_rate=oracle_rate, point_count=point_count)
    return out


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    config_path = REPO_ROOT / "config" / "base_config.yaml"
    params = load_experiment_config(config_path)
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.A = Colourist()._RGBtoYUV(pc.A)
    lp = params.sequential_params.lagrange_proportional
    print(f"Loaded point cloud: {pc.V.shape[0]} points "
          f"(expected {TOTAL_POINTS_EXPECTED}, match={pc.V.shape[0] == TOTAL_POINTS_EXPECTED})")

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    rd_points = []          # Part 1 output rows
    sensitivity_records = []  # Part 2 output rows (SENSITIVITY_Q only)

    for bsize in BLOCK_SIZES:
        t0 = time.time()
        partitioner = MortonBlockPartition()
        _, all_blocks = partitioner.partition(pc, bsize=bsize)
        rows_by_block = load_rows_by_block(conn, bsize)
        print(f"\n--- bsize={bsize}: {len(all_blocks)} blocks, "
              f"{len(rows_by_block)} in DB ---")

        tasks = []
        for b in all_blocks:
            b.init_data(pc.V, pc.A)
            block_id = b.metadata.block_id
            rows_by_q = rows_by_block.get(block_id)
            if rows_by_q is None:
                print(f"  [WARN] {block_id} not found in DB, skipping")
                b.clear_data()
                continue
            tasks.append((b.Vblock.copy(), b.Ablock.copy(), block_id, rows_by_q))
            b.clear_data()

        results = Parallel(n_jobs=-1, backend="loky", batch_size=8)(
            delayed(process_block)(Vb, Ab, bid, rq, lp, SENSITIVITY_Q, str(SCRATCH_DIR))
            for Vb, Ab, bid, rq in tasks
        )

        # Sanity check: recomputed plane slopes vs the stored ones, first 5 blocks (linear mode
        # only -- oracle_upper_bound.db has no slope_x/y/z columns since Absolute Oracle never
        # fits a plane; its cross-check is the oracle_dist comparison further below instead).
        if MODE == "linear":
            print("  Sanity check (recomputed vs stored slope_x/y/z, first 5 blocks):")
            for block_id, static, _ in results[:5]:
                db_row = conn.execute(
                    "SELECT slope_x, slope_y, slope_z FROM results WHERE block_id=? LIMIT 1",
                    (block_id,)).fetchone()
                print(f"    {block_id}: recomputed={np.round(static['slopes'], 5).tolist()} "
                      f"stored={list(db_row)}")

        # Aggregate per q_step: concatenate quantized Y streams in block order, real RLGR encode
        # -- once for the oracle's chosen config, once for the structural (p=0) baseline.
        by_block_ordered = {block_id: out for block_id, _, out in results}
        static_by_block = {block_id: static for block_id, static, _ in results}
        max_qerr_reldiff = 0.0
        n_checked = 0
        for q_step in Q_STEPS:
            streams = {"oracle": [], "base": []}
            sse = {"oracle": 0.0, "base": 0.0}
            total_pts = 0
            for b in all_blocks:
                block_id = b.metadata.block_id
                out = by_block_ordered.get(block_id)
                if out is None or q_step not in out:
                    continue
                d = out[q_step]
                for key in ("oracle", "base"):
                    streams[key].append(d[key]["quant"].reshape(-1, 1))
                    sse[key] += d[key]["qerror_sq"]
                total_pts += rows_by_block[block_id][q_step]["point_count"]
                db_dist = rows_by_block[block_id][q_step]["oracle_dist"]
                if db_dist and db_dist > 1e-9:
                    reldiff = abs(d["oracle"]["qerror"] - db_dist) / db_dist
                    max_qerr_reldiff = max(max_qerr_reldiff, reldiff)
                    n_checked += 1

            point = dict(bsize=bsize, q_step=q_step, total_points=total_pts,
                         n_blocks=len(streams["oracle"]))
            for key in ("oracle", "base"):
                rlgr_n, _ = compress_to_bytes(streams[key], str(SCRATCH_DIR))
                rlgr_bits = rlgr_n * 8
                bpv = rlgr_bits / total_pts
                psnr_y = (-10 * np.log10(sse[key] / (255.0 ** 2 * total_pts))
                          if sse[key] > 0 else float("inf"))
                point[f"{key}_rlgr_bits"] = rlgr_bits
                point[f"{key}_bpv"] = bpv
                point[f"{key}_psnr_y"] = psnr_y
            rd_points.append(point)
            print(f"  Q={q_step:3d}  oracle: bpv={point['oracle_bpv']:.5f} "
                  f"PSNR={point['oracle_psnr_y']:.3f}dB  |  "
                  f"baseline: bpv={point['base_bpv']:.5f} PSNR={point['base_psnr_y']:.3f}dB")

            if MODE == "linear" and q_step == SENSITIVITY_Q:
                for b in all_blocks:
                    block_id = b.metadata.block_id
                    out = by_block_ordered.get(block_id)
                    if out is None or q_step not in out:
                        continue
                    d = out[q_step]["oracle"]
                    static = static_by_block[block_id]
                    row = rows_by_block[block_id][q_step]
                    sensitivity_records.append(dict(
                        block_id=block_id, bsize=bsize,
                        point_count=row["point_count"], y_var=static["y_var"],
                        fit_rmse=static["fit_rmse"], slope_mag=static["slope_mag"],
                        planarity=static["planarity"],
                        best_p=row["best_p"], best_w=row["best_w"], gain=row["gain"],
                        compaction=d["compaction"], sparsity=d["sparsity"],
                        real_bpv=d["real_bpv_isolated"],
                        psnr_y=(-10 * np.log10(d["qerror_sq"] / (255.0 ** 2 * row["point_count"]))
                                if d["qerror_sq"] > 0 else None),
                    ))

        print(f"  Max relative diff, recomputed vs stored oracle_dist "
              f"({n_checked} rows checked, all Q): {max_qerr_reldiff:.2e}")
        print(f"  bsize={bsize} done in {time.time()-t0:.1f}s")

        # Write incrementally after every bsize -- so a later bsize crashing (or running long)
        # doesn't lose the already-validated results, and progress can be inspected mid-run.
        with open(OUT_RD, "w") as f:
            json.dump(rd_points, f, indent=1)
        msg = f"  [+] checkpointed {OUT_RD.name} ({len(rd_points)} pts)"
        if MODE == "linear":
            # Absolute mode never touches this file -- it has no sensitivity records to write,
            # and must not clobber the ones the linear run already produced.
            with open(OUT_RECORDS, "w") as f:
                json.dump(sensitivity_records, f, indent=1)
            msg += f", {OUT_RECORDS.name} ({len(sensitivity_records)} recs)"
        print(msg)

    conn.close()
    print(f"\nDone [{MODE}]. {len(rd_points)} total RD points"
          + (f", {len(sensitivity_records)} sensitivity records." if MODE == "linear" else "."))


if __name__ == "__main__":
    main()
