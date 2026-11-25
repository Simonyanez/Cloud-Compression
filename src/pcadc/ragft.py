"""
RA-GFT (Region-Adaptive Graph Fourier Transform) -- Python port of the reference MATLAB
implementation (matlab_code/Region_Adaptive_GFT.m, iRegion_Adaptive_GFT.m), reusing PCADC's
existing single-level graph/GFT machinery wherever it already implements the same computation,
rather than reimplementing graph construction, the weighted Laplacian, or connectivity
detection from scratch.

Reference: E. Pavez, B. Girault, A. Ortega, P. A. Chou, "Region Adaptive Graph Fourier
Transform for 3D Point Clouds," ICIP 2020. https://arxiv.org/abs/2003.01866

WHAT'S REUSED FROM src/pcadc (unmodified):
  - Block boundary detection: the same formula MortonBlockPartition.partition uses
    (V_coarse = floor(V/bsize)*bsize, diff, nonzero) -- identical to block_indices.m.
  - StructuralGraph._compute_structural_graph: inverse-distance graph, sqrt(3) threshold --
    identical to compute_graph_MSR.m. Used directly (unmodified) for every graph this module
    builds, at every level.
  - Colourist: RGB<->YUV -- identical to RGBtoYUV.m/YUVtoRGB.m (not used directly in this
    module; callers apply it before/after, exactly as PCADC's own scripts do).

  The weighted-Laplacian GFT itself (compute_GFT.m: Qm = Q^-1/2, L_q = Qm@L@Qm) is the SAME
  math as ConnectedGFTProcessor._compute_laplacian/_compute_GFT already implement -- but this
  module does not call that class directly. `_weighted_gft` below reimplements exactly the
  same formula, specialized to the diagonal-Q case this module always uses, because
  ConnectedGFTProcessor._prepare_Q computes Q^-1/2 via scipy.linalg.fractional_matrix_power
  (a general, Schur-decomposition-based routine, correct for an arbitrary matrix but
  needlessly expensive for a diagonal one) -- profiled as the dominant per-block cost once
  logging overhead was removed, unsurprising since this module calls into the weighted GFT far
  more often per frame (every block, every cascade level) than PCADC's own single-level
  pipeline ever does. `_weighted_gft` is verified numerically identical to
  ConnectedGFTProcessor's own output, not just "equivalent in spirit" -- see
  test_single_level_matches_existing_connected_gft_processor.

WHAT'S NEW HERE (genuinely, not just re-derived):
  - The multi-level recursive orchestration: propagating one representative coefficient and
    its accumulated weight up from each block to the next-coarser level, and mirroring that
    exactly for the inverse. PCADC's own single-level pipeline never does this.
  - Disconnected-block handling AT LEVELS ABOVE THE FINEST. PCADC's existing
    DisconnectedGFTProcessor (transforms.py) does not thread a non-identity Q through its
    per-component sub-GFTs or its second-level mean-graph weighting -- it always assumes unit
    weight per point (correct only at level 0, where RA-GFT's own Qin genuinely is ones(N)).
    This module reimplements that orchestration using the same underlying primitives
    (ConnectedGFTProcessor for each per-component sub-GFT and for the second-level mean-graph
    GFT, StructuralGraph with distance_threshold=inf for the complete graph over component
    centroids -- matching complete_graph.m exactly) so it stays correct with propagated,
    non-unit weights at every level.

WEIGHT PROPAGATION (derived, not assumed -- verified against both Region_Adaptive_GFT.m and
iRegion_Adaptive_GFT.m, which independently recompute Q via compute_Q_lower_resolution rather
than reusing the forward pass's own Qout, confirming Q is purely structurally derivable):
the weight of the ONE coefficient that propagates from any block to the next-coarser level is
always exactly sum(Q) over that block's points, regardless of whether the block's graph is
connected or splits into multiple components. This lets the full multi-level Q schedule be
precomputed once, from geometry alone, before any transform math runs (_multilevel_structure).

BLOCK-RELATIVE COEFFICIENT ORDERING (mirrors the MATLAB reference exactly, including its mild
arbitrariness in the disconnected case -- documented rather than "fixed", since forward and
inverse must agree on the identical convention):
  - Connected block: position 0 of the block's returned coefficients is the lowest graph
    frequency (ConnectedGFTProcessor already sorts ascending) -- the natural DC, propagated up.
  - Disconnected block (>1 connected component): components are processed in
    scipy.sparse.csgraph.connected_components' label order (0, 1, 2, ...); each component's own
    DC is corrected via a second-level GFT over a complete graph of per-component weighted
    centroids; corrected DCs are placed first (component 0's, then component 1's, ...),
    followed by every component's non-DC coefficients. Position 0 (component 0's corrected DC
    -- an arbitrary choice, the same one the MATLAB reference makes) is the one propagated
    upward; every other coefficient, including OTHER components' corrected DCs, is terminal at
    this level. This matches the reference exactly, including this specific limitation: a block
    that splits into >1 component only carries ONE component's information further up the
    hierarchy.

DELIBERATE SIMPLIFICATION vs. the MATLAB reference: graph frequencies (Gfreq/freqs) are
computed internally but not returned -- neither test_ragft_forward_inverse.m nor
encoder_RAGFT_RLGR_large_blocks.m actually consume them downstream (diagnostic-only in the
reference), so tracking them here would add real bookkeeping for zero functional benefit to
forward/inverse correctness, quantization, or entropy coding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .blocks import Block, BlockMetadata
from .graph import StructuralGraph

# PCADC's own modules (graph.py, transforms.py, blocks.py) log at DEBUG level on every call
# (basicConfig(level=logging.DEBUG, ...) at import time). RA-GFT's per-level, per-block
# recursion calls into them orders of magnitude more often than PCADC's own single-level
# pipeline does, so unlike there, the logging overhead here is measurable (profiled: ~30% of
# per-block runtime went to logging calls that are filtered out downstream anyway). Same fix
# scripts/exp_e10_multisequence_benchmark.py's own worker functions already use
# (logging.disable(logging.CRITICAL)) -- not a new pattern, just applied at import time here
# since this module IS the hot path, not a per-call worker.
logging.disable(logging.CRITICAL)


def _weighted_gft(Adj: np.ndarray, Qvec: np.ndarray) -> np.ndarray:
    """Fast path for ConnectedGFTProcessor's own math (transforms.py:_compute_laplacian /
    _compute_GFT), specialized to the case this module always uses: Q is diagonal. Verified
    numerically identical to ConnectedGFTProcessor's output (see
    test_single_level_matches_existing_connected_gft_processor).

    ConnectedGFTProcessor._prepare_Q calls scipy.linalg.fractional_matrix_power(Q, -0.5) on
    the full Q MATRIX via a general Schur-decomposition-based algorithm -- correct for an
    arbitrary matrix Q, but this module only ever passes a diagonal one, for which the same
    result is just an elementwise power of the diagonal. Profiled: fractional_matrix_power was
    the single largest cost in the per-block transform once per-call logging overhead was
    removed (~40% of remaining runtime) -- unsurprising given this module calls into the
    weighted GFT far more often per frame (every block, every cascade level) than PCADC's own
    single-level pipeline ever does. No self-loop term (C=0): this module never wraps an
    AttributeGraph, only ever a plain StructuralGraph, so ConnectedGFTProcessor's
    `isinstance(graph, AttributeGraph)` branch is always False for every caller here.
    """
    D = np.diag(Adj.sum(axis=0))
    L = D - Adj
    Qm = Qvec ** -0.5
    L_q = (Qm[:, None] * L) * Qm[None, :]

    eigvals, eigvecs = np.linalg.eigh(L_q)
    idx = np.argsort(eigvals)
    GFT_mat = eigvecs[:, idx]
    # Same per-row sign convention as ConnectedGFTProcessor._compute_GFT: for orthonormal
    # eigvecs, flipping a row's sign across all columns is left-multiplication by a diagonal
    # +-1 matrix, which preserves both GFT_mat.T @ GFT_mat = I and GFT_mat @ GFT_mat.T = I.
    flip = GFT_mat[:, 0] < 0
    GFT_mat[flip, :] *= -1
    return GFT_mat


# ---------------------------------------------------------------------------
# Level structure (geometry/weight-only, shared by forward and inverse -- exactly mirrors
# iRegion_Adaptive_GFT.m independently recomputing V_MR/Q/starti/endi from V and params,
# rather than trusting anything saved by the forward pass).
# ---------------------------------------------------------------------------

@dataclass
class RAGFTLevel:
    bsize: int
    start_indices: np.ndarray
    end_indices: np.ndarray
    Vcurr: np.ndarray   # (n_level, 3) int, this level's point positions
    Qin: np.ndarray      # (n_level,) float, weight of each point BEFORE this level's transform


def _block_boundaries(Vcurr: np.ndarray, bsize: int) -> np.ndarray:
    """Identical formula to MortonBlockPartition.partition's boundary detection /
    block_indices.m."""
    V_coarse = np.floor(Vcurr / bsize) * bsize
    variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)
    variation = np.concatenate(([1], variation))
    return np.nonzero(variation)[0]


def _multilevel_structure(V: np.ndarray, bsize_per_level: List[int]) -> List[RAGFTLevel]:
    """bsize_per_level[0] is applied first (finest, on full-resolution V); bsize_per_level[-1]
    is applied last (coarsest). Each entry must be a power of 2."""
    for b in bsize_per_level:
        assert (b & (b - 1)) == 0 and b > 0, "each block size must be a power of 2"

    levels: List[RAGFTLevel] = []
    Vcurr = np.floor(np.asarray(V, dtype=np.float64)).astype(np.int64)
    Qcurr = np.ones(Vcurr.shape[0], dtype=np.float64)

    for bsize in bsize_per_level:
        start_idx = _block_boundaries(Vcurr, bsize)
        n = Vcurr.shape[0]
        end_idx = np.concatenate([start_idx[1:] - 1, [n - 1]])
        levels.append(RAGFTLevel(bsize=bsize, start_indices=start_idx, end_indices=end_idx,
                                  Vcurr=Vcurr, Qin=Qcurr))

        Qnext = np.array([Qcurr[s:e + 1].sum() for s, e in zip(start_idx, end_idx)])
        Vnext = np.floor(Vcurr[start_idx] / bsize).astype(np.int64)
        Vcurr, Qcurr = Vnext, Qnext

    return levels


# ---------------------------------------------------------------------------
# Single-block transform (connected + disconnected), reusing ConnectedGFTProcessor and
# StructuralGraph as the underlying primitives for both cases.
# ---------------------------------------------------------------------------

def _make_block(Vblock: np.ndarray, Ablock: np.ndarray, bsize: int) -> Block:
    n = Vblock.shape[0]
    meta = BlockMetadata(start=0, end=n - 1, source="ragft", frame=0, block_size=bsize, block_idx=0)
    block = Block(meta)
    block.set_data(Vblock, Ablock)
    return block


def _connected_block_transform(Vblock: np.ndarray, Ablock: np.ndarray, Qblock: np.ndarray,
                                bsize: int, graph: Optional[StructuralGraph] = None
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (coeffs (n,C), weights (n,), GFT_mat (n,n)). Position 0 = DC (lowest graph
    frequency; ConnectedGFTProcessor already sorts ascending). Matches
    RAGFT_connected_graph: every coefficient gets weight = sum(Q) over the whole block.
    `graph` lets a caller that already built the StructuralGraph (e.g. for a connectivity
    check) pass it in rather than have it rebuilt (the distance-matrix construction is not
    free when called this many times across a multi-level cascade)."""
    if graph is None:
        graph = StructuralGraph(_make_block(Vblock, Ablock, bsize).metadata)
        graph.set_data(V=Vblock)
    GFT_mat = _weighted_gft(graph.weights, Qblock)
    coeffs = GFT_mat.T @ Ablock
    weights = np.full(Vblock.shape[0], Qblock.sum(), dtype=np.float64)
    return coeffs, weights, GFT_mat


def _disconnected_block_transform(Vblock: np.ndarray, Ablock: np.ndarray, Qblock: np.ndarray,
                                   bsize: int, labels: np.ndarray, num_components: int
                                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Own orchestration (not DisconnectedGFTProcessor -- see module docstring for why):
    threads the real, propagated Qblock through both the per-component sub-GFTs and the
    second-level mean-graph GFT. Matches RAGFT_disconnected_graph exactly: corrected DCs
    (one per component, component-label order) come first, then every component's own
    non-DC coefficients; every corrected DC gets weight = sum(Q) over the WHOLE block, every
    non-DC coefficient gets weight = sum(Q) over just its own component."""
    n = Vblock.shape[0]
    U_parts = []
    isDC = np.zeros(n, dtype=bool)
    Vmean = np.zeros((num_components, 3), dtype=np.float64)
    Qcomp_sum = np.zeros(num_components, dtype=np.float64)
    sub_gft_mats = []
    pos = 0
    comp_order = []  # sub_idxs per component, in processing order (needed for inverse)

    for comp_id in range(num_components):
        sub_idxs = np.where(labels == comp_id)[0]
        comp_order.append(sub_idxs)
        Vsub, Asub, Qsub = Vblock[sub_idxs], Ablock[sub_idxs], Qblock[sub_idxs]

        if Vsub.shape[0] == 1:
            GFT_sub = np.array([[1.0]])
        else:
            sub_graph = StructuralGraph(_make_block(Vsub, Asub, bsize).metadata)
            sub_graph.set_data(V=Vsub)
            GFT_sub = _weighted_gft(sub_graph.weights, Qsub)
        sub_gft_mats.append(GFT_sub)

        Utmp = np.zeros((n, len(sub_idxs)))
        Utmp[sub_idxs, :] = GFT_sub
        U_parts.append(Utmp)

        isDC[pos] = True
        pos += len(sub_idxs)

        Vmean[comp_id, :] = np.average(Vsub, axis=0, weights=Qsub)
        Qcomp_sum[comp_id] = Qsub.sum()

    U = np.concatenate(U_parts, axis=1)
    Coeffs = U.T @ Ablock

    isDC_idx = np.where(isDC)[0]
    notDC_idx = np.where(~isDC)[0]
    Coeffs_low = Coeffs[isDC_idx, :]
    Coeffs_high = Coeffs[notDC_idx, :]

    mean_graph = StructuralGraph(_make_block(Vmean, Vmean, bsize).metadata)
    mean_graph.metadata.distance_threshold = np.inf  # complete graph, matches complete_graph.m
    mean_graph.set_data(V=Vmean)
    GFT_mean = _weighted_gft(mean_graph.weights, Qcomp_sum)

    Coeffs_low_fixed = GFT_mean.T @ Coeffs_low
    Coeffs_out = np.concatenate([Coeffs_low_fixed, Coeffs_high], axis=0)

    total_w = Qblock.sum()
    weights_out = np.concatenate([
        np.full(num_components, total_w, dtype=np.float64),
        np.concatenate([np.full(len(comp_order[c]) - 1, Qcomp_sum[c], dtype=np.float64)
                         for c in range(num_components)]) if n > num_components else np.array([]),
    ])

    inv_ctx = dict(U=U, isDC_idx=isDC_idx, notDC_idx=notDC_idx, GFT_mean=GFT_mean,
                   num_components=num_components)
    return Coeffs_out, weights_out, inv_ctx


def _transform_one_block(Vblock: np.ndarray, Ablock: np.ndarray, Qblock: np.ndarray, bsize: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    n = Vblock.shape[0]
    if n == 1:
        return Ablock.copy(), Qblock.copy()

    sgraph_check = StructuralGraph(_make_block(Vblock, Ablock, bsize).metadata)
    sgraph_check.set_data(V=Vblock)
    num_components, labels = connected_components(csr_matrix(sgraph_check.weights),
                                                    directed=False, return_labels=True)
    if num_components == 1:
        coeffs, weights, _ = _connected_block_transform(Vblock, Ablock, Qblock, bsize,
                                                          graph=sgraph_check)
    else:
        coeffs, weights, _ = _disconnected_block_transform(Vblock, Ablock, Qblock, bsize,
                                                             labels, num_components)
    return coeffs, weights


def _inverse_one_block(Vblock: np.ndarray, Coeffs: np.ndarray, Qblock: np.ndarray, bsize: int
                        ) -> np.ndarray:
    """Exact structural mirror of _transform_one_block, rebuilding the same graph(s) from
    geometry+weights alone (never trusts anything saved by the forward pass)."""
    n = Vblock.shape[0]
    if n == 1:
        return Coeffs.copy()

    sgraph_check = StructuralGraph(_make_block(Vblock, Coeffs, bsize).metadata)
    sgraph_check.set_data(V=Vblock)
    num_components, labels = connected_components(csr_matrix(sgraph_check.weights),
                                                    directed=False, return_labels=True)

    if num_components == 1:
        _, _, GFT_mat = _connected_block_transform(Vblock, np.zeros_like(Coeffs), Qblock, bsize,
                                                     graph=sgraph_check)
        return GFT_mat @ Coeffs

    # disconnected: rebuild U, isDC, GFT_mean exactly as forward did
    Vmean = np.zeros((num_components, 3), dtype=np.float64)
    Qcomp_sum = np.zeros(num_components, dtype=np.float64)
    U_parts = []
    isDC = np.zeros(n, dtype=bool)
    pos = 0
    for comp_id in range(num_components):
        sub_idxs = np.where(labels == comp_id)[0]
        Vsub, Qsub = Vblock[sub_idxs], Qblock[sub_idxs]
        if Vsub.shape[0] == 1:
            GFT_sub = np.array([[1.0]])
        else:
            sub_graph = StructuralGraph(_make_block(Vsub, Vsub, bsize).metadata)
            sub_graph.set_data(V=Vsub)
            GFT_sub = _weighted_gft(sub_graph.weights, Qsub)
        Utmp = np.zeros((n, len(sub_idxs)))
        Utmp[sub_idxs, :] = GFT_sub
        U_parts.append(Utmp)
        isDC[pos] = True
        pos += len(sub_idxs)
        Vmean[comp_id, :] = np.average(Vsub, axis=0, weights=Qsub)
        Qcomp_sum[comp_id] = Qsub.sum()
    U = np.concatenate(U_parts, axis=1)

    mean_graph = StructuralGraph(_make_block(Vmean, Vmean, bsize).metadata)
    mean_graph.metadata.distance_threshold = np.inf
    mean_graph.set_data(V=Vmean)
    GFT_mean = _weighted_gft(mean_graph.weights, Qcomp_sum)

    isDC_idx = np.where(isDC)[0]
    notDC_idx = np.where(~isDC)[0]
    numDCs = num_components

    A1 = GFT_mean @ Coeffs[:numDCs, :]  # invert 2nd-level GFT
    A1_permuted = np.zeros((n, Coeffs.shape[1]))
    A1_permuted[isDC_idx, :] = A1
    A1_permuted[notDC_idx, :] = Coeffs[numDCs:, :]

    return U @ A1_permuted


# ---------------------------------------------------------------------------
# Public forward / inverse
# ---------------------------------------------------------------------------

def region_adaptive_gft(V: np.ndarray, A: np.ndarray, bsize_per_level: List[int]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    V: (N,3) voxel positions (Morton-ordered, as MortonBlockPartition itself assumes).
    A: (N,C) attributes (e.g. YUV; caller applies Colourist beforehand, matching PCADC's own
       convention).
    bsize_per_level: block sizes applied in order, index 0 FIRST (finest, on full-resolution V)
       through index -1 LAST (coarsest).

    Returns (Ahat, weights): Ahat is (N,C), ordered [root DC(s), coarsest-level high-pass, ...,
    finest-level high-pass] (matches the MATLAB reference's own ordering exactly). weights is
    (N,) matching Ahat row-for-row.
    """
    levels = _multilevel_structure(V, bsize_per_level)
    Acurr = np.asarray(A, dtype=np.float64).copy()
    Qcurr = np.ones(Acurr.shape[0], dtype=np.float64)
    high_pass_coeffs: List[np.ndarray] = []
    high_pass_weights: List[np.ndarray] = []

    for lvl in levels:
        n = lvl.Vcurr.shape[0]
        Acurr_hat = Acurr.copy()
        Qout = lvl.Qin.copy()
        for s, e in zip(lvl.start_indices, lvl.end_indices):
            if e == s:
                continue
            coeffs, weights = _transform_one_block(lvl.Vcurr[s:e + 1], Acurr[s:e + 1],
                                                     lvl.Qin[s:e + 1], lvl.bsize)
            Acurr_hat[s:e + 1] = coeffs
            Qout[s:e + 1] = weights

        mask = np.ones(n, dtype=bool)
        mask[lvl.start_indices] = False
        high_pass_coeffs.insert(0, Acurr_hat[mask])
        high_pass_weights.insert(0, Qout[mask])

        Acurr = Acurr_hat[lvl.start_indices]
        Qcurr = Qout[lvl.start_indices]

    Ahat = np.concatenate([Acurr] + high_pass_coeffs, axis=0)
    weights = np.concatenate([Qcurr] + high_pass_weights)
    return Ahat, weights


def inverse_region_adaptive_gft(V: np.ndarray, Coeff: np.ndarray, bsize_per_level: List[int]
                                 ) -> np.ndarray:
    """Exact structural mirror of region_adaptive_gft. V and bsize_per_level must match the
    forward call exactly; Coeff is (N,C) as produced by region_adaptive_gft (possibly
    dequantized). Returns the reconstructed (N,C) attribute array, in ORIGINAL point order."""
    levels = _multilevel_structure(V, bsize_per_level)
    # number of root elements = number of blocks at the coarsest level
    n_root = len(levels[-1].start_indices)

    pos = n_root
    high_pass_chunks = []
    for lvl in reversed(levels):
        n = lvl.Vcurr.shape[0]
        n_high = n - len(lvl.start_indices)
        high_pass_chunks.append(Coeff[pos:pos + n_high, :])
        pos += n_high
    assert pos == Coeff.shape[0]

    Acurr = Coeff[:n_root, :].copy()

    for lvl, high_chunk in zip(reversed(levels), high_pass_chunks):
        n = lvl.Vcurr.shape[0]
        C = Coeff.shape[1]
        coeff_level = np.zeros((n, C))
        coeff_level[lvl.start_indices, :] = Acurr
        mask = np.ones(n, dtype=bool)
        mask[lvl.start_indices] = False
        coeff_level[mask, :] = high_chunk

        out = coeff_level.copy()
        for s, e in zip(lvl.start_indices, lvl.end_indices):
            if e == s:
                continue
            rec = _inverse_one_block(lvl.Vcurr[s:e + 1], coeff_level[s:e + 1],
                                      lvl.Qin[s:e + 1], lvl.bsize)
            out[s:e + 1] = rec
        Acurr = out

    return Acurr
