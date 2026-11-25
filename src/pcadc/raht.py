"""
RAHT (Region-Adaptive Hierarchical Transform) — from-scratch reference implementation.

Reference: R. L. de Queiroz and P. A. Chou, "Compression of 3D Point Clouds Using a
Region-Adaptive Hierarchical Transform," IEEE Trans. Image Process., vol. 25, no. 8, 2016.

LIMITATIONS (read before trusting any RD number derived from this module):
  1. This is BLOCK-LOCAL RAHT: the octree is rooted at each PCADC block's own corner (side
     length = bsize), independently per block -- NOT full-cloud RAHT as in the real G-PCC/
     TMC13 reference software, which shares low-pass context across the entire point cloud's
     octree depth. Expect this to be a lower bound on true RAHT's rate-distortion
     performance, not a faithful reproduction of published TMC13 numbers.
  2. RAHT is a single, deterministic, non-adaptive transform: it takes no graph/Laplacian
     argument and has no candidate/mode to select among. There is nothing here for PCADC's
     gating/Viterbi/codebook-signaling logic to attach to -- by design, not by omission (see
     scripts/raht_compare_report.py for why this does not invalidate the comparison).

This module is deliberately NOT a GFTProcessorStrategy subclass (see transforms.py):
GFTProcessorStrategy.compute(block, graph, Q) requires a graph/Laplacian argument that RAHT
has no use for -- forcing RAHT into that interface would misrepresent its actual data
dependency (occupancy derived from voxel positions only, not a distance-weighted graph).

Mathematics
-----------
Elementary 2-point weighted Haar merge: given two occupied children with (weight, value)
pairs (w1, x1) and (w2, x2), w = w1 + w2:

    DC = (sqrt(w1)*x1 + sqrt(w2)*x2) / sqrt(w)      # low-pass, propagates up, carries weight w
    AC = (sqrt(w2)*x1 - sqrt(w1)*x2) / sqrt(w)       # high-pass, quantized/coded, terminal

This is exactly a 2x2 orthonormal rotation (cos(theta) = sqrt(w1/w), sin(theta) = sqrt(w2/w)),
so T @ T.T == I by construction. A node with only one occupied child at some merge step
passes through unchanged (weight/value untouched, no AC emitted) -- not a plain-Haar
approximation, an exact orthonormal identity on that coordinate.

Per octree level, the 8 candidate children are merged via 3 SEQUENTIAL 1-D passes (z, then y,
then x -- matching Morton bit-interleave order), a separable 3-stage butterfly, not a direct
8-way merge. This telescopes from voxel resolution up to a single root DC across
n_levels = log2(bsize) levels (3*n_levels total elementary merge steps).

Root DC invariant (derived by induction, NOT the plain weighted mean -- an easy, silent,
orthonormality-preserving bug if implemented the "obvious" way):

    DC_root = ( sum_i w_i * x_i ) / sqrt( sum_i w_i )   =  sqrt(W_total) * weighted_mean(x)

i.e. the weighted mean of leaf values, SCALED by sqrt(N) for unit leaf weights -- not the
mean itself. tests/test_raht.py asserts this explicitly and asserts it differs from the naive
`np.average(x, weights=w)`.

Octree without an explicit tree object: block-local voxel positions are sorted by their
full-resolution Morton code (a defensive re-sort -- this module does not trust any upstream
ordering assumption); siblings under a common parent are then automatically contiguous once
grouped by parent coordinate, so the octree is fully represented by per-step index/weight
bookkeeping (RAHTStep) rather than pointer-based tree nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tree bookkeeping (geometry-only -- both encoder and decoder rebuild this
# independently from voxel positions alone, exactly as PCADC's own decoder
# reconstructs its GFT basis from geometry without needing the real signal;
# see decode_block_attributes in scripts/exp_e9_multi_entropy_suite.py).
# ---------------------------------------------------------------------------

@dataclass
class RAHTStep:
    """One elementary axis-merge step. Forward consumes `in`-side node values in original
    input order (0..n_in-1); inverse reconstructs them from the step's output values."""
    axis: int
    n_in: int
    n_out: int
    # Real 2-point merges at this step (each entry consumes 2 inputs, produces 1 output +
    # 1 AC coefficient):
    merge_lo_idx: np.ndarray   # (K,) input index of the "0" sibling
    merge_hi_idx: np.ndarray   # (K,) input index of the "1" sibling
    merge_out_idx: np.ndarray  # (K,) output node index this merge produced
    merge_w_lo: np.ndarray     # (K,) weight of the lo sibling (precomputed, geometry-only)
    merge_w_hi: np.ndarray     # (K,) weight of the hi sibling
    # Pass-throughs at this step (single input -> same output, no AC):
    pass_in_idx: np.ndarray    # (P,)
    pass_out_idx: np.ndarray   # (P,)
    out_weight: np.ndarray     # (n_out,) weight of each output node (feeds the next step)


@dataclass
class RAHTTreeInfo:
    """Deterministic merge structure for one block, derived solely from occupied voxel
    positions. Never serialized or transmitted."""
    bsize: int
    n_levels: int
    n_leaves: int                    # distinct occupied voxel positions
    leaf_weights: np.ndarray         # (n_leaves,) float, duplicate-position counts
    dup_inverse_idx: np.ndarray      # (n_points,) -> leaf index, to re-expand per-point output
    steps: List[RAHTStep]            # forward order: finest level first, root last


def _morton_encode(coords: np.ndarray, bits: int) -> np.ndarray:
    """coords: (N,3) non-negative ints < 2**bits. Interleaves x,y,z bits -> (N,) int64 code."""
    x = coords[:, 0].astype(np.int64)
    y = coords[:, 1].astype(np.int64)
    z = coords[:, 2].astype(np.int64)
    code = np.zeros_like(x)
    for b in range(bits):
        code |= ((x >> b) & 1) << (3 * b)
        code |= ((y >> b) & 1) << (3 * b + 1)
        code |= ((z >> b) & 1) << (3 * b + 2)
    return code


def _morton_decode(code: np.ndarray, bits: int) -> np.ndarray:
    n = code.shape[0]
    coords = np.zeros((n, 3), dtype=np.int64)
    for b in range(bits):
        coords[:, 0] |= ((code >> (3 * b)) & 1) << b
        coords[:, 1] |= ((code >> (3 * b + 1)) & 1) << b
        coords[:, 2] |= ((code >> (3 * b + 2)) & 1) << b
    return coords


def _merge_axis_step(coords: np.ndarray, weight: np.ndarray, axis: int) -> RAHTStep:
    """One elementary axis-merge: groups nodes sharing all coords except `axis`'s low bit."""
    n_in = coords.shape[0]

    if n_in == 1:
        # Trivial pass-through -- keeps step count uniform (3*n_levels always) regardless of
        # how sparse the block is, which simplifies the inverse loop (no special-casing).
        return RAHTStep(
            axis=axis, n_in=1, n_out=1,
            merge_lo_idx=np.array([], dtype=np.int64),
            merge_hi_idx=np.array([], dtype=np.int64),
            merge_out_idx=np.array([], dtype=np.int64),
            merge_w_lo=np.array([]), merge_w_hi=np.array([]),
            pass_in_idx=np.array([0]), pass_out_idx=np.array([0]),
            out_weight=weight.copy(),
        )

    parent_coord = coords.copy()
    parent_coord[:, axis] //= 2
    sibling_bit = coords[:, axis] % 2

    order = np.lexsort((sibling_bit, parent_coord[:, 2], parent_coord[:, 1], parent_coord[:, 0]))
    pc_sorted = parent_coord[order]
    sib_sorted = sibling_bit[order]

    change = np.any(pc_sorted[1:] != pc_sorted[:-1], axis=1)
    group_start = np.concatenate(([0], np.nonzero(change)[0] + 1))
    group_end = np.concatenate((group_start[1:], [len(pc_sorted)]))

    merge_lo, merge_hi, merge_out = [], [], []
    pass_in, pass_out = [], []
    out_weight = []
    for out_idx, (gs, ge) in enumerate(zip(group_start, group_end)):
        members = order[gs:ge]
        if ge - gs == 2:
            bits_here = sib_sorted[gs:ge]
            lo = members[bits_here == 0][0]
            hi = members[bits_here == 1][0]
            merge_lo.append(lo)
            merge_hi.append(hi)
            merge_out.append(out_idx)
            out_weight.append(weight[lo] + weight[hi])
        else:
            # ge - gs == 1: a node with no sibling occupied at this axis -- pass-through.
            only = members[0]
            pass_in.append(only)
            pass_out.append(out_idx)
            out_weight.append(weight[only])

    n_out = len(group_start)
    return RAHTStep(
        axis=axis, n_in=n_in, n_out=n_out,
        merge_lo_idx=np.array(merge_lo, dtype=np.int64),
        merge_hi_idx=np.array(merge_hi, dtype=np.int64),
        merge_out_idx=np.array(merge_out, dtype=np.int64),
        merge_w_lo=weight[np.array(merge_lo, dtype=np.int64)] if merge_lo else np.array([]),
        merge_w_hi=weight[np.array(merge_hi, dtype=np.int64)] if merge_hi else np.array([]),
        pass_in_idx=np.array(pass_in, dtype=np.int64),
        pass_out_idx=np.array(pass_out, dtype=np.int64),
        out_weight=np.array(out_weight, dtype=np.float64),
    )


def build_tree(Vblock: np.ndarray, bsize: int) -> RAHTTreeInfo:
    """
    Vblock: (N,3) voxel positions (absolute; localized internally to the block's own corner).
    bsize: block side length, must be a power of 2 (same convention MortonBlockPartition
    itself enforces).
    """
    assert (bsize & (bsize - 1)) == 0 and bsize > 0, "bsize must be a power of 2"
    n_levels = int(np.log2(bsize))

    Vloc = np.asarray(Vblock, dtype=np.float64)
    Vloc = np.round(Vloc - Vloc.min(axis=0)).astype(np.int64)
    assert Vloc.min() >= 0 and Vloc.max() < bsize, "voxel position outside block bounds"

    codes_full = _morton_encode(Vloc, n_levels)
    order = np.argsort(codes_full, kind="stable")
    codes_sorted = codes_full[order]

    uniq_codes, dup_inverse_sorted, counts = np.unique(
        codes_sorted, return_inverse=True, return_counts=True
    )
    dup_inverse_idx = np.empty(Vblock.shape[0], dtype=np.int64)
    dup_inverse_idx[order] = dup_inverse_sorted

    n_leaves = len(uniq_codes)
    leaf_weights = counts.astype(np.float64)
    coords = _morton_decode(uniq_codes, n_levels)

    steps: List[RAHTStep] = []
    cur_coords, cur_weight = coords, leaf_weights
    for _level in range(n_levels):
        for axis in (2, 1, 0):  # z, y, x -- matches Morton bit-interleave order
            step = _merge_axis_step(cur_coords, cur_weight, axis)
            steps.append(step)
            # Recompute output coords for bookkeeping continuity (structure only; weight
            # already carried in step.out_weight).
            next_coords = np.zeros((step.n_out, 3), dtype=np.int64)
            if step.merge_out_idx.size:
                next_coords[step.merge_out_idx] = cur_coords[step.merge_lo_idx]
                next_coords[step.merge_out_idx, axis] //= 2
            if step.pass_out_idx.size:
                next_coords[step.pass_out_idx] = cur_coords[step.pass_in_idx]
                next_coords[step.pass_out_idx, axis] //= 2
            cur_coords, cur_weight = next_coords, step.out_weight

    assert cur_coords.shape[0] == 1, "RAHT tree did not collapse to a single root"

    return RAHTTreeInfo(
        bsize=bsize, n_levels=n_levels, n_leaves=n_leaves,
        leaf_weights=leaf_weights, dup_inverse_idx=dup_inverse_idx, steps=steps,
    )


# ---------------------------------------------------------------------------
# Forward / inverse transform
# ---------------------------------------------------------------------------

def raht_forward(tree: RAHTTreeInfo, values_per_leaf: np.ndarray) -> np.ndarray:
    """values_per_leaf: (n_leaves,) float, one value per DISTINCT voxel (already
    duplicate-averaged by the caller -- see RAHTProcessor.compute).
    Returns coeffs: (n_leaves,) float, ordered [DC_root, AC_coarsest, ..., AC_finest]."""
    assert values_per_leaf.shape[0] == tree.n_leaves
    cur = values_per_leaf.astype(np.float64)
    per_step_ac: List[np.ndarray] = []

    for step in tree.steps:
        out = np.empty(step.n_out, dtype=np.float64)
        ac = np.zeros(step.merge_out_idx.shape[0], dtype=np.float64)
        if step.merge_out_idx.size:
            x_lo = cur[step.merge_lo_idx]
            x_hi = cur[step.merge_hi_idx]
            w_lo, w_hi = step.merge_w_lo, step.merge_w_hi
            w = w_lo + w_hi
            sw_lo, sw_hi = np.sqrt(w_lo), np.sqrt(w_hi)
            sw = np.sqrt(w)
            out[step.merge_out_idx] = (sw_lo * x_lo + sw_hi * x_hi) / sw
            ac = (sw_hi * x_lo - sw_lo * x_hi) / sw
        if step.pass_out_idx.size:
            out[step.pass_out_idx] = cur[step.pass_in_idx]
        per_step_ac.append(ac)
        cur = out

    assert cur.shape[0] == 1
    dc_root = cur
    ac_coarse_to_fine = np.concatenate(list(reversed(per_step_ac))) if per_step_ac else np.array([])
    return np.concatenate([dc_root, ac_coarse_to_fine])


def raht_inverse(tree: RAHTTreeInfo, coeffs: np.ndarray) -> np.ndarray:
    """Exact structural mirror of raht_forward. coeffs: (n_leaves,) as produced by
    raht_forward (possibly dequantized). Returns values_per_leaf: (n_leaves,)."""
    assert coeffs.shape[0] == tree.n_leaves
    cur = coeffs[:1].copy()  # root DC, shape (1,)
    pos = 1
    # Split the AC tail back into per-step chunks, in the SAME (coarsest-first) order they
    # were concatenated, then walk steps in reverse (root -> leaves).
    ac_chunks = []
    for step in reversed(tree.steps):
        k = step.merge_out_idx.shape[0]
        ac_chunks.append(coeffs[pos:pos + k])
        pos += k
    assert pos == tree.n_leaves

    for step, ac in zip(reversed(tree.steps), ac_chunks):
        out = np.empty(step.n_in, dtype=np.float64)
        if step.merge_out_idx.size:
            dc = cur[step.merge_out_idx]
            w_lo, w_hi = step.merge_w_lo, step.merge_w_hi
            w = w_lo + w_hi
            sw_lo, sw_hi = np.sqrt(w_lo), np.sqrt(w_hi)
            sw = np.sqrt(w)
            x_lo = (sw_lo * dc + sw_hi * ac) / sw
            x_hi = (sw_hi * dc - sw_lo * ac) / sw
            out[step.merge_lo_idx] = x_lo
            out[step.merge_hi_idx] = x_hi
        if step.pass_out_idx.size:
            out[step.pass_in_idx] = cur[step.pass_out_idx]
        cur = out

    assert cur.shape[0] == tree.n_leaves
    return cur


# ---------------------------------------------------------------------------
# Block-facing wrapper
# ---------------------------------------------------------------------------

class RAHTProcessor:
    """Block-local RAHT (de Queiroz & Chou, TIP 2016) for one PCADC block's single attribute
    channel (Y, matching PCADC's own luma-only convention).

    NOT a GFTProcessorStrategy subclass -- see module docstring for why. Mirrors
    ConnectedGFTProcessor's separation of "build structure from geometry" (build_tree, here
    equivalent to constructing the Laplacian) from "compute coefficients from a channel's
    values" (compute/inverse, here equivalent to the GFT projection) only for readability;
    there is no basis matrix materialized in the production path (this is an O(N log N)
    recursive transform, unlike ConnectedGFTProcessor's O(N^3) eigendecomposition of an
    explicit N x N Laplacian -- a real architectural difference worth stating alongside any
    RD numbers, not just a numeric one).
    """

    def __init__(self, bsize: int):
        assert (bsize & (bsize - 1)) == 0 and bsize > 0, "bsize must be a power of 2"
        self.bsize = bsize

    def compute(self, Vblock: np.ndarray, values: np.ndarray,
                materialize_matrix: bool = False
                ) -> Tuple[Optional[np.ndarray], np.ndarray, RAHTTreeInfo]:
        """values: (N,) float, one scalar per point in Vblock (e.g. the Y channel).
        Returns (basis_matrix_or_None, coeffs (n_leaves,), tree). Duplicate-position points
        are pre-averaged onto their shared leaf before the transform, per module docstring
        §1.4 (standard RAHT/voxelization convention).

        materialize_matrix=True builds the explicit n_leaves x n_leaves basis by applying
        raht_forward to each standard basis vector -- O(n_leaves^2), for small-N unit tests
        only, never used in the benchmark encode path."""
        tree = build_tree(Vblock, self.bsize)
        values_per_leaf = np.zeros(tree.n_leaves, dtype=np.float64)
        counts = np.zeros(tree.n_leaves, dtype=np.float64)
        np.add.at(values_per_leaf, tree.dup_inverse_idx, values)
        np.add.at(counts, tree.dup_inverse_idx, 1.0)
        values_per_leaf = values_per_leaf / counts

        coeffs = raht_forward(tree, values_per_leaf)

        basis = None
        if materialize_matrix:
            n = tree.n_leaves
            basis = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                basis[:, i] = raht_forward(tree, e_i)

        return basis, coeffs, tree

    def inverse(self, tree: RAHTTreeInfo, coeffs: np.ndarray) -> np.ndarray:
        """Returns values_per_leaf (n_leaves,); caller re-expands to per-point via
        tree.dup_inverse_idx if needed (duplicate points share their leaf's reconstructed
        value -- consistent with how they were pre-averaged going in)."""
        return raht_inverse(tree, coeffs)
