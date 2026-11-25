# Test file for the RA-GFT port (src/pcadc/ragft.py)
#
# Correctness gate for a from-scratch multi-level orchestration built on top of PCADC's
# existing (already-tested) single-level graph/GFT machinery. See the module docstring in
# src/pcadc/ragft.py for exactly what's reused vs. genuinely new, and for the weight-
# propagation and coefficient-ordering conventions being verified here.
import numpy as np
import pytest

from pcadc.ragft import (
    region_adaptive_gft, inverse_region_adaptive_gft, _multilevel_structure,
    _transform_one_block, _connected_block_transform,
)
from pcadc.blocks import Block, BlockMetadata
from pcadc.graph import StructuralGraph
from pcadc.transforms import ConnectedGFTProcessor


def _grid_points(bsize):
    return np.array(np.meshgrid(*[range(bsize)] * 3)).T.reshape(-1, 3).astype(np.float64)


def _sorted_by_morton_ish(V):
    """Sort so block-boundary detection (which assumes Morton/contiguous ordering) behaves --
    a simple lexsort is sufficient for the small synthetic blocks used here (any ordering where
    same-coarse-cell points are contiguous works)."""
    order = np.lexsort((V[:, 2], V[:, 1], V[:, 0]))
    return V[order]


# ---------------------------------------------------------------------------
# 1. Single-level cross-check against PCADC's own existing single-level pipeline
# ---------------------------------------------------------------------------

def test_single_level_matches_existing_connected_gft_processor():
    """With bsize_per_level = [bsize] (one level only, whole point set as one block), the new
    orchestration must reduce to exactly what PCADC's own ConnectedGFTProcessor already does
    for that same block, with unit weights -- the trivial case where nothing new should change
    anything."""
    rng = np.random.default_rng(0)
    bsize = 8
    V = _sorted_by_morton_ish(_grid_points(bsize))
    n = V.shape[0]
    A = rng.normal(size=(n, 1))

    Ahat, weights = region_adaptive_gft(V, A, bsize_per_level=[bsize])

    # Direct call to the existing single-level machinery for comparison.
    meta = BlockMetadata(start=0, end=n - 1, source="x", frame=0, block_size=bsize, block_idx=0)
    block = Block(meta)
    block.set_data(V, A)
    graph = StructuralGraph(meta)
    graph.set_data(V=V)
    GFT_mat, coeffs_direct = ConnectedGFTProcessor().compute(block, graph)  # Q=None -> identity

    # region_adaptive_gft's single-level output is [DC, high-pass...] i.e. GFT-sorted-ascending
    # coefficients in a possibly reordered layout (DC moved to front is already the natural
    # order for a connected single block) -- for a single connected block spanning the whole
    # point set, this must be an exact match.
    np.testing.assert_allclose(Ahat, coeffs_direct, atol=1e-8)
    np.testing.assert_allclose(weights, np.full(n, float(n)), atol=1e-8)


# ---------------------------------------------------------------------------
# 2. Per-block orthonormality (connected case), across random occupancy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_connected_block_transform_orthonormal(seed):
    rng = np.random.default_rng(seed)
    bsize = 8
    grid = _grid_points(bsize)
    n = rng.integers(4, 40)
    idx = rng.choice(grid.shape[0], size=n, replace=False)
    V = _sorted_by_morton_ish(grid[idx])
    Q = rng.uniform(0.5, 3.0, size=n)

    coeffs, weights, GFT_mat = _connected_block_transform(
        V, rng.normal(size=(n, 1)), Q, bsize)

    np.testing.assert_allclose(GFT_mat @ GFT_mat.T, np.eye(n), atol=1e-7)
    np.testing.assert_allclose(GFT_mat.T @ GFT_mat, np.eye(n), atol=1e-7)
    np.testing.assert_allclose(weights, np.full(n, Q.sum()), atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Perfect reconstruction, single level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_single_level_perfect_reconstruction(seed):
    rng = np.random.default_rng(seed)
    bsize = 8
    grid = _grid_points(bsize)
    n = rng.integers(20, 80)
    idx = rng.choice(grid.shape[0], size=n, replace=False)
    V = _sorted_by_morton_ish(grid[idx])
    A = rng.normal(size=(n, 2))

    Ahat, _ = region_adaptive_gft(V, A, bsize_per_level=[bsize])
    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=[bsize])

    np.testing.assert_allclose(A_rec, A, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Perfect reconstruction, multiple levels (the genuinely new part)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [20, 21, 22, 23])
def test_multilevel_perfect_reconstruction(seed):
    rng = np.random.default_rng(seed)
    bsize_finest = 16
    grid = _grid_points(bsize_finest)
    n = rng.integers(80, 400)
    idx = rng.choice(grid.shape[0], size=n, replace=False)
    V = _sorted_by_morton_ish(grid[idx])
    A = rng.normal(size=(n, 3))

    bsize_per_level = [2, 2, 2, 2]  # 4 levels, 2x2x2 each, matches the reference test script's
                                     # own convention (test_ragft_forward_inverse.m uses ten
                                     # levels of bsize=2 for a depth-10 cloud; 4 levels here
                                     # covers bsize_finest=16's full depth: 2^4=16)
    Ahat, weights = region_adaptive_gft(V, A, bsize_per_level=bsize_per_level)

    assert Ahat.shape == A.shape
    assert weights.shape == (n,)
    # Parseval / energy conservation across the whole multi-level transform (orthonormal
    # composition of orthonormal per-block/per-level operators must conserve energy exactly).
    np.testing.assert_allclose(np.sum(Ahat ** 2), np.sum(A ** 2), atol=1e-6)

    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=bsize_per_level)
    np.testing.assert_allclose(A_rec, A, atol=1e-6)


def test_multilevel_reconstruction_on_dense_full_cube():
    """Fully-occupied 16^3 cube -- every level's every block is maximally populated and always
    connected, the least adversarial but highest-volume case."""
    bsize_finest = 16
    V = _sorted_by_morton_ish(_grid_points(bsize_finest))
    rng = np.random.default_rng(42)
    A = rng.normal(size=(V.shape[0], 1))

    bsize_per_level = [2, 2, 2, 2]
    Ahat, _ = region_adaptive_gft(V, A, bsize_per_level=bsize_per_level)
    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=bsize_per_level)
    np.testing.assert_allclose(A_rec, A, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Disconnected-component block, forced explicitly, single level and propagated through a
#    second (coarser) level -- the part with no direct PCADC precedent (see module docstring).
# ---------------------------------------------------------------------------

def test_disconnected_block_single_level_reconstruction():
    """Two clusters of points far enough apart (> sqrt(3)) to form 2 connected components
    within one block, forcing the disconnected-block code path."""
    cluster_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    cluster_b = cluster_a + np.array([10.0, 10.0, 10.0])  # far apart -> disconnected
    V = np.concatenate([cluster_a, cluster_b], axis=0)
    rng = np.random.default_rng(5)
    A = rng.normal(size=(V.shape[0], 2))
    bsize = 16  # large enough that both clusters land in the same block

    Ahat, weights = region_adaptive_gft(V, A, bsize_per_level=[bsize])
    np.testing.assert_allclose(np.sum(Ahat ** 2), np.sum(A ** 2), atol=1e-6)

    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=[bsize])
    np.testing.assert_allclose(A_rec, A, atol=1e-6)

    # weight check: the propagated (position-0) coefficient must carry the WHOLE block's total
    # weight (8 points, unit weight each at level 0) -- matches the module docstring's claim
    # that propagated weight is always sum(Q) over the whole block regardless of connectivity.
    assert weights[0] == pytest.approx(8.0)


def test_disconnected_block_propagates_through_coarser_level():
    """The disconnected block from the test above, now as the FINEST of two levels, so its
    propagated (weighted, non-unit-Q) representative must be correctly consumed by a second,
    coarser level -- exercising exactly the gap PCADC's own DisconnectedGFTProcessor has
    (unit-weight-only) that this module's own orchestration fixes."""
    cluster_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    cluster_b = cluster_a + np.array([10.0, 10.0, 10.0])
    # A second, ordinary connected block elsewhere, far from both clusters, so the coarsest
    # level still has >1 point to form a real (connected) graph over propagated representatives.
    cluster_c = np.array([[20, 0, 0], [21, 0, 0]], dtype=np.float64)
    V = _sorted_by_morton_ish(np.concatenate([cluster_a, cluster_b, cluster_c], axis=0))
    rng = np.random.default_rng(6)
    A = rng.normal(size=(V.shape[0], 1))

    bsize_per_level = [16, 32]
    Ahat, _ = region_adaptive_gft(V, A, bsize_per_level=bsize_per_level)
    np.testing.assert_allclose(np.sum(Ahat ** 2), np.sum(A ** 2), atol=1e-6)

    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=bsize_per_level)
    np.testing.assert_allclose(A_rec, A, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Degenerate single-point block
# ---------------------------------------------------------------------------

def test_single_point_block():
    V = np.array([[5.0, 5.0, 5.0]])
    A = np.array([[42.0]])
    Ahat, weights = region_adaptive_gft(V, A, bsize_per_level=[16])
    assert Ahat.shape == (1, 1)
    assert Ahat[0, 0] == pytest.approx(42.0)
    assert weights[0] == pytest.approx(1.0)

    A_rec = inverse_region_adaptive_gft(V, Ahat, bsize_per_level=[16])
    assert A_rec[0, 0] == pytest.approx(42.0)
