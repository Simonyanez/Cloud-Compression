# Test file for the RAHT module (src/pcadc/raht.py)
#
# These are the correctness gate for a from-scratch transform implementation: every test here
# must pass before scripts/exp_raht_baseline.py is trusted with real sequence data. See the
# module docstring in src/pcadc/raht.py for the math being verified.
import numpy as np
import pytest

from pcadc.raht import build_tree, raht_forward, raht_inverse, RAHTProcessor


def _values_per_leaf(tree, values):
    """Mirrors RAHTProcessor.compute()'s internal duplicate-averaging remap: original-point-
    order `values` -> leaf-order (Morton-sorted, deduplicated) values. Needed whenever a test
    wants to compare `coeffs` against a materialized basis matrix T directly (T's columns are
    in leaf order, not necessarily the caller's original point order)."""
    n_leaves = tree.n_leaves
    out = np.zeros(n_leaves, dtype=np.float64)
    counts = np.zeros(n_leaves, dtype=np.float64)
    np.add.at(out, tree.dup_inverse_idx, values)
    np.add.at(counts, tree.dup_inverse_idx, 1.0)
    return out / counts


# ---------------------------------------------------------------------------
# 1. Hand-computed 2-point merge matches the formula exactly
# ---------------------------------------------------------------------------

def test_raht_two_point_merge_matches_formula():
    # w1=1, w2=3 realized as actual occupancy: one point at z=0, three duplicate points at
    # z=1 (duplicate positions get pre-averaged and weight=count, per module docstring §1.4).
    w1, w2, x1, x2 = 1.0, 3.0, 10.0, -4.0
    w = w1 + w2
    expected_dc = (np.sqrt(w1) * x1 + np.sqrt(w2) * x2) / np.sqrt(w)
    expected_ac = (np.sqrt(w2) * x1 - np.sqrt(w1) * x2) / np.sqrt(w)

    Vblock = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    values = np.array([x1, x2, x2, x2])
    proc = RAHTProcessor(bsize=2)
    _, coeffs, tree = proc.compute(Vblock, values)

    assert tree.n_leaves == 2
    np.testing.assert_allclose(tree.leaf_weights.sum(), w)
    assert coeffs.shape == (2,)
    assert coeffs[0] == pytest.approx(expected_dc, abs=1e-9)
    assert abs(coeffs[1]) == pytest.approx(abs(expected_ac), abs=1e-9)

    # The 2x2 transform matrix (acting on leaf-ordered, i.e. post-dedup, values) must be
    # orthonormal regardless of the weights used to build it.
    T, _, tree2 = proc.compute(Vblock, values, materialize_matrix=True)
    assert T is not None
    np.testing.assert_allclose(T @ T.T, np.eye(2), atol=1e-9)
    np.testing.assert_allclose(T.T @ T, np.eye(2), atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Basis orthonormality on a synthetic, irregularly-occupied block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_raht_basis_orthonormal(seed):
    rng = np.random.default_rng(seed)
    bsize = 8
    n_candidates = bsize ** 3
    n = rng.integers(2, 38)
    all_coords = np.array(np.meshgrid(*[range(bsize)] * 3)).T.reshape(-1, 3)
    idx = rng.choice(n_candidates, size=n, replace=False)
    Vblock = all_coords[idx].astype(np.float64)
    values = rng.normal(size=n)

    proc = RAHTProcessor(bsize=bsize)
    T, coeffs, tree = proc.compute(Vblock, values, materialize_matrix=True)

    assert T.shape == (n, n)
    np.testing.assert_allclose(T @ T.T, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(T.T @ T, np.eye(n), atol=1e-8)
    # coeffs from the non-matrix path must match T @ values_per_leaf exactly (same underlying
    # raht_forward either way; T's columns are in leaf/Morton-sorted order, which need not
    # match the caller's original Vblock order, hence the explicit remap before comparing).
    np.testing.assert_allclose(coeffs, T @ _values_per_leaf(tree, values), atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Independent cross-check on a fully-occupied unit-weight 2x2x2 cube.
#
# NOTE ON DESIGN: RAHT's defining property is that the low-pass (DC) term alone propagates
# to the next coarser level while each high-pass (AC) term is extracted and terminated
# immediately -- unlike a flat (non-hierarchical) Hadamard transform, where every stage's
# both outputs keep participating in further butterflies. An earlier draft of this test
# wrongly assumed RAHT on a fully-occupied unit-weight block reduces to the flat 8-point
# Hadamard transform (uniform |T_ij| = 1/sqrt(8) everywhere); verified numerically that this
# is false -- e.g. the finest-level AC rows only ever combine 2 of the 8 leaves and so have
# magnitude 1/sqrt(2), not 1/sqrt(8). That non-uniformity is exactly what makes RAHT a
# multiresolution transform, not a flat one, and is not a bug. What DOES hold, and is
# Hadamard-derived, is the DC (root) row: it is the one row where every leaf still
# contributes, with the standard Hadamard DC magnitude.
# ---------------------------------------------------------------------------

def test_raht_full_block_root_dc_row_is_hadamard_dc():
    bsize = 2
    Vblock = np.array(np.meshgrid(*[range(bsize)] * 3)).T.reshape(-1, 3).astype(np.float64)
    values = np.zeros(8)  # values irrelevant, only the basis matrix is checked
    proc = RAHTProcessor(bsize=bsize)
    T, _, _ = proc.compute(Vblock, values, materialize_matrix=True)

    assert T.shape == (8, 8)
    # Row 0 = DC_root (this module's coefficient-ordering convention, see raht_forward):
    # for a fully-occupied unit-weight block, every leaf still contributes to it, each with
    # the standard Hadamard DC magnitude 1/sqrt(8).
    np.testing.assert_allclose(np.abs(T[0, :]), 1.0 / np.sqrt(8), atol=1e-9)
    # The transform as a whole is still exactly orthonormal (the real invariant that matters,
    # already covered more thoroughly by test_raht_basis_orthonormal; repeated here for this
    # specific, hand-verifiable 8-point case).
    np.testing.assert_allclose(T @ T.T, np.eye(8), atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Perfect reconstruction, including duplicate-voxel-position points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_raht_perfect_reconstruction(seed):
    rng = np.random.default_rng(seed)
    bsize = 8
    n_leaves = rng.integers(3, 30)
    all_coords = np.array(np.meshgrid(*[range(bsize)] * 3)).T.reshape(-1, 3)
    leaf_idx = rng.choice(all_coords.shape[0], size=n_leaves, replace=False)
    leaf_coords = all_coords[leaf_idx].astype(np.float64)

    # Duplicate some positions (exercises the weight>1 dedup path).
    dup_counts = rng.integers(1, 4, size=n_leaves)
    Vblock = np.repeat(leaf_coords, dup_counts, axis=0)
    n_points = Vblock.shape[0]
    values = rng.normal(size=n_points)

    proc = RAHTProcessor(bsize=bsize)
    _, coeffs, tree = proc.compute(Vblock, values)
    rec_per_leaf = proc.inverse(tree, coeffs)

    # Expected: each leaf's reconstructed value equals the mean of the duplicate points that
    # were averaged onto it going in (RAHT round-trips the averaged leaf value exactly; the
    # averaging itself, not RAHT, is what's lossy for duplicate points -- correct behavior).
    expected = np.zeros(tree.n_leaves)
    counts = np.zeros(tree.n_leaves)
    np.add.at(expected, tree.dup_inverse_idx, values)
    np.add.at(counts, tree.dup_inverse_idx, 1.0)
    expected = expected / counts

    np.testing.assert_allclose(rec_per_leaf, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Corrected root-DC invariant -- and an explicit guard against the naive/wrong version
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [20, 21, 22])
def test_raht_root_dc_is_scaled_weighted_mean(seed):
    rng = np.random.default_rng(seed)
    bsize = 8
    n = rng.integers(4, 40)
    all_coords = np.array(np.meshgrid(*[range(bsize)] * 3)).T.reshape(-1, 3)
    idx = rng.choice(all_coords.shape[0], size=n, replace=False)
    Vblock = all_coords[idx].astype(np.float64)
    values = rng.normal(size=n)

    proc = RAHTProcessor(bsize=bsize)
    _, coeffs, tree = proc.compute(Vblock, values)

    w = tree.leaf_weights  # all 1.0 here (no duplicates), but keep the general formula
    total_w = w.sum()
    correct_dc = np.sum(w * values) / np.sqrt(total_w)
    naive_wrong_dc = np.average(values, weights=w)  # plain weighted mean -- NOT the invariant

    assert coeffs[0] == pytest.approx(correct_dc, abs=1e-8)
    # Guard against a future "fix" silently reintroducing the wrong invariant: for n > 1 these
    # two must differ (they only coincide at n=1, where sqrt(total_w)=1=mean scale trivially).
    assert n == 1 or abs(coeffs[0] - naive_wrong_dc) > 1e-6


# ---------------------------------------------------------------------------
# 6. Trivial single-point block
# ---------------------------------------------------------------------------

def test_raht_single_point_block():
    Vblock = np.array([[3.0, 5.0, 2.0]])
    values = np.array([42.0])
    proc = RAHTProcessor(bsize=8)
    _, coeffs, tree = proc.compute(Vblock, values)

    assert tree.n_leaves == 1
    assert coeffs.shape == (1,)
    assert coeffs[0] == pytest.approx(42.0, abs=1e-9)

    rec = proc.inverse(tree, coeffs)
    assert rec[0] == pytest.approx(42.0, abs=1e-9)
