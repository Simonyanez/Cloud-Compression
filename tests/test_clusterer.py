# Test file for Clusterer class
import numpy as np
import pytest
from pcadc.clusterer import Clusterer, Codebook


# -------------------------------
# Codebook tests
# -------------------------------


def test_modify_centroid():
    centroids = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
    labels = np.array([0, 1])
    cb = Codebook(centroids=centroids, labels=labels)

    new_centroid = np.array([[7.0, 8.0, 9.0]])
    cb.modify_centroid(1, new_centroid)

    assert np.allclose(cb.centroids[1], [7.0, 8.0, 9.0])


# -------------------------------
# Clusterer tests
# -------------------------------
def test_normalize_matrix_l2():
    X = np.array([[3.0, 4.0, 0.0],
                  [0.0, 0.0, 0.0]])  # includes zero vector
    normalized = Clusterer.normalize_matrix(X, method="l2")

    # First row should be normalized to unit length
    assert np.allclose(normalized[0], [0.6, 0.8, 0.0])
    # Second row stays zero
    assert np.allclose(normalized[1], [0.0, 0.0, 0.0])


def test_fixed_centroid_kmeans_converges():
    X = np.array([[0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0]])

    clusterer = Clusterer(n_clusters=2)
    centers, labels = clusterer.fixed_centroid_kmeans(
        X, fixed_center=np.array([0.0, 0.0, 0.0]), max_iter=10)

    # Should produce two centroids
    assert centers.shape[0] == 2
    # Labels should be valid indices
    assert set(labels).issubset({0, 1})
