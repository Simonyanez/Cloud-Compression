import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import pairwise_distances_argmin_min, root_mean_squared_error
import logging

# Assume you already have these implemented
from .blocks import Block
from .color import FitCollection

logger = logging.getLogger(__name__)


@dataclass
class Codebook:
    centroids: np.ndarray
    labels: np.ndarray

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def modify_centroid(self, idx: int, centroid: np.ndarray) -> None:
        assert idx < self.centroids.shape[0], "Index exceeds number of clusters"
        assert centroid.shape[0] == 1, "Centroid must be a single row vector"
        assert centroid.shape[1] == self.centroids.shape[1], "Centroid shape mismatch"
        self.centroids[idx, :] = centroid

    def find_best_centroid(self, block: Block) -> np.ndarray:
        """Find the centroid that best matches a given block (direction-based)."""
        Vblock, Ablock = block.get_data()
        (N, _) = Vblock.shape

        # Predictions for each centroid
        Y_estimates = self.centroids @ Vblock
        Y_truth = np.tile(Ablock[:, 0], (N, 3))

        # Compute RMSE for each centroid
        rmse_mat = root_mean_squared_error(Y_truth, Y_estimates)
        best_idx = np.argmin(rmse_mat, axis=1)

        return self.centroids[best_idx], best_idx


class Clusterer:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def __call__(self, fit_collection: FitCollection) -> Codebook:
        slope_matrix = self.get_slope_matrix(fit_collection, norm_flag=True)
        centroids, labels = self.fixed_centroid_kmeans(slope_matrix)
        return Codebook(centroids=centroids, labels=labels)

    @staticmethod
    def get_slope_matrix(fit_collection: FitCollection, norm_flag: bool = True) -> np.ndarray:
        slope_matrix = fit_collection.get_slopes()
        if norm_flag:
            slope_matrix = Clusterer.normalize_matrix(slope_matrix)
        return slope_matrix

    @staticmethod
    def normalize_matrix(matrix: np.ndarray, method: Optional[str] = "l2") -> np.ndarray:
        if method not in ("l2", "max"):
            raise ValueError("method must be 'l2' or 'max'")

        norms = np.linalg.norm(matrix, axis=1) if method == "l2" else np.max(
            np.abs(matrix), axis=1)
        zero_norms = norms == 0
        if np.any(zero_norms):
            logger.warning(
                f"{np.sum(zero_norms)} rows have zero norm; leaving them unchanged.")
            norms[zero_norms] = 1.0  # avoid division by zero

        return matrix / norms[:, np.newaxis]

    def fixed_centroid_kmeans(
        self,
        X: np.ndarray,
        fixed_center: Optional[np.ndarray] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        if fixed_center is None:
            fixed_center = np.zeros((1, X.shape[1]))

        rng = np.random.default_rng(seed=42)
        other_centers = rng.choice(X, size=self.n_clusters - 1, replace=False)
        centers = np.vstack([fixed_center, other_centers])

        for it in range(max_iter):
            labels = pairwise_distances_argmin_min(X, centers)[0]
            new_centers = [fixed_center]

            for k in range(1, self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers.append(members.mean(axis=0))
                else:
                    new_centers.append(rng.choice(X))  # avoid dead cluster

            new_centers = np.vstack(new_centers)
            shift = np.linalg.norm(centers - new_centers)

            if verbose:
                logger.debug(f"Iteration {it}, centroid shift: {shift:.6f}")

            if shift < tol:
                break
            centers = new_centers

        logger.info(f"Custom K-means finished with {self.n_clusters} clusters")
        return centers, labels
