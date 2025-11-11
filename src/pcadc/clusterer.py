import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from sklearn.metrics import pairwise_distances_argmin_min, root_mean_squared_error
from sklearn.cluster import kmeans_plusplus
import logging

# Assume you already have these implemented
from .blocks import Block
from .color import FitCollection, Approximator

logger = logging.getLogger(__name__)


@dataclass
class Codebook:
    centroids: np.ndarray
    labels: np.ndarray
    assignation: np.ndarray

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def modify_centroid(self, idx: int, centroid: np.ndarray) -> None:
        assert idx < self.centroids.shape[0], "Index exceeds number of clusters"
        assert centroid.shape[0] == 1, "Centroid must be a single row vector"
        assert centroid.shape[1] == self.centroids.shape[1], "Centroid shape mismatch"
        self.centroids[idx, :] = centroid

    def assign(self, blocks: List[Block], V: np.ndarray, A: np.ndarray):
        # NOTE: This assignation is made based on TRUE LUMINANSCE
        # While the previous clusters are made based on LINEAR FIT
        # FIXME: Block should be normalized for calculation
        for i, block in enumerate(blocks):
            block.init_data(V, A)
            self.assignation[i] = self.find_best_centroid(block)
            block.clear_data()

        # Log the final distribution of assignments
        counts = np.bincount(self.assignation)
        num_centroids = self.centroids.shape[0]

        assignment_info = f"Final assignment distribution across {num_centroids} centroids:\n"
        for i in range(num_centroids):
            if i < len(counts):
                count = counts[i]
            else:
                count = 0  # Handle centroids with no assignments
            assignment_info += f"Centroid {i}: {count} members\n"

        logger.info(assignment_info)

    # FIXME: Not sure if this is right
    def find_best_centroid(self, block: Block) -> np.intp:
        """Find the centroid that best matches a given block (direction-based)."""
        Vblock, Ablock = block.get_data()
        N = Vblock.shape[0]
        num_centroids = self.centroids.shape[0]

        # Handle the single-point block edge case to prevent errors
        if N < 2:
            # For a single point, the 'best fit' is trivially the first centroid,
            # or you could assign a specific 'flat' centroid.
            # Returning a consistent index is important for downstream logic.
            logger.info(
                f"One-point block found. Returning flat centroid label.")
            return np.intp(0)

        # Predictions for each centroid against the block's Vblock coordinates
        # The result should be (num_centroids, N)
        Y_estimates = Vblock @ self.centroids.T

        # The ground truth luminance (Y_truth)
        Y_truth = Ablock[:, 0].reshape(-1, 1)

        # Compute RMSE for each centroid's predictions
        # We use broadcasting to compare each row of Y_estimates to Y_truth
        # The result will be a 1D array of shape (num_centroids,)
        rmse_per_centroid = np.sqrt(
            np.mean((Y_estimates - Y_truth)**2, axis=0))

        # Find the index of the centroid with the minimum RMSE
        best_idx = np.argmin(rmse_per_centroid)

        # Log the single best centroid and its RMSE
        logger.info(
            f"Block {block.block_id} assigned to centroid {best_idx} with RMSE of {rmse_per_centroid[best_idx]:.4f}.")

        return best_idx

    def get_assigned_centroid(self, block_idx: int):
        return self.centroids[self.assignation[block_idx]]

    def get_labeled_centroid(self, block_idx: int):
        return self.centroids[self.labels[block_idx]]

    def find_centroid_label(self, centroid: np.ndarray) -> int:
        """
        Return the index (label) of the given centroid in the codebook.
        Raises ValueError if not found.
        """
        centroid = np.asarray(centroid).reshape(1, -1)  # ensure 2D row
        matches = np.all(self.centroids == centroid,
                         axis=1)  # row-wise compare
        indices = np.where(matches)[0]

        if len(indices) == 0:
            err_msg = f"Centroid ({centroid}) not found in codebook."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return indices[0]  # or return indices if you expect multiple


class Clusterer:
    def __init__(self, n_clusters: int, normalize_slopes: bool):
        self.n_clusters = n_clusters
        self.normalize_slopes = normalize_slopes
        logger.info(
            f"Clusterer initialized with {n_clusters} number of clusters")

    def __call__(self, fit_collection: FitCollection) -> Codebook:
        slope_matrix = self.get_slope_matrix(
            fit_collection, self.normalize_slopes)
        centroids, labels = self.fixed_centroid_kmeans(slope_matrix)
        assignation = np.zeros_like(labels)
        return Codebook(centroids=centroids, labels=labels, assignation=assignation)

    @staticmethod
    def get_slope_matrix(fit_collection: FitCollection, norm_flag: bool) -> np.ndarray:
        slope_matrix = fit_collection.get_slopes()
        if norm_flag:
            logger.info(
                f"Normalize is activated in clusterer. Normalizing slope matrix...")
            slope_matrix = Clusterer.normalize_matrix(slope_matrix)
        return slope_matrix

    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Scales each row of the matrix to the range [-1, 1] using Min-Max scaling.
        """
        min_vals = np.min(matrix, axis=1, keepdims=True)
        max_vals = np.max(matrix, axis=1, keepdims=True)

        # Avoid division by zero when max == min
        denom = max_vals - min_vals
        zero_range = denom == 0
        if np.any(zero_range):
            logger.warning(f"{np.sum(zero_range)} rows have zero range; leaving them unchanged.")
            denom[zero_range] = 1.0

        scaled = 2 * ((matrix - min_vals) / denom) - 1
        return scaled

    def fixed_centroid_kmeans(
        self,
        X: np.ndarray,
        fixed_center: Optional[np.ndarray] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        if fixed_center is None:
            logger.info(
                f"Automatic fixed center for constant luminansce centroid {fixed_center}")
            fixed_center = np.zeros((1, X.shape[1]))

        # Use k-means++ for better initialization of the other centers
        other_centers, _ = kmeans_plusplus(
            X, n_clusters=self.n_clusters - 1, random_state=42
        )
        centers = np.vstack([fixed_center, other_centers])

        rng = np.random.default_rng(seed=42)

        for it in range(max_iter):
            labels = pairwise_distances_argmin_min(X, centers)[0]
            new_centers = [fixed_center]

            for k in range(1, self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers.append(members.mean(axis=0))
                else:
                    # If a cluster becomes empty, re-initialize it with a random point
                    # This is a simple way to handle dead clusters
                    new_centers.append(rng.choice(X))

            new_centers = np.vstack(new_centers)
            shift = np.linalg.norm(centers - new_centers)

            if verbose:
                logger.debug(f"Iteration {it}, centroid shift: {shift:.6f}")

            if shift < tol:
                break
            centers = new_centers

        logger.info(
            f"Custom K-means finished after {it + 1} iterations with {self.n_clusters} clusters.")

        centroids_info = ""
        for i, centroid in enumerate(centers):
            member_count = np.sum(labels == i)
            centroids_info += f"Centroid {i}: {centroid} - Members: {member_count}\n"
        logger.info(
            "Centroid distribution for coefficients (might change using luminance) \n" + centroids_info)
        return centers, labels
