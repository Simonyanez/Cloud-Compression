import numpy as np
from copy import deepcopy
from .blocks import BlockMetadata
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from scipy.spatial.distance import cdist
import logging
logging.basicConfig(filename="logs/graphs.log", 
                    filemode="w", 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)

# NOTE: This is a good example of the Decorator Pattern
# NOTE: This is a good example of the Template Pattern

@dataclass
class GraphMetadata:
    block_id: str
    graph_type: str
    distance_threshold: float
    luminance_centroid: np.ndarray
    self_loop_threshold: Optional[float]
    self_loop_weight: Optional[float]

    def get_graph_id(self):
        graph_id = f"{self.block_id}_{self.graph_type}"
        if self.self_loop_threshold is not None:
            graph_id += f"_sl{self.self_loop_threshold}_{self.self_loop_weight}"
        return graph_id


class GraphBase(ABC):
    def __init__(self, metadata: GraphMetadata):
        self.metadata = metadata
        self.weights: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None

    def set_data(self, weights: np.ndarray, edges: np.ndarray) -> None:
        self.weights = weights
        self.edges = edges

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.weights is not None, "Graph weights haven't been initialized"
        assert self.edges is not None, "Graph edges haven't been initialized"
        return self.weights, self.edges

    def clear_data(self) -> None:
        self.weights = None
        self.edges = None


class StructuralGraph(GraphBase):
    def __init__(self, block_metadata: BlockMetadata):

        block_id = block_metadata.get_block_id()
        metadata = GraphMetadata(
            block_id=block_id,
            graph_type="Structural",
            distance_threshold=np.sqrt(3),
            luminance_centroid=np.array([0, 0 ,0]),
            self_loop_threshold=None,
            self_loop_weight=None
        )
        super().__init__(metadata)

    def set_data(self,
                 V: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None,
                 edges: Optional[np.ndarray] = None):
        """
        Initialize either from vertex data (V) or precomputed weights/edges.
        """
        if V is not None:
            weights, edges = self._compute_structural_graph(V)
        elif weights is None or edges is None:
            raise ValueError("Either V or (weights, edges) must be provided")

        super().set_data(weights=weights, edges=edges)

    @staticmethod
    def _euclidean_distance_matrix(V: np.ndarray):
        return cdist(V, V)

    def _inverse_distance_matrix(self, D: np.ndarray, epsilon=1e-5):
        iD = np.zeros_like(D)
        non_zero_mask = (D > 0) & (D <= self.metadata.distance_threshold + epsilon)
        iD[non_zero_mask] = 1 / D[non_zero_mask]
        return iD

    def _compute_structural_graph(self, V: np.ndarray):
        D = self._euclidean_distance_matrix(V)
        iD = self._inverse_distance_matrix(D)
        weights = iD.T + iD
        edges = np.column_stack(np.nonzero(iD))
        return weights, edges


class AttributeGraph(GraphBase):
    def __init__(self,
                 structural_graph: StructuralGraph,
                 luminance_centroid: np.ndarray,
                 self_loop_threshold: float,
                 self_loop_weight: float):
        metadata = deepcopy(structural_graph.metadata)
        metadata.graph_type = "Attribute"
        metadata.luminance_centroid = luminance_centroid
        metadata.self_loop_threshold = self_loop_threshold
        metadata.self_loop_weight = self_loop_weight

        super().__init__(metadata)
        self.structural_graph = structural_graph
        self.self_loop_threshold = self_loop_threshold
        self.self_loop_weight = self_loop_weight

    def __getattr__(self, name):
        """Forward unknown attributes/methods to the structural graph."""
        return getattr(self.structural_graph, name)

    def set_data(
        self,
        V: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
    ):
        """
        Initialize either from vertex/attribute data (V, A) or precomputed weights/edges.
        """
        if V is not None and A is not None:
            self.structural_graph.set_data(V)
            weights, edges = self.structural_graph.get_data()
            super().set_data(weights=weights.copy(), edges=edges.copy())
            self._compute_attribute_graph(A)

        elif weights is not None and edges is not None:
            self.weights = weights
            self.edges = edges
        else:
            raise ValueError("Either (V, A) or (weights, edges) must be provided")

    def _compute_attribute_graph(self, A: np.ndarray) -> None:
        self.M = self._attribute_motion_matrix(A)
        self.S = self._sink_nodes_vector(self.M, normalization="minmax")
        self._self_loops_selection()

    def _self_loops_selection(self):
        """Select nodes for self-loops based on sink vector `self.S`."""
        self.most_pointed = np.argsort(self.S)[::-1]

        if self.self_loop_threshold is None:
            raise ValueError("No threshold provided for self-loops")

        self.selected_nodes = np.argwhere(self.S >= self.self_loop_threshold).flatten()
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

    def _attribute_motion_matrix(self, A: np.ndarray) -> np.ndarray:
        Y = A[:, 0]
        return self.weights * np.subtract.outer(Y, Y) / 255

    def _sink_nodes_vector(self, M: np.ndarray, normalization: str = "standard") -> np.ndarray:
        sink_vector = np.zeros(M.shape[0])
        unique, count = self._get_decreasing_count(M)
        sink_vector[unique] = count

        if normalization == "standard":
            neighbors_count = np.sum(self.weights > 0, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                sink_vector = np.true_divide(sink_vector, neighbors_count)
                sink_vector[~np.isfinite(sink_vector)] = 0

        elif normalization == "minmax":
            min_val, max_val = np.min(sink_vector), np.max(sink_vector)
            if max_val > min_val:
                sink_vector = (sink_vector - min_val) / (max_val - min_val)
            else:
                sink_vector[:] = 0
        else:
            raise ValueError(f"Unknown normalization mode: {normalization}")

        return sink_vector

    def _get_decreasing_count(self, M: np.ndarray) -> np.ndarray:
        M_masked = np.copy(M)
        M_masked[self.weights == 0] = np.inf
        np.fill_diagonal(M_masked, np.inf)
        dec_i = np.argmin(M_masked, axis=1)
        return np.unique(dec_i, return_counts=True)


if __name__ == "__main__":
    pass
