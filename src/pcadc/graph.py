import numpy as np
from copy import deepcopy
from .blocks import BlockMetadata, AuxiliaryBlockMetadata
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from scipy.spatial.distance import cdist
import logging
import os

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "graphs.log")

logging.basicConfig(filename=log_file_path,
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
    centroid_label: int
    self_loop_percentage: Optional[float]
    self_loop_weight: Optional[float]

    @property
    def graph_id(self):
        graph_id = f"{self.block_id}_{self.graph_type}"
        if self.self_loop_percentage is not None:
            graph_id += f"_sl_{self.self_loop_percentage}_{self.self_loop_weight}"
        return graph_id

    @property
    def graph_descriptor(self):
        return f"{self.graph_type}_{self.centroid_label}"

    @property
    def graph_cluster_descriptor(self):
        return f"{self.block_id}_{self.graph_type}_{self.centroid_label}"


class GraphBase(ABC):
    def __init__(self, metadata: GraphMetadata):
        self.metadata = metadata
        self.weights: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        logger.debug(
            f"Graph object initialized with metadata: {self.metadata.graph_id}")

    @property
    def graph_id(self):
        return self.metadata.graph_id

    def set_data(self, weights: np.ndarray, edges: np.ndarray) -> None:
        logger.debug(f"Updating weights and edges for graph: {self.graph_id}")
        logger.debug(
            f"Weights shape: {weights.shape}, Edges shape: {edges.shape}")
        self.weights = weights
        self.edges = edges

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.weights is not None, "Graph weights haven't been initialized"
        assert self.edges is not None, "Graph edges haven't been initialized"
        logger.debug(
            f"Retrieving data for graph: {self.graph_id}. Weights shape: {self.weights.shape}, Edges shape: {self.edges.shape}")
        return self.weights, self.edges

    def clear_data(self) -> None:
        self.weights = None
        self.edges = None
        logger.debug(f"Data cleared for graph: {self.graph_id}")


class StructuralGraph(GraphBase):
    def __init__(self, block_metadata: BlockMetadata | AuxiliaryBlockMetadata):

        block_id = block_metadata.block_id
        metadata = GraphMetadata(
            block_id=block_id,
            graph_type="Structural",
            distance_threshold=np.sqrt(3),
            centroid_label=0,
            luminance_centroid=np.array([0, 0, 0]),
            self_loop_percentage=None,
            self_loop_weight=None
        )
        super().__init__(metadata)

    def set_metadata(self, metadata: GraphMetadata):
        self.metadata = metadata

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
        non_zero_mask = (D > 0) & (
            D <= self.metadata.distance_threshold + epsilon)
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
                 centroid_label: int,
                 self_loop_percentage: float,
                 self_loop_weight: float):
        metadata = deepcopy(structural_graph.metadata)
        metadata.graph_type = "Attribute"
        metadata.luminance_centroid = luminance_centroid
        metadata.centroid_label = centroid_label
        metadata.self_loop_percentage = self_loop_percentage
        metadata.self_loop_weight = self_loop_weight

        super().__init__(metadata)
        self.structural_graph = structural_graph
        self.self_loop_percentage = self_loop_percentage
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
            raise ValueError(
                "Either (V, A) or (weights, edges) must be provided")

    def _compute_attribute_graph(self, A: np.ndarray) -> None:
        self.M = self._attribute_motion_matrix(A)
        self.S = self._sink_nodes_vector(self.M)
        self._self_loops_selection()

    def _self_loops_selection(self):
        """Select nodes for self-loops based on sink vector `self.S`."""
        self.most_pointed = np.argsort(self.S)[::-1]

        if self.self_loop_percentage is None:
            raise ValueError("No percentage provided for self-loops")

        # self.selected_nodes = np.argwhere(
        #     self.S >= self.self_loop_percentage).flatten()
        index = int(self.self_loop_percentage * self.S.shape[0])
        self.selected_nodes = self.most_pointed[:index]
        
        # Detailed logging for self-loop diagnostics
        # TODO: Sink vector is crucial for this implemention. Consider creating its own class for easier monitoring
        max_s_value = np.max(self.S) if self.S.size > 0 else 0
        logger.info(
            f"Graph {self.metadata.graph_cluster_descriptor}: "
            f"Found {len(self.selected_nodes)} nodes for self-loops "
            f"with percentage {self.self_loop_percentage}. "
            f"Max S value was {max_s_value:.4f}."
        )

        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

    def _attribute_motion_matrix(self, A: np.ndarray) -> np.ndarray:
        # FIXME: Does it make sense to normalize by max possible value if the Y is an approximated version based in luminance fits?
        Y = A[:, 0]
        # TODO: Check if it's pertinent to use 255 for the cluster slopes
        return self.weights * np.subtract.outer(Y, Y) #/ 255

    def _sink_nodes_vector(self, M: np.ndarray) -> np.ndarray:
        sink_vector = np.zeros(M.shape[0])
        unique, count = self._get_decreasing_count(M)
        sink_vector[unique] = count
        return sink_vector

    def _get_decreasing_count(self, M: np.ndarray) -> np.ndarray:
        # NOTE: This workout avoid non-dynamic neighbours and self-node comparison
        M_masked = np.copy(M)
        M_masked[self.weights == 0] = np.inf
        np.fill_diagonal(M_masked, np.inf)
        # Get the most decreased nodes
        dec_i = np.argmin(M_masked, axis=1)
        return np.unique(dec_i, return_counts=True)


if __name__ == "__main__":
    pass
