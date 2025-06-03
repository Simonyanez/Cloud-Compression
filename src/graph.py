import numpy as np
import os
main_folder = os.getcwd()
import sys
sys.path.insert(0, main_folder)
from typing import Optional
from uuid import *

class Graph():
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.weights = None
        self.edges = None

    def _init_data(self, weights: np.ndarray, edges: np.ndarray):
        self.weights = weights
        self.edges = edges

    def _del_data(self):
        self.weights = None
        self.edges = None    

class StructuralGraph(Graph):
    def __init__(self, block_id: int):
        super().__init__(block_id)
        self.id = ('0.0_0.0')

    def _init_data(
        self,
        V: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        threshold: float = np.sqrt(3),
        epsilon: float = 1e-5,
    ):
        """
        Initialize either from vertex data (V) or precomputed weights/edges.
        """
        if V is not None:
            # Compute from vertices
            self.threshold = threshold
            self.epsilon = epsilon
            weights, edges = self._compute_structural_graph(V)
        elif weights is not None and edges is not None:
            # Direct initialization (reconstruction)
            self.threshold = threshold  # Optional: Load from storage if needed
            self.epsilon = epsilon
        else:
            raise ValueError("Either V or (weights, edges) must be provided")

        super()._init_data(weights=weights, edges=edges)

    def _del_data(self):
        return super()._del_data()
        
    def _euclidean_distance_matrix(self, V: np.ndarray):
        # FIXME: Has no self calls
        N = V.shape[0]
        squared_norms = np.sum(V**2, axis=1)  # Compute Euclidean Distance Matrix (EDM)
        D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))
        return D 

    def _inverse_distance_matrix(self, D: np.ndarray):
        # FIXME: Has no self calls
        iD = np.zeros_like(D) 
        non_zero_mask = (D > 0) & (D <= self.threshold + self.epsilon)
        iD[non_zero_mask] = 1 / D[non_zero_mask]
        iD[np.where(D > self.threshold + self.epsilon)] = 0
        iD[np.where(D == 0)] = 0
        return iD

    # Keep existing methods (_euclidean_distance_matrix, etc.) unchanged
    def _compute_structural_graph(self, V: np.ndarray):
        D = self._euclidean_distance_matrix(V)
        iD = self._inverse_distance_matrix(D)
        weights = iD.T + iD
        edges = np.column_stack(np.nonzero(iD))
        return weights, edges

class AttributeGraph(StructuralGraph):
    def __init__(self, block_id: int, sl_weight: float, sl_percentage: Optional[float] = None, sl_threshold: Optional[float] = None):
        super().__init__(block_id)
        self.sl_weight = sl_weight
        self.sl_percentage = sl_percentage
        self.sl_threshold = sl_threshold
        self.id = f"{sl_weight}_{sl_percentage}"

    def _init_data(
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
            # Compute from vertices/attributes
            super()._init_data(V=V)  # Calls _compute_structural_graph
            self._compute_attribute_graph(A)
        elif weights is not None and edges is not None:
            # Direct initialization (reconstruction)
            super()._init_data(weights=weights, edges=edges)
        else:
            raise ValueError("Either (V, A) or (weights, edges) must be provided")
    
    def _del_data(self):
        return super()._del_data()
    
    def _compute_attribute_graph(self, A: np.ndarray) -> None:
        # TODO: Refactor and optimize this process. This implementation is horrible
        self.M = self._attribute_motion_matrix(A)
        self.S = self._sink_nodes_vector(self.M)
        self._self_loops_selection()

    def _self_loops_selection(self):
        """
        Selects nodes for self-loops based on sink vector `self.S`.

        - If `threshold` is provided, selects nodes with sink values >= threshold.
        - Else, selects the top `sl_percentage` of nodes by sink score.
        """
        self.most_pointed = np.argsort(self.S)[::-1]

        if self.sl_threshold is not None:
            self.selected_nodes = np.argwhere(self.S >= self.sl_threshold)
        elif self.sl_percentage is not None:
            num_nodes = int(np.round(len(self.most_pointed) * self.sl_percentage))
            self.selected_nodes = self.most_pointed[:num_nodes]
        else:
            raise ValueError("There is no value selected")
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.sl_weight
        self.edges = np.append(self.edges, pairs, axis=0)

    def _attribute_motion_matrix(self, A: np.ndarray) -> np.ndarray:
        Y = A[:, 0]
        row_wise = Y
        col_wise = Y[:, np.newaxis]
        M = self.weights*(row_wise - col_wise)/ 255*2
        return M

    def _sink_nodes_vector(self, M: np.ndarray) -> np.ndarray:
        # Initialize sink vector
        sink_vector = np.zeros(M.shape[0])
        unique, count = self._get_decreasing_count(M)
        sink_vector[unique] = count

        # Normalization
        neighbors_count = np.sum(self.weights > 0, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sink_vector = np.true_divide(sink_vector, neighbors_count)
            sink_vector[~np.isfinite(sink_vector)] = 0
        return sink_vector

    def _get_decreasing_count(self, M: np.ndarray) -> np.ndarray:
        M_masked = np.copy(M)
        M_masked[self.weights == 0] = np.inf
        np.fill_diagonal(M_masked, np.inf)
        dec_i = np.argmin(M_masked, axis=1)
        unique, count = np.unique(dec_i, return_counts=True)
        return unique, count

if __name__ == "__main__":
    from src.objects import *
    from src.visualization import *
    import src.ply as ply
    from transforms import *
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm

    point_cloud = PointCloud()
    point_cloud(Path("res/longdress_vox10_1051.ply"))
    point_cloud.do_block_partitioning(bsize=16)

    V = point_cloud.V
    A = point_cloud.A
    blocks = point_cloud.get_all_blocks()
    degree = 4

    for i, block in tqdm(enumerate(blocks), "Polyfit per block"):
        block._init_data(V, A)

        # Use AttributeGraph just to get access to graph.S
        graph = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.5)
        graph._init_data(block.Vblock, block.Ablock)
        ordered_idx = np.argsort(graph.S)[::-1]
        print(f"Ordered_idx {ordered_idx} \n Ordered value {graph.S[ordered_idx]} \n Ordered positions {block.Vblock[ordered_idx]} vs Unordered positions: {block.Vblock} ")
        # Prepare data
        x = np.arange(len(graph.S))
        y = np.array(graph.S)

        # If y is multidimensional, pick the first attribute
        if y.ndim > 1:
            y = y[:, 0]

        i0 = np.argwhere(y>0)
        # Fit polynomial of degree 2
        fit = np.polyfit(x[i0][:,0], y[i0][:,0], degree)
        p = np.poly1d(fit)
        y_fit = p(x)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'bo', label='Original Data')
        plt.plot(x, y_fit, 'r-', label=f'Polyfit (degree {degree})')
        plt.title(f'Polynomial Fit of graph.S for Block {block.id}')
        plt.xlabel('Node Index')
        plt.ylabel('Attribute Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        graph._del_data()
        block._del_data()
