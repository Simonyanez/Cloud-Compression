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
    def __init__(self, block_id: int, sl_weight: float, sl_percentage: float):
        super().__init__(block_id)
        self.sl_weight = sl_weight
        self.sl_percentage = sl_percentage
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
        most_pointed = np.argsort(self.S)[::-1]
        num_nodes = int(np.round((len(most_pointed)*self.sl_percentage)))
        self.selected_nodes = most_pointed[:num_nodes]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:,0], pairs[:,1]] = self.sl_weight
        self.edges = np.append(self.edges, pairs, axis=0)
        
    def _attribute_motion_matrix(self, A: np.ndarray) -> np.ndarray:
        Y = A[:, 0]
        row_wise = self.weights * Y
        col_wise = self.weights * Y[:, np.newaxis]
        M = (row_wise - col_wise)/ 255*2
        return M

    def _sink_nodes_vector(self, M: np.ndarray) -> np.ndarray:
        # Initialize sink vector
        sink_vector = np.zeros(M.shape[0])
        unique, count = self._get_decreasing_count(M)
        sink_vector[unique] = count
        # Normalization
        neighbors_count = np.sum(self.weights > 0, axis=1)
        sink_vector = sink_vector/neighbors_count 
        return sink_vector

    def _get_decreasing_count(self, M: np.ndarray) -> np.ndarray:
        np.fill_diagonal(M, np.inf)
        dec_i = np.argmin(M, axis=1)
        unique, count = np.unique(dec_i, return_counts=True)
        return unique, count 

if __name__ == "__main__":
    from src.objects import *
    from src.visualization import *
    import src.ply as ply
    from transforms import *

    file_conditions = os.path.exists('V_longdress.npy') and os.path.exists('C_longdress.npy')
    if file_conditions:
        V = np.load('V_longdress.npy')
        C_rgb = np.load('C_longdress.npy')
    else:
        V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  
        np.save('V_longdress.npy',V)
        np.save('C_longdress.npy',C_rgb) 
    
    point_cloud = PointCloud(V,C_rgb, bsize=16)
    block = point_cloud.get_block(2400)
    Vblock, Ablock = block.Vblock, block.Ablock
    
    graph = AttributeGraph(Vblock, Ablock,block_fraction=0.01)
    # M = graph._attribute_motion_matrix(Ablock)
    # print(f"This is attribute motion matrix {M}")
    # S = graph._sink_nodes_vector(M)
    # print(f"This is sink vector {S}")
    GFT_processor = GFT()
    GFT_matrix, Coeffs = GFT_processor(graph, block)
    visualizer = Visualizer()
    visualizer(graph, block)
    visualizer.visualize_block()
    visualizer.add_selected_nodes()
    visualizer.display()
    print(min(GFT_matrix[:,0]), max(GFT_matrix[:,0]))
    visualizer.visualize_base(GFT_matrix[:,0])
    visualizer.add_selected_nodes()
    visualizer.display()
    visualizer.visualize_motion_matrix()
    visualizer.display()
    visualizer.visualize_sink()
    visualizer.display()
    visualizer.visualize_graph()
    visualizer.display()