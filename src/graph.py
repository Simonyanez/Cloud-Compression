import numpy as np
import os
main_folder = os.getcwd()
import sys
sys.path.insert(0, main_folder)
from typing import Optional
from uuid import *

class Graph():
    def __init__(self, weights: Optional[np.ndarray], edges: Optional[np.ndarray]):
        self.weights = weights
        self.edges = edges
        self.graph_id = uuid4()

class StructuralGraph(Graph):
    def __init__(self, V, threshold: float = np.sqrt(3), epsilon = 1e-5):
        self.threshold = threshold
        self.epsilon = epsilon
        weights, edges = self._compute_structural_graph(V)
        super().__init__(weights=weights, edges=edges)

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

    def _compute_structural_graph(self, V: np.ndarray):
        D = self._euclidean_distance_matrix(V)
        iD = self._inverse_distance_matrix(D)
        weights = iD.T + iD
        idx = np.nonzero(iD)
        edges = np.column_stack(( idx[1], idx[0]))
        return weights, edges

class AttributeGraph(StructuralGraph):
    # NOTE: Consider renaming this class. This can be misleading
    def __init__(self,  V: np.ndarray, A: np.ndarray, sl_weight: float, block_fraction: float):
        super().__init__(V)
        self._compute_attribute_graph(A, sl_weight=sl_weight, block_fraction=block_fraction)
        
    def _compute_attribute_graph(self, A: np.ndarray, sl_weight: float, block_fraction: float) -> None:
        # TODO: Refactor and optimize this process. This implementation is horrible
        self.M = self._attribute_motion_matrix(A)
        self.S = self._sink_nodes_vector(self.M)
        most_pointed = np.argsort(self.S)[::-1]
        num_nodes = int(np.round((len(most_pointed)*block_fraction)))
        self.selected_nodes = most_pointed[:num_nodes]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:,0], pairs[:,1]] = sl_weight
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
    import utils.ply as ply
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