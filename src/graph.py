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
        if self.sl_percentage is not None:
            self.id = f"{sl_weight}_{sl_percentage}"
        if self.sl_threshold is not None:
            self.id = f"{sl_weight}_{sl_threshold}"

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
        self.S = self._sink_nodes_vector(self.M, normalization="minmax")
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

    def _sink_nodes_vector(self, M: np.ndarray, normalization: str = "standard") -> np.ndarray:
        """
        Compute a sink vector with optional normalization.

        Parameters:
        - M: Adjacency or count matrix.
        - normalization: "standard" (default) or "minmax".

        Returns:
        - sink_vector: np.ndarray
        """
        # Initialize sink vector
        sink_vector = np.zeros(M.shape[0])
        unique, count = self._get_decreasing_count(M)
        sink_vector[unique] = count

        if normalization == "standard":
            neighbors_count = np.sum(self.weights > 0, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                sink_vector = np.true_divide(sink_vector, neighbors_count)
                sink_vector[~np.isfinite(sink_vector)] = 0

        elif normalization == "minmax":
            min_val = np.min(sink_vector)
            max_val = np.max(sink_vector)
            if max_val > min_val:
                sink_vector = (sink_vector - min_val) / (max_val - min_val)
            else:
                sink_vector[:] = 0  # all values are the same

        else:
            raise ValueError(f"Unknown normalization mode: {normalization}")

        return sink_vector

    def _get_decreasing_count(self, M: np.ndarray) -> np.ndarray:
        M_masked = np.copy(M)
        M_masked[self.weights == 0] = np.inf
        np.fill_diagonal(M_masked, np.inf)
        dec_i = np.argmin(M_masked, axis=1)
        unique, count = np.unique(dec_i, return_counts=True)
        return unique, count

def run_length_encoding(arr):
    """
    Computes the run-length encoding of a 1D array and estimates storage cost.
    
    Parameters:
        arr (np.ndarray): Input array of integer values.
    
    Returns:
        run_values (np.ndarray): Unique values for each run.
        run_lengths (np.ndarray): Lengths of each run.
        storage_bits (dict): Estimated bit cost for original and RLE-encoded data.
    """
    arr = np.asarray(arr)
    
    # Find change points
    change_points = np.where(np.diff(arr) != 0)[0] + 1
    run_starts = np.insert(change_points, 0, 0)
    run_ends = np.append(change_points, len(arr))

    run_values = arr[run_starts]
    run_lengths = run_ends - run_starts

    # Estimate bit usage
    max_val = int(arr.max())
    max_len = int(run_lengths.max())

    vbits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
    lbits = int(np.ceil(np.log2(max_len + 1))) if max_len > 0 else 1

    rle_bits = len(run_values) * (vbits + lbits)
    original_bits = len(arr) * vbits

    storage_bits = {
        'original_bits': original_bits,
        'rle_bits': rle_bits,
        'compression_ratio': rle_bits / original_bits if original_bits else 0.0
    }

    return run_values, run_lengths, storage_bits

if __name__ == "__main__":
    from scipy.spatial import cKDTree
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
    qsteps = [24,28,32,40,48,56,64]
    V = point_cloud.V
    A = point_cloud.A
    blocks = point_cloud.get_all_blocks()
    visualizer = Visualizer()
    degree = 4
    bits_needed = dict(zip(qsteps, [0]*len(qsteps)))
    idx_dist = {q: [] for q in qsteps}
    spatial_dist = {q: [] for q in qsteps}
    for i, block in tqdm(enumerate(blocks), "Polyfit per block"):
        block._init_data(V, A)

        # Use AttributeGraph just to get access to graph.S
        graph = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.7)
        graph._init_data(block.Vblock, block.Ablock)
        sgraph = StructuralGraph(block.id)
        sgraph._init_data(block.Vblock, block.Ablock)
        visualizer(graph, block)
        visualizer.visualize_base(graph.S, title="Sink Vector Projection")
        visualizer.add_selected_nodes()        
        plt.show()
        visualizer.visualize_block()
        visualizer.add_selected_nodes()        
        Vblock = block.Vblock  # shape: (N, D)
        V_min = Vblock.min(axis=0)
        V_max = Vblock.max(axis=0)
        Vblock_norm = (Vblock - V_min) / (V_max - V_min + 1e-8)  # avoid division by zero
        selected_idx = graph.selected_nodes.flatten()
        if len(selected_idx)>0:
            Vselected = Vblock_norm[selected_idx]
            tree = cKDTree(Vselected)   # Build a KD-tree on sink points
            _, labels = tree.query(Vblock_norm)           # Assign each point to nearest sink node
            labels_idx = np.unique(labels)
            for label_idx in labels_idx:
                mask = labels == label_idx
                cluster_mean = np.mean(Vblock[mask], axis=0)
                sink_pos = Vblock[selected_idx[label_idx]]  # original sink position
                dist = np.linalg.norm(cluster_mean - sink_pos)

            print(f"Cluster {label_idx}:")
            print(f" - Mean position: {cluster_mean}")
            print(f" - Sink node position: {sink_pos}")
            print(f" - Distance: {dist:.4f}")
        GFT_computer = GFT()
        _, aCoeffs = GFT_computer(graph, block)
        _, nCoeffs = GFT_computer(sgraph, block)
        for qstep in qsteps:
            if np.count_nonzero(np.round(aCoeffs/qstep) ) < np.count_nonzero(np.round(nCoeffs/qstep)):
                bits_needed[qstep] += len(selected_idx)*np.ceil(np.log2(len(labels)))
                idx_dist[qstep].extend(selected_idx.tolist()) 
                if Vselected is not None: 
                    spatial_dist[qstep].append(Vselected)
        visualizer.visualize_base(labels, title = "KDTree Clustering")
        values, lengths, storage = run_length_encoding(labels)
        
        Ablock = block.Ablock
        visualizer.visualize_base(Ablock[:,0], title = "Y Channel Projection")
        visualizer.add_selected_nodes()
        print("Run values:", values)
        print("Run lengths:", lengths)
        print("Storage estimate:", storage)
        print("Send points estimate:", len(selected_idx)*np.ceil(np.log2(len(labels))))

        visualizer.add_selected_nodes()
        print(f"Selected points geomtric mean and std: {np.mean(Vselected, axis=0), np.std(Vselected, axis=0)}")
        visualizer.visualize_block()
        visualizer.display()
        # ordered_idx = np.argsort(graph.S)[::-1]
        # print(f"Ordered_idx {ordered_idx} \n Ordered value {graph.S[ordered_idx]} \n Ordered positions {block.Vblock[ordered_idx]} vs Unordered positions: {block.Vblock} ")
        # Prepare data
        # x = np.arange(len(graph.S))
        # y = np.array(graph.S)

        # If y is multidimensional, pick the first attribute
        # if y.ndim > 1:
        #     y = y[:, 0]

        # i0 = np.argwhere(y>0)
        # # Fit polynomial of degree 2
        # fit = np.polyfit(x[i0][:,0], y[i0][:,0], degree)
        # p = np.poly1d(fit)
        # y_fit = p(x)

        # # Plot
        # plt.figure(figsize=(8, 4))
        # plt.plot(x, y, 'bo', label='Original Data')
        # plt.plot(x, y_fit, 'r-', label=f'Polyfit (degree {degree})')
        # plt.title(f'Polynomial Fit of graph.S for Block {block.id}')
        # plt.xlabel('Node Index')
        # plt.ylabel('Attribute Value')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        graph._del_data()
        block._del_data()
    print(bits_needed)
    # Plot distribution of selected indices for each quantization step
    def plot_selected_idx_distribution(idx_dist):
        plt.figure(figsize=(12, 6))
        for qstep, indices in idx_dist.items():
            if len(indices) > 0:
                plt.hist(indices, bins=50, alpha=0.5, label=f"q={qstep}")
        plt.title("Distribution of Selected Indices per Quantization Step")
        plt.xlabel("Selected Node Index")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 🔔 Call the plot function after processing all blocks
    plot_selected_idx_distribution(idx_dist)

    # After the main loop (which fills spatial_dist[qstep] with Vselected_norm), call:

    def plot_histograms_by_axis(spatial_dist):
        axes = ['X', 'Y', 'Z']
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        for i, axis in enumerate(axes):
            for qstep, coords_list in spatial_dist.items():
                if coords_list:
                    all_coords = np.vstack(coords_list)
                    axs[i].hist(all_coords[:, i], bins=30, alpha=0.4, label=f'q={qstep}')
            
            axs[i].set_title(f'{axis}-axis Distribution (Normalized)')
            axs[i].set_ylabel("Frequency")
            axs[i].grid(True)

        axs[-1].set_xlabel("Normalized Position [0, 1]")
        axs[0].legend()
        plt.tight_layout()
        plt.show()

    # 🔔 Call it after the loop
    plot_histograms_by_axis(spatial_dist)
