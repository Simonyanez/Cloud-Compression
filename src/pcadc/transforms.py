# Other imports
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile
# from graph.create import *
from .graph import *
from .blocks import *
from .visualization import *
from .factories import *
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power, eigh
# from scipy.linalg import eigh
import logging
logging.basicConfig(filename="logs/graph.log",
                    filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)


# NOTE: This is a Strategy Pattern

class GFTProcessorStrategy(ABC):
    @abstractmethod
    def compute(self, block: Block, graph: StructuralGraph | AttributeGraph, Q: Optional[np.ndarray] = None):
        pass


class ConnectedGFTProcessor(GFTProcessorStrategy):
    # First strategy for fully connected graph in block
    def compute(self, block: Block, graph: StructuralGraph | AttributeGraph, Q: Optional[np.ndarray] = None):
        Vblock, Ablock = block.get_data()
        logger.info(
            f"[ConnectedGFT] Computing GFT for block with {Vblock.shape[0]} nodes")

        # One point solution
        if Vblock.shape[0] == 1:
            logger.debug("[ConnectedGFT] Single-node block detected")
            GFT_mat = np.array([[1.0]])
            Gfreq = np.array([0.0])

        # Multiple points solution
        else:
            N = Ablock.shape[0]
            logger.debug(f"[ConnectedGFT] Preparing Laplacian for N={N}")
            Qm = self._prepare_Q(N, Q)
            L_q = self._compute_laplacian(graph, Qm)
            GFT_mat, G_freqs = self._compute_GFT(L_q)
            logger.debug(
                f"[ConnectedGFT] Computed GFT matrix shape: {GFT_mat.shape}")

        Coeffs = self._compute_coeffs(GFT_mat, Ablock)
        logger.debug(f"[ConnectedGFT] Coeffs shape: {Coeffs.shape}")
        return GFT_mat, Coeffs

    def _prepare_Q(self, N: int, Q: Optional[np.ndarray]) -> np.ndarray:
        return np.identity(N) if Q is None else fractional_matrix_power(Q, -0.5)

    def _compute_laplacian(self, graph: StructuralGraph | AttributeGraph, Qm: np.ndarray) -> np.ndarray:
        A, _ = graph.get_data()  # Adjacency matrix
        D = np.diag(np.sum(A, axis=0))  # Degree matrix
        C = np.zeros(A.shape)
        if isinstance(graph, AttributeGraph):
            C = np.diag(np.diag(A))  # Self-loops matrix
        L = D - A + C
        L_q = Qm @ L @ Qm
        logger.debug(f"[ConnectedGFT] Laplacian shape: {L_q.shape}")
        return L_q

    def _compute_GFT(self, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = np.linalg.eigh(L)
        eigvals_idxsorted = np.argsort(eigvals)
        GFT_matrix = eigvecs[:, eigvals_idxsorted]
        for i in range(GFT_matrix.shape[0]):
            if GFT_matrix[i, 0] < 0:
                GFT_matrix[i, :] *= -1
        Gfreq = eigvals[eigvals_idxsorted]
        logger.debug(
            f"[ConnectedGFT] Eigenvalues range: [{Gfreq.min()}, {Gfreq.max()}]")
        return GFT_matrix, Gfreq

    def _compute_coeffs(self, GFT_mat: np.ndarray, Ablock: np.ndarray):
        return GFT_mat.T @ Ablock


class DisconnectedGFTProcessor(GFTProcessorStrategy):
    def compute(self, block: Block, graph: StructuralGraph | AttributeGraph,
                num_components: int, labels: np.ndarray,
                Q: Optional[np.ndarray] = None):

        Vblock, Ablock = block.get_data()
        logger.info(
            f"[DisconnectedGFT] Starting with {num_components} components and {Vblock.shape[0]} nodes")

        Q_norm, Vmean, isDC = self._init_arrays(num_components, labels)
        U_parts = []
        posDC = 0

        # --- process each disconnected component ---
        for comp_id in range(num_components):
            sub_idxs = np.where(labels == comp_id)[0]
            logger.debug(
                f"[DisconnectedGFT] Component {comp_id}: {len(sub_idxs)} nodes")

            subgraph, subblock = self._build_subobjects(
                block, graph, sub_idxs, comp_id)

            GFT_sub, _ = ConnectedGFTProcessor().compute(subblock, subgraph)
            logger.debug(
                f"[DisconnectedGFT] Sub-GFT matrix shape: {GFT_sub.shape}")

            U_parts.append(self._fill_disconnected_transform(
                graph, sub_idxs, GFT_sub))

            # mark low-frequency indexes (DC coeff)
            isDC[posDC] = True
            posDC += len(sub_idxs)

            Vmean[comp_id, :] = np.mean(Vblock[sub_idxs, :], axis=0)
            Q_norm[comp_id, comp_id] = len(sub_idxs)

        # --- concatenate U ---
        U = np.concatenate(U_parts, axis=1)
        logger.debug(
            f"[DisconnectedGFT] Final concatenated U shape: {U.shape}")

        return self._unique_dc_processing(block, Ablock, Vmean, Q_norm, U, isDC, num_components)

    # ---------------- private helpers ----------------

    def _init_arrays(self, num_components: int, labels: np.ndarray):
        N = labels.shape[0]
        return (np.zeros((num_components, num_components)),
                np.zeros((num_components, 3)),
                np.zeros(N, dtype=bool))

    def _unique_dc_processing(self, block: Block, Ablock: np.ndarray,
                              Vmean: np.ndarray, Q_norm: np.ndarray,
                              U: np.ndarray, isDC: np.ndarray,
                              num_components: int):

        Coeffs = U.T @ Ablock
        logger.debug(f"[DisconnectedGFT] Raw coeffs shape: {Coeffs.shape}")

        Coeffs_low = Coeffs[isDC, :]
        Coeffs_high = Coeffs[~isDC, :]
        logger.debug(
            f"[DisconnectedGFT] Low coeffs: {Coeffs_low.shape}, High coeffs: {Coeffs_high.shape}")

        meangraph, meanblock = self._build_meanobjects(
            block, Vmean, num_components)

        GFT_mean, _ = ConnectedGFTProcessor().compute(meanblock, meangraph, Q_norm)
        Coeffs_low_fixed = GFT_mean.T @ Coeffs_low
        logger.debug(
            f"[DisconnectedGFT] Corrected low coeffs shape: {Coeffs_low_fixed.shape}")

        Coeffs_fix = np.concatenate([Coeffs_low_fixed, Coeffs_high])
        logger.info(
            f"[DisconnectedGFT] Finished. Final coeffs shape: {Coeffs_fix.shape}")
        return U, Coeffs_fix

    @staticmethod
    def _build_subobjects(block: Block, graph: Graph,
                          subgraph_indexes: np.ndarray, component: int):
        factory = SubGraphCreator(parent_block=block,
                                  parent_graph=graph,
                                  sub_idxs=subgraph_indexes,
                                  task=f"Disconnected Component N° {component}")
        return factory.factory_method()

    @staticmethod
    def _build_meanobjects(block: Block, components_Vmean: np.ndarray,
                           num_components: int):
        factory = MeanGraphFactory(parent_block=block,
                                   components_Vmean=components_Vmean,
                                   num_components=num_components)
        return factory.factory_method()

    @staticmethod
    def _fill_disconnected_transform(graph: StructuralGraph | AttributeGraph,
                                     subgraph_indexes: np.ndarray,
                                     GFT_matrix: np.ndarray):
        num_nodes = graph.weights.shape[0]
        Utmp = np.zeros((num_nodes, len(subgraph_indexes)))
        Utmp[subgraph_indexes, :] = GFT_matrix
        return Utmp


class GFTStrategyWraper:
    def __call__(self, block, graph):
        num_components, labels = self._check_connected(graph)
        if n_components > 1:
            logger.debug("Found disconnected graph")
            GFT_mat, Coeffs = ConnectedGFTProcessor().compute(block, graph)
        else:
            GFT_mat, Coeffs = DisconnectedGFTProcessor().compute(
                block, graph, num_components, labels)
        return GFT_mat, Coeffs

    @staticmethod
    def _check_connected(self, graph: StructuralGraph | AttributeGraph) -> tuple[np.ndarray, np.ndarray]:
        Adj = graph.weights
        Adj_sparse = csr_matrix(Adj)
        num_components, labels = connected_components(
            Adj_sparse, directed=False, return_labels=True)
        return num_components, labels


# class GFT():
#     # FIXME: Disconnected components not working correctly
#     def __init__(self):
#         self.visualizer = Visualizer()
#         # TODO: Give parameters
#         pass
#
#     def __call__(self, graph: Graph, block: Block, Q: Optional[np.ndarray] = None, sl_flag=True) -> tuple[np.ndarray, np.ndarray]:
#         self.graph = graph
#         self.block = block
#         self.sl_flag = sl_flag
#         GFT_matrix, Coeffs = self._exec(Q)
#         return GFT_matrix, Coeffs
#
#     def _exec(self, Q):
#         n_components, labels = self._check_connected()
#         if n_components > 1:
#             logger.debug("Found disconnected graph")
#             GFT_matrix, Coeffs = self._process_disconnected(
#                 n_components, labels)
#         else:
#             GFT_matrix, Coeffs = self._process_connected(Q)
#         return GFT_matrix, Coeffs
#
#     def _check_connected(self) -> tuple[np.ndarray, np.ndarray]:
#         Adj = self.graph.weights
#         Adj_sparse = csr_matrix(Adj)
#         num_components, labels = connected_components(
#             Adj_sparse, directed=False, return_labels=True)
#         return num_components, labels
#
#     @profile
#     def _process_disconnected(self, num_components, labels):
#         GFT_processor = GFT()
#         Q_norm = np.zeros((num_components, num_components))
#         N = self.graph.weights.shape[0]
#         U = None
#         Vmean = np.zeros((num_components, 3))
#         isDC = np.zeros(N, dtype=bool)
#         i = 0
#         # FIXME isn't pos and component the same?
#         for pos, component in enumerate(range(num_components)):
#             subgraph_indexes = np.where(labels == component)[0]
#             Q_norm[pos, pos] = len(subgraph_indexes)
#             subgraph, subblock = self._create_subobjects(subgraph_indexes)
#             GFT_sub, _ = GFT_processor(subgraph, subblock)
#             U = self._fill_disconnected_transform(subgraph_indexes, GFT_sub, U)
#             isDC[i] = 1
#             i += len(subgraph_indexes)
#             Vmean[component, :] = np.mean(
#                 self.block.Vblock[subgraph_indexes, :], axis=0)
#         Coeffs = U.T @ self.block.Ablock
#         Coeffs_low, Coeffs_high = Coeffs[isDC,
#                                          :], Coeffs[np.logical_not(isDC), :]
#         meangraph, meanblock = self._create_meanobjects(Vmean)
#         self.visualizer(meangraph, meanblock)
#         GFT_mean, _ = GFT_processor(
#             meangraph, meanblock, Q_norm, sl_flag=False)
#         Coeffs_low_fixed = GFT_mean.T @ Coeffs_low
#         Coeffs_fix = np.concatenate([Coeffs_low_fixed, Coeffs_high])
#         return U, Coeffs_fix
#
#     def _create_subobjects(self, subgraph_indexes) -> tuple[Graph, Block]:
#         Asubblock = self.block.Ablock[subgraph_indexes, :]
#         Vsubblock = self.block.Vblock[subgraph_indexes, :]
#         W = self.graph.weights
#         W_sub = W[subgraph_indexes, :][:, subgraph_indexes]
#         aux_tuple = (-1, -1)
#         subblock = Block((-1, -1), block_num=-1)
#         subblock._init_auxiliary(
#             Vblock=Vsubblock, Ablock=Asubblock, subidxs=subgraph_indexes)
#         subgraph = Graph(subblock.id)
#         # Creates a subgraph without connections
#         subgraph._init_data(weights=W_sub, edges=[])
#         return subgraph, subblock
#
#     def _create_meanobjects(self, Vmean: np.ndarray):
#         meanblock = Block(idxs=(-1, -1), block_num=-2)
#         # Use Vmean auxiliary for attributes only for calling. Coeffs will be useless
#         meanblock._init_auxiliary(Vblock=Vmean, Ablock=Vmean, subidxs=None)
#         meangraph = StructuralGraph(meanblock.id)
#         meangraph._init_data(Vmean, threshold=np.inf)
#         return meangraph, meanblock
#
#     def _fill_disconnected_transform(self, subgraph_indexes: np.ndarray, GFT_matrix: np.ndarray, U: np.ndarray):
#         """
#         Fill an auxiliary
#         """
#         num_nodes = self.graph.weights.shape[0]
#         Utmp = np.zeros((num_nodes, len(subgraph_indexes)))
#         Utmp[subgraph_indexes, :] = GFT_matrix
#         if U is None:
#             return Utmp
#         return np.concatenate([U, Utmp], axis=1)
#
#     # def reorder_coeffs_by_vmean(self, Vmean: np.ndarray, Coeffs_low_fixed: np.ndarray) -> np.ndarray:
#     #     """
#     #     Reorder the rows of Coeffs_low_fixed to match the spatial order in Vmean.
#     #     This fixes random flips/swaps from spectral decomposition.
#     #     """
#     #     from sklearn.preprocessing import normalize
#     #     from scipy.optimize import linear_sum_assignment
#     #
#     #     Vmean_norm = normalize(Vmean)
#     #     coeffs_norm = normalize(Coeffs_low_fixed)
#     #
#     #     # Compute cosine similarity
#     #     similarity = Vmean_norm @ coeffs_norm.T  # shape: (num_components, num_components)
#     #     cost = -np.abs(similarity)
#     #     row_ind, col_ind = linear_sum_assignment(cost)
#     #
#     #     # Reorder rows
#     #     Coeffs_low_sorted = Coeffs_low_fixed[col_ind]
#     #     return Coeffs_low_sorted
#
#     @profile
#     def _process_connected(self, Q: Optional[np.ndarray] = None):
#         """
#         Process a connected block to compute the GFT matrix and coefficients.
#
#         Args:
#             Q (Optional[np.ndarray]): The weighting matrix. Defaults to the identity matrix.
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: The GFT matrix and the coefficients.
#         """
#         if Q is None:
#             n = self.block.Ablock.shape[0]
#             Q = np.identity(n)
#             Qm = Q
#
#         else:
#             Qm = fractional_matrix_power(Q, -0.5)
#         # Handle 1-point blocks
#         if Q.shape[0] == 1:
#             GFT_matrix = np.array([[1.0]])
#             Coeffs = self.block.Ablock
#             return GFT_matrix, Coeffs
#         try:
#             L = self._get_laplacian(Qm)
#             GFT_matrix, _ = self._compute_GFT(L)
#             A = self.block.Ablock
#             Coeffs = GFT_matrix.T @ A
#             return GFT_matrix, Coeffs
#         except:
#             logger.debug(f"Q is {Q.shape}  --> {Q}")
#
#     def _get_laplacian(self, Qm: np.ndarray) -> np.ndarray:
#         """
#         Compute the normalized Laplacian matrix.
#
#         Args:
#             Qm (np.ndarray): The square root of the inverse weighting matrix.
#
#         Returns:
#             np.ndarray: The normalized Laplacian matrix.
#         """
#         A = self.graph.weights  # Adjacency matrix
#         D = np.diag(np.sum(A, axis=0))  # Degree matrix
#         C = np.diag(np.diag(A))  # Self-loops matrix
#         if not self.sl_flag:
#             C = np.zeros(A.shape)
#         L = D - A + C
#         L_q = Qm @ L @ Qm
#         return L_q
#
#     @profile
#     def _compute_GFT(self, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Compute the Graph Fourier Transform (GFT) matrix and frequencies.
#
#         Args:
#             L (np.ndarray): The Laplacian matrix.
#
#         Returns:
#             tuple[np.ndarray, np.ndarray]: The GFT matrix and the frequencies.
#         """
#         if L.shape[0] == 1:
#             # Handle 1-point blocks
#             GFT_matrix = np.array([[1.0]])
#             Gfreq = np.array([0.0])
#         else:
#             # Compute eigenvalues and eigenvectors for larger blocks
#             eigvals, eigvecs = np.linalg.eigh(L)
#             eigvals_idxsorted = np.argsort(eigvals)  # Changed from abs value
#             GFT_matrix = eigvecs[:, eigvals_idxsorted]
#             # Ensure the first eigenvector is positive
#             for i in range(GFT_matrix.shape[0]):
#                 if GFT_matrix[i, 0] < 0:
#                     GFT_matrix[i, :] = GFT_matrix[i, :] * (-1)
#             Gfreq = eigvals[eigvals_idxsorted]
#         return GFT_matrix, Gfreq
#

if __name__ == "__main__":
    pass
