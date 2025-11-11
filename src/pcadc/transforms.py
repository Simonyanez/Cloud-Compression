# Other imports
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile
# from graph.create import *
from .graph import *
from .blocks import *
# from .visualization import *
from .factories import *
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power, eigh
# from scipy.linalg import eigh
import logging
import os

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "transforms.log")

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)


@dataclass
class CoeffsContainer:
    block: Block
    graphs: List[GraphMetadata]
    coeffs: List[np.ndarray]

@dataclass
# NOTE: This is a Strategy Pattern
class GFTProcessorStrategy(ABC):
    @abstractmethod
    def compute(self, block: Block, graph: StructuralGraph | AttributeGraph, Q: Optional[np.ndarray] = None):
        pass


class ConnectedGFTProcessor(GFTProcessorStrategy):
    # First strategy for fully connected graph in block
    def compute(self, block: Block | AuxiliaryBlock, graph: StructuralGraph | AttributeGraph | GraphBase, Q: Optional[np.ndarray] = None):
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

    def _compute_laplacian(self, graph: StructuralGraph | AttributeGraph | GraphBase, Qm: np.ndarray) -> np.ndarray:
        A, _ = graph.get_data()  # Adjacency matrix
        D = np.diag(np.sum(A, axis=0))  # Degree matrix
        C = np.zeros(A.shape)
        if isinstance(graph, AttributeGraph):
            C = np.diag(np.diag(A))  # Self-loops matrix
        logger.debug(f"[ConnectedGFT] D,A,C shapes: {D.shape} {A.shape} {C.shape}")
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

            subblock, subgraph = self._build_subobjects(
                block, graph, sub_idxs, comp_id)

            GFT_sub, _ = ConnectedGFTProcessor().compute(block=subblock, graph=subgraph)
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

        meanblock, meangraph = self._build_meanobjects(
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
    def _build_subobjects(block: Block, graph: StructuralGraph | AttributeGraph,
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
        if num_components == 1:
            logger.debug("Found disconnected graph")
            GFT_mat, Coeffs = ConnectedGFTProcessor().compute(block, graph)
        else:
            GFT_mat, Coeffs = DisconnectedGFTProcessor().compute(
                block, graph, int(num_components), labels)
        return GFT_mat, Coeffs

    @staticmethod
    def _check_connected(graph: StructuralGraph | AttributeGraph) -> tuple[np.ndarray, np.ndarray]:
        Adj = graph.weights
        Adj_sparse = csr_matrix(Adj)
        num_components, labels = connected_components(
            Adj_sparse, directed=False, return_labels=True)
        return num_components, labels


if __name__ == "__main__":
    pass
