from typing import Dict
from uuid import *
from line_profiler import profile
import numpy as np
from scipy.optimize import minimize
import logging
logging.basicConfig(filename="logs/decider.log",
                    filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Decider:
    def __init__(self, mode: str):
        self.mode = mode

    @profile
    def __call__(self, q_step: int, coeff_dict: Dict[str, np.ndarray], r=0.85):
        self.lagrange_mult = r * q_step**2
        self.q_step = q_step
        Coeffs_list = list(coeff_dict.values())
        graph_ids = list(coeff_dict.keys())
        selected_graph_id, selected_coeffs, rd_cost = self._RDO(
            Coeffs_list, graph_ids)
        return selected_graph_id, selected_coeffs, rd_cost

    def _quantize(self, Coeffs):
        Coeffs_quant = np.round(Coeffs / self.q_step)
        return Coeffs_quant

    def _qError(self, Coeffs: np.ndarray, Coeffs_quant: np.ndarray):
        N = Coeffs.shape[0]
        Coeffs_dequant = Coeffs_quant * self.q_step
        norm_value = np.linalg.norm(Coeffs - Coeffs_dequant)
        if self.mode == "2":
            proportion = np.array([0.695, 0.130, 0.175])
            norm_value = norm_value * proportion
        if self.mode in ["1", "2"]:
            norm_value = np.sum(norm_value, axis=1)
        # psnr_Y = -10 * np.log10((norm_value**2) / (N * 255**2))
        return norm_value

    def _zeroNorm(self, Coeffs_quant: np.ndarray):
        # Count non-zero values per row (i.e., per coefficient vector)
        if self.mode in ["1", "2"]:
            return np.count_nonzero(Coeffs_quant, axis=1).sum()
        return np.count_nonzero(Coeffs_quant, axis=0).sum()

    def _RDcost(self, Coeffs: np.ndarray):
        """
        Rate-Distorsion cost
        """
        # TODO: Better mode naming
        if self.mode == "0":
            obj_coeffs = Coeffs[:, 0]
            pass
        if self.mode in ["1", "2"]:
            obj_coeffs = Coeffs
            pass

        obj_coeffs_quant = self._quantize(obj_coeffs)
        qerror = self._qError(obj_coeffs, obj_coeffs_quant)
        sparsity = self._zeroNorm(obj_coeffs_quant)
        logger.debug(f"Quality error {qerror} - Sparsity {sparsity}")
        # OG: qerror + self.lagrange_mult * sparsity
        return (self.lagrange_mult * sparsity) + qerror

    @profile
    def _RDO(self, Coeffs_list: list[np.ndarray], graph_ids: list[str]):
        """
        Perform Rate-Distortion Optimization (RDO) to select the best coefficient-graph pair.

        Args:
            Coeffs_list (list[np.ndarray]): List of coefficient arrays (each of shape Nx3).
            graph_ids (list[str]): List of graph IDs corresponding to the coefficient arrays.

        Returns:
            tuple: Selected graph ID and selected coefficient array.
        """
        # Pair each coefficient with its graph ID
        coeff_id_pairs = list(zip(Coeffs_list, graph_ids))

        # Initialize variables to store the best result
        min_cost = float('inf')
        selected_coeff = None
        selected_graph_id = None

        # Iterate over each coefficient-graph pair
        for coeff, graph_id in coeff_id_pairs:
            # Perform minimization for the current coefficient
            res = self._RDcost(coeff)
            # Check if this is the best result so far
            if res < min_cost:
                min_cost = res
                selected_coeff = coeff  # Reshape back to original shape
                selected_graph_id = graph_id

        if self.mode == "0":
            selected_coeff[:, 1:] = struct_coeff[:, 1:]

        logger.debug(f"Selected graph id {selected_graph_id}")
        return selected_graph_id, selected_coeff, min_cost
