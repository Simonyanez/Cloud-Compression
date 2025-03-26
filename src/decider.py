from typing import Dict
from uuid import *
import numpy as np
from scipy.optimize import minimize
import logging
logging.basicConfig(filename="logs/decider.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Decider:
    def __init__(self):
        pass

    def __call__(self, q_step: int, coeff_dict: Dict[str, np.ndarray], r=0.85):
        self.lagrange_mult = r * q_step**2   
        self.q_step = q_step
        Coeffs_list = list(coeff_dict.values())
        graph_ids = list(coeff_dict.keys())
        selected_graph_id, selected_coeffs = self._RDO(Coeffs_list, graph_ids)
        return selected_graph_id, selected_coeffs

    def _quantize(self, Y_coeffs):
        Y_coeffs_quant = np.round(Y_coeffs / self.q_step)
        return Y_coeffs_quant

    def _qError(self, Y_coeffs: np.ndarray, Y_coeffs_quant: np.ndarray):
        N = Y_coeffs.shape[0]
        Y_coeff_dequant = Y_coeffs_quant * self.q_step
        norm_value = np.linalg.norm(Y_coeffs - Y_coeff_dequant)
        psnr_Y = -10 * np.log10((norm_value**2) / (N * 255**2))
        return psnr_Y

    def _RDcost(self, Y_coeffs: np.ndarray):
        """
        Rate-Distorsion cost
        """
        Y_coeffs_quant = self._quantize(Y_coeffs)
        qerror = self._qError(Y_coeffs, Y_coeffs_quant)
        sparsity = self._zeroNorm(Y_coeffs_quant)
        logger.debug(f"Quantization Error {qerror} - Zero norm (sparsity) {sparsity} - RDO Cost: {qerror + self.lagrange_mult * sparsity}")
        return qerror + self.lagrange_mult * sparsity

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
            res = self._RDcost(coeff[:,0])
            # Check if this is the best result so far
            logger.debug(f"Current graph id {graph_id}")
            if res < min_cost:
                min_cost = res
                selected_coeff = coeff # Reshape back to original shape
                selected_graph_id = graph_id

        logger.debug(f"Selected graph id {selected_graph_id}")
        return selected_graph_id, selected_coeff

    def _zeroNorm(self, Coeffs_quant: np.ndarray):
        zeroNorm = np.linalg.norm(Coeffs_quant, 0)
        return zeroNorm