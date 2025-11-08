from typing import Dict, Tuple, List
from .transforms import CoeffsContainer
from .graph import GraphMetadata
from .clusterer import Codebook
from dataclasses import dataclass
from line_profiler import profile
import numpy as np
import logging

logging.basicConfig(filename="logs/decider.log",
                    filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class RDO_Decision:
    mode: str
    cost: float
    rates: List[float]
    distorsions: List[float]
    selected_coeffs: np.ndarray
    selected_graph_metadata: GraphMetadata

    def get_label_from_luminance(self, codebook: Codebook):
        luminance_centroid = self.selected_graph_metadata.luminance_centroid
        return codebook.find_centroid_label(luminance_centroid)


class Decider:
    def __init__(self, mode: str, lagrange_proportional: float):
        self.mode = mode
        self.lagrange_proportional = lagrange_proportional
        logger.info(f"Decider initialized in mode: {self.mode}.")

    @profile
    def __call__(self, q_step: int, coeffs_container: CoeffsContainer) -> RDO_Decision:
        self._set_vars(q_step)
        logger.info(
            f"Starting RDO with q_step={q_step} and lambda={self.lagrange_mult:.4f}.")
        return self._RDO(coeffs_container)
    
    def _set_vars(self, q_step: int):
        self.q_step = q_step
        self.lagrange_mult = self.get_lagrange_mult(q_step)

    def get_lagrange_mult(self,q_step):
        return self.lagrange_proportional * q_step**2
        
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
        return norm_value

    def _zeroNorm(self, Coeffs_quant: np.ndarray):
        if self.mode in ["1", "2"]:
            return np.count_nonzero(Coeffs_quant, axis=1).sum()
        return np.count_nonzero(Coeffs_quant, axis=0).sum()

    def _RDcost(self, Coeffs: np.ndarray):
        """Rate-Distortion cost."""
        if self.mode == "0":
            obj_coeffs = Coeffs[:, 0] # Luminansce channel only
        elif self.mode in ["1", "2"]:
            obj_coeffs = Coeffs
        else:
            logger.error(
                f"Invalid mode: {self.mode}. Skipping cost calculation.")
            return float('inf')

        obj_coeffs_quant = self._quantize(obj_coeffs)
        qerror = self._qError(obj_coeffs, obj_coeffs_quant)
        sparsity = self._zeroNorm(obj_coeffs_quant)

        rd_cost = (self.lagrange_mult * sparsity) + qerror
        logger.debug(
            f"Calculated RD cost. Quantization Error: {qerror:.4f}, Sparsity: {sparsity}, Total Cost: {rd_cost:.4f}.")
        return rd_cost, sparsity, qerror

    @profile
    def _RDO(self, coeffs_container: CoeffsContainer) -> RDO_Decision:
        """
        Perform Rate-Distortion Optimization (RDO) to select the best coefficient-graph pair.
        """
        coeffs_list = coeffs_container.coeffs
        metadata_list = coeffs_container.graphs
        coeff_id_pairs = list(zip(coeffs_list, metadata_list))

        min_cost = float('inf')
        selected_coeff = None
        selected_metadata = None
        struct_coeff = None
        distorsions = []
        rates = []

        logger.info(
            f"Evaluating {len(coeff_id_pairs)} candidate coefficient-graph pairs.")
        for i, (coeff, graph_obj) in enumerate(coeff_id_pairs):
            res, sparsity, qerror = self._RDcost(coeff)
            logger.info(
                f"Candidate {i+1}/{len(coeff_id_pairs)} (Graph ID: {graph_obj.graph_id}) has a total cost of {res:.4f}.")
            rates.append(sparsity)
            distorsions.append(qerror)

            if res < min_cost:
                min_cost = res
                selected_coeff = coeff
                selected_metadata = graph_obj

            if graph_obj.graph_type == "Structural":
                struct_coeff = coeff

        if selected_coeff is None:
            err_msg = "No valid graph was found. RDO failed."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if self.mode == "0":
            if struct_coeff is not None:
                selected_coeff[:, 1:] = struct_coeff[:, 1:]
                logger.debug(
                    "Applied structural coefficients for U and V channels.")
            else:
                warn_err = "Structural graph was not a candidate. Mode 0 may be compromised."
                logger.warning(warn_err)
                raise ValueError(warn_err)

        logger.info(
            f"RDO finished. Selected graph ID: {selected_metadata.graph_id} with a minimum cost of {min_cost:.4f}.")

        return RDO_Decision(mode=self.mode,
                            cost=min_cost,
                            rates=rates,
                            distorsions=distorsions,
                            selected_coeffs=selected_coeff,
                            selected_graph_metadata=selected_metadata)
