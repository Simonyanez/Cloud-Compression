from typing import Tuple
from line_profiler import profile
from utils.encode_rlgr import *
import logging
from dataclasses import dataclass
import numpy as np
import constriction
import os

# A more robust setup would use a handler.
if os.path.exists("logs/encoder.log"):
    os.remove("logs/encoder.log")

import logging
import os

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "encoder.log")

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)


@dataclass
class EncodeResult:
    q_step: int
    PSNR: float
    bpv: float
    bitstream_size: int
    overhead_bitstream_size: int
    overhead_bpv: float

    @property
    def total_bpv(self):
        return self.bpv + self.overhead_bpv

    @property
    def total_bitstream(self):
        return self.bitstream_size + self.overhead_bitstream_size


class Encoder:
    def __init__(self):
        logger.info("Encoder object initialized.")
        pass

    @profile
    def __call__(self, Coeffs, qstep, indexes, assignation):
        logger.info(f"Starting encoding process with quantization step: {qstep}.")
        self.indexes = indexes
        self.Coeffs = Coeffs
        self.qstep = qstep
        
        self._quantize()
        PSNR = self.get_PSNR()
        self._sort_coeffs()
        
        bsize, bpv = self.get_bpv()
        overhead_bsize, overhead_bpv = self.get_overhead_bpv(assignation)
        
        # Log the final results
        logger.info(f"Encoding completed. PSNR: {PSNR:.2f} dB.")
        logger.info(f"Bitstream size (BPV): {bpv:.4f} bits/value.")
        logger.info(f"Overhead bitstream size (BPV): {overhead_bpv:.4f} bits/value.")
        logger.info(f"Total bits per value: {bpv + overhead_bpv:.4f} bpv.")

        result_obj = EncodeResult(q_step=qstep,
                                  PSNR=PSNR,
                                  bpv=bpv,
                                  bitstream_size=bsize,
                                  overhead_bitstream_size=overhead_bsize,
                                  overhead_bpv = overhead_bpv)
        return result_obj

    def _quantize(self):
        self.Coeffs_quant = np.round(self.Coeffs/self.qstep)
        logger.info(f"Quantization successful for coefficients with qstep={self.qstep}.")
        logger.debug(
            f"Quantized coefficients min/max: {np.min(self.Coeffs_quant), np.max(self.Coeffs_quant)}. Shape: {self.Coeffs_quant.shape}")

    def _sort_coeffs(self):
        N = self.Coeffs_quant[:, 0].shape[0]
        mask_lo = np.zeros(N, dtype=bool)
        for start_idx, _ in self.indexes:
            mask_lo[start_idx] = True
        mask_hi = np.logical_not(mask_lo)

        Coeffs_quant_lo = self.Coeffs_quant[mask_lo, :]
        Coeffs_quant_hi = self.Coeffs_quant[mask_hi, :]
        
        self.Coeffs_quant = np.concatenate((Coeffs_quant_lo, Coeffs_quant_hi))
        logger.info("Coefficients sorted into low and high frequency components.")

    def _RLGR(self):
        logger.info("Starting Run-Length Golomb-Rice encoding for Y, U, V channels.")
        try:
            numbits_Y = encode_rlgr(
                self.Coeffs_quant[:, 0], os.path.join('res', 'bitstream_Y.bin'))
            numbits_U = encode_rlgr(
                self.Coeffs_quant[:, 1], os.path.join('res', 'bitstream_U.bin'))
            numbits_V = encode_rlgr(
                self.Coeffs_quant[:, 2], os.path.join('res', 'bitstream_V.bin'))
            
            bs_size = numbits_Y + numbits_U + numbits_V
            logger.info(f"RLGR encoding finished. Bitstream size: {bs_size} bits.")
            return bs_size
        except Exception as e:
            logger.error(f"Error during RLGR encoding: {e}")
            return 0

    def get_PSNR(self) -> float:
        N = self.Coeffs[:, 0].shape[0]
        Coeff_dequant = self.Coeffs_quant*self.qstep
        
        # Original debug log is useful, but should be at a lower level
        logger.debug(f"Original vs. dequantized coefficients. Min/Max Original: ({np.min(self.Coeffs):.2f}, {np.max(self.Coeffs):.2f}). Min/Max Dequantized: ({np.min(Coeff_dequant):.2f}, {np.max(Coeff_dequant):.2f}).")

        norm_value = np.linalg.norm(self.Coeffs[:, 0] - Coeff_dequant[:, 0])
        if N > 0 and 255 > 0:
            psnr_Y = -10 * np.log10((norm_value ** 2) / (N * 255 ** 2))
            logger.info(f"Calculated PSNR for Y channel: {psnr_Y:.2f} dB.")
            return psnr_Y
        else:
            logger.warning("Cannot calculate PSNR: number of coefficients is zero or norm value is invalid.")
            return 0.0

    def get_bpv(self) -> Tuple[int, float]:
        bs_size = self._RLGR()
        bpv = self._as_bpv(bs_size)
        logger.info(f"Calculated BPV: {bpv:.4f} bits/value.")
        return bs_size, bpv


    def get_overhead_bpv(self, assignation: np.ndarray) -> Tuple[int, float]:
        if assignation.size == 0:
            logger.warning("Assignation array is empty, overhead size is 0.")
            return 0, 0.0

        logger.info("Starting arithmetic coding overhead computation.")

        # 1) Build empirical probability model
        unique_labels, counts = np.unique(assignation, return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()

        logger.debug(f"Unique labels: {unique_labels}")
        logger.debug(f"Counts: {counts}")
        logger.debug(f"Probabilities: {probabilities}")

        # Map labels to contiguous indices [0, ..., K-1]
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        indexed_assignation = np.array(
            [label_to_index[x] for x in assignation],
            dtype=np.int32
        )

        logger.debug(f"Label → index mapping: {label_to_index}")

        # Create categorical entropy model
        model = constriction.stream.model.Categorical(
            probabilities.astype(np.float32),
            perfect=False
        )

        # Encode using RangeEncoder
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(indexed_assignation, model)
        compressed = encoder.get_compressed()

        # Each word is uint32 → 32 bits
        bs_size = len(compressed) * 32

        logger.info(f"Arithmetic-coded bitstream size: {bs_size} bits.")
        logger.debug(f"Compressed words: {compressed}")

        # Compute bits per value over all voxels
        bpv = self._as_bpv(bs_size)

        logger.info(f"Arithmetic overhead BPV: {bpv:.6f} bits/value.")

        return bs_size, bpv

    def _as_bpv(self, bitstream_size: int):
        N = self.Coeffs[:, 0].shape[0]
        if N > 0:
            return bitstream_size/N
        else:
            logger.warning("Number of coefficients is zero, BPV is 0.")
            return 0.0


