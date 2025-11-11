import numpy as np
from .blocks import Block
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
import logging
import os

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "color.log")

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)


class Colourist():

    def __init__(self):
        self.__init__transformations()
        pass

    def __init__transformations(self):
        self.Q_RGBtoYUV = np.array(
            [
                [0.29899999, -0.1687, 0.5],
                [0.587, -0.3313, -0.4187],
                [0.114, 0.5, -0.0813],
                [0, 0.50196078, 0.50196078],
            ]
        )

        self.M_YUVtoRGB = np.array(
            [
                [1, 1, 1],
                [0, -0.34414, 1.772],
                [1.402, -0.71414, 0],
                [-0.703749019, 0.53121505, -0.88947451],
            ]
        )

    def _YUVtoRGB(self, A_yuv: np.ndarray, rounding: bool = True):
        A_yuv_1 = np.concatenate(
            (A_yuv / 255, np.ones((A_yuv.shape[0], 1))), axis=1)
        A_rgb = np.dot(A_yuv_1, self.M_YUVtoRGB)
        A_rgb = 255 * np.clip(A_rgb, 0, 1)
        if rounding:
            A_rgb = A_rgb.round().astype(np.uint8)

        return A_rgb

    def _RGBtoYUV(self, A_rgb: np.ndarray, rounding=False) -> np.ndarray:
        A_rgb_1 = np.concatenate(
            (A_rgb / 255, np.ones((A_rgb.shape[0], 1))), axis=1)
        A_yuv = np.dot(A_rgb_1, self.Q_RGBtoYUV)
        A_yuv = 255 * np.clip(A_yuv, 0, 1)
        if rounding:
            A_yuv = A_yuv.round().astype(np.uint8)
        return A_yuv


@dataclass
class FitResult:
    coeffs: np.ndarray
    rmse: float
    feature_names: np.ndarray

    def __str__(self) -> str:
        return (
            f"Fit Result\n"
            f"==============================\n"
            f"Coeffs: {self.coeffs}\n"
            f"RMSE: {self.rmse}\n"
            f"Feature names: {self.feature_names}\n"
        )


class FitCollection:
    def __init__(self):
        self._results = []

    def add(self, fit_result: FitResult):
        self._results.append(fit_result)

    def get_coeffs(self):
        return [r.coeffs for r in self._results]

    def get_slopes(self) -> np.ndarray:
        return np.array([r.coeffs[1:] for r in self._results])

    def get_rmses(self):
        return [r.rmse for r in self._results]

    def best_fit(self):
        return min(self._results, key=lambda r: r.rmse)

    def worst_fit(self):
        return max(self._results, key=lambda r: r.rmse)


class Approximator:
    def __init__(self, fit_degree: int = 1):
        self.fit_degree = fit_degree

    def __call__(self, block: Block) -> FitResult:
        Vblock, Ablock = block.get_data()
        if Vblock.shape[0] == 1:
            return self._one_point_block(Ablock)
        Vblock_normed = self._spatial_norm(Vblock)
        return self._luminance_fit(Vblock_normed, Ablock)

    def _one_point_block(self, Ablock) -> FitResult:
        # FIXME: Check if this is right
        logger.warning(
            "Block has fewer points than the fit degree. Returning a constant fit.")
        coeffs = np.array([Ablock[0, 0], 0, 0, 0])
        feature_names = np.array(["x", "y", "z"])
        rmse = 0.0
        return FitResult(coeffs, rmse, feature_names)

    def _luminance_fit(
        self,
        Vblock: np.ndarray,
        Ablock: np.ndarray
    ) -> FitResult:
        Y = Ablock[:, 0]
        poly = PolynomialFeatures(degree=self.fit_degree, include_bias=True)
        Vblock_poly = poly.fit_transform(Vblock)

        model = LinearRegression(fit_intercept=False)
        model.fit(Vblock_poly, Y)
        Y_pred = model.predict(Vblock_poly)

        coeffs = model.coef_
        feature_names = poly.get_feature_names_out(['x', 'y', 'z'])
        rmse = root_mean_squared_error(Y, Y_pred)

        # Relevant logger info
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"Coefficients: {dict(zip(feature_names, coeffs))}")

        return FitResult(coeffs, rmse, feature_names)

    def _spatial_norm(self, Vblock: np.ndarray) -> np.ndarray:
        Vblock_centered = self.center_block(Vblock)
        if Vblock.shape[0] > 1:
            Vblock_rotated = self.rotate_block(Vblock_centered)
            return Vblock_rotated
        return Vblock_centered

    @staticmethod
    def center_block(Vblock: np.ndarray) -> np.ndarray:
        mean = np.mean(Vblock, axis=0)
        logger.debug(f"Block mean for centering: {mean}")
        return Vblock - mean

    @staticmethod
    def rotate_block(Vblock: np.ndarray) -> np.ndarray:
        cov = np.cov(Vblock, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Canonical orientation
        eigvecs[:, 0] *= np.sign(eigvecs[0, 0])
        eigvecs[:, 1] *= np.sign(eigvecs[1, 1])
        eigvecs[:, 2] = np.cross(eigvecs[:, 0], eigvecs[:, 1])

        logger.debug(f"PCA eigenvalues: {eigvals[idx]}")
        logger.debug(f"PCA eigenvectors:\n{eigvecs}")
        return Vblock @ eigvecs
