
import numpy as np

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

    def _YUVtoRGB(self, A_yuv:np.ndarray, rounding: bool = True):
        A_yuv_1 = np.concatenate((A_yuv / 255, np.ones((A_yuv.shape[0], 1))), axis=1)
        A_rgb = np.dot(A_yuv_1, self.M_YUVtoRGB)
        A_rgb = 255 * np.clip(A_rgb, 0, 1)
        if rounding:
            A_rgb = A_rgb.round().astype(np.uint8)

        return A_rgb
    
    def _RGBtoYUV(self, A_rgb: np.ndarray, rounding=False) -> np.ndarray:
        A_rgb_1 = np.concatenate((A_rgb / 255, np.ones((A_rgb.shape[0], 1))), axis=1)
        A_yuv = np.dot(A_rgb_1, self.Q_RGBtoYUV)
        A_yuv = 255 * np.clip(A_yuv, 0, 1)
        if rounding:
            A_yuv = A_yuv.round().astype(np.uint8)
        return A_yuv