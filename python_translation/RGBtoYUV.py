import numpy as np

def RGBtoYUV(RGB):
    """
    Convert RGB color space to YUV color space.

    Args:
    RGB (ndarray): Nx3 matrix of RGB values (uint8).

    Returns:
    ndarray: Nx3 matrix of YUV values (uint8).
    """
    # Convert RGB to range 0.0 to 1.0 and add a column of ones.
    RGB1 = np.hstack((RGB / 255, np.ones((RGB.shape[0], 1))))

    # Define color transform matrix.
    Q = np.array([[0.29899999, -0.1687, 0.5],
                  [0.587, -0.3313, -0.4187],
                  [0.114, 0.5, -0.0813],
                  [0, 0.50196078, 0.50196078]])

    # Do the transform.
    YUV = RGB1 @ Q

    # Clip to range 0.0 to 1.0 and convert to uint8.
    YUV = np.clip(YUV, 0, 1) * 255
    return YUV
