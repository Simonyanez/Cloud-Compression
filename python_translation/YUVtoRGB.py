import numpy as np

def YUVtoRGB(YUV):
    """
    Convert YUV to RGB.

    Args:
        YUV (numpy.ndarray): YUV matrix.

    Returns:
        RGB (numpy.ndarray): RGB matrix.
    """
    # First convert YUV to range 0.0 to 1.0, and add a column of ones.
    YUV1 = np.concatenate([YUV / 255, np.ones((YUV.shape[0], 1))], axis=1)

    # Define color transform.
    M = np.array([[1, 1, 1],
                  [0, -0.34414, 1.772],
                  [1.402, -0.71414, 0],
                  [-0.703749019, 0.53121505, -0.88947451]])

    # Do the transform.
    RGB = np.dot(YUV1, M)

    # Clip to range 0.0 to 1.0, and convert to uint8.
    RGB = np.uint8(255 * np.clip(RGB, 0, 1))

    return RGB
