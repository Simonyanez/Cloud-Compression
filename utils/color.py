import numpy as np

# Functions below translated from MATLAB to Python from
# https://github.com/STAC-USC/RA-GFT

Q_RGBtoYUV = np.array([
        [0.29899999,    -0.1687,         0.5],
        [0.587,         -0.3313,        -0.4187],
        [0.114,          0.5,           -0.0813],
        [0,              0.50196078,     0.50196078]
])

M_YUVtoRGB = np.array([
        [1,             1,           1],
        [0,            -0.34414,     1.772],
        [1.402,        -0.71414,     0],
        [-0.703749019,   0.53121505, -0.88947451]
])


def RGBtoYUV(rgb, rounding=False):

    # Limit values between 0 and 1 (instead of 0 and 255)
    # Append column filled with ones
    rgb1 = np.concatenate(
        (rgb/255, np.ones((rgb.shape[0], 1))), axis=1)

    # Conversion
    yuv = np.dot(rgb1, Q_RGBtoYUV)

    # Limit values between 0 and 255
    yuv = 255*np.clip(yuv, 0, 1)

    # Round to integer values
    if rounding:
        yuv = yuv.round().astype(np.uint8)

    return yuv


def YUVtoRGB(yuv, rounding=True):

    # Limit values between 0 and 1 (instead of 0 and 255)
    # Append column filled with ones
    yuv1 = np.concatenate(
        (yuv/255, np.ones((yuv.shape[0], 1))), axis=1)

    # Conversion
    rgb = np.dot(yuv1, M_YUVtoRGB)

    # Limit values between 0 and 255
    rgb = (255*np.clip(rgb, 0, 1))

    # Round to integer values
    if rounding:
        rgb = rgb.round().astype(np.uint8)

    return rgb

if __name__ == "__main__":
    Crec = np.random.randint(0, 256, (5, 3), dtype=np.uint8)
    Crgb_rec = YUVtoRGB(Crec)
    #Crgb_rec= Crgb_rec.astype(np.float64)
    print(type(Crgb_rec[0][0]))
