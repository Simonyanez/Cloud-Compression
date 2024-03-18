import numpy as np
from ply import ply_read8i, ply_write
from color_conversion import RGBtoYUV, YUVtoRGB
from ragft import Region_Adaptive_GFT, iRegion_Adaptive_GFT

# Load point cloud data
filename = 'longdress_vox10_1051.ply'
V, Crgb, J = ply_read8i(filename)

# Convert RGB to YUV
C = RGBtoYUV(Crgb)

# Parameters for RA-GFT
bsize = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
param = {'V': V, 'J': J, 'bsize': bsize, 'isMultiLevel': 0}

# Perform RA-GFT encoding
step = 64
Coeff, Gfreq, weights = Region_Adaptive_GFT(C, param)

# Quantize coefficients
Coeff_quant = np.round(Coeff / step) * step

# Perform inverse RA-GFT
start_indices, end_indices, V_MR, Crec = iRegion_Adaptive_GFT(Coeff_quant, param)

# Convert reconstructed YUV to RGB
Crgb_rec = YUVtoRGB(Crec)

# Calculate PSNR for luminance channel
Y = Coeff[:, 0]
psnr_Y = -10 * np.log10(np.linalg.norm(Y - Coeff_quant[:, 0])**2 / (N * 255**2))

# Write original and reconstructed point clouds to PLY files
ply_write('PC_original.ply', V, Crgb, [])
ply_write('PC_coded.ply', V, Crgb_rec, [])