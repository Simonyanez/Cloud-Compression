import os
import numpy as np
from utils import ply
from src.encoder import *

if __name__ == "__main__":
    ply_file = "res/longdress_vox10_1051.ply"
    condition = os.path.exists('V_longdress.npy') and os.path.exists('C_longdress.npy')
    if condition:
        V = np.load('V_longdress.npy')
        C_rgb = np.load('C_longdress.npy')
    else:
        V,C_rgb,_ = ply.ply_read8i(ply_file)
        np.save('V_longdress.npy',V)
        np.save('C_longdress.npy',C_rgb)

    directional_encoder = DirectionalEncoder(V,C_rgb, plots_flag=True)
    directional_encoder.block_indexes(16)
    directional_encoder.block_visualization(100)
    Vblock, Ablock = directional_encoder.get_block(100)
    W, edges = directional_encoder.structural_graph(100)
    GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(100,W, idx_map = None)
    GFT_inv, Ablockrec = directional_encoder.igft_transform(100, W, idx_map = None, Ablockhat= Ablockhat)
    DC_components = Ablockhat[0,:]
    AC_components = Ablockhat[1:,:]
    sorted_components = np.concatenate(DC_components,AC_components)