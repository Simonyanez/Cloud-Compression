
import numpy as np
from utils import ply
from utils.visualization import *
from utils.color import *
if __name__ == "__main__":
    V,C_rgb,J = ply.ply_read8i("res/longdress_vox10_1051.ply")         # Read Test
    N = V.shape(0)
    C = RGBtoYUV(C_rgb)
    bsize = 16
    params = {
        'V' : V,
        'J' : J,
        'bsize' : bsize,
        'isMultiLevel' : False,
    }
    step = 64
    block_visualization(C,params) 
    print(f"This is geometric shape {np.shape(V[:,1])}")
    aspect_ratio = visualization(V,C,1,1)