# Full Import
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

# From Library
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Own Imports
from utils import ply
from utils.visualization import *
from utils.color import *
from graph.transforms import *
from graph.create import *

if __name__ == "__main__":
    V,C_rgb,J = ply.ply_read8i("res/longdress_vox10_1051.ply")         # Read
    # N = V.shape()[0]
    C = RGBtoYUV(C_rgb)                                                # Attribute to YUV
    bsize = 16

    params = {
        'V' : V,
        'J' : J,
        'bsize' : bsize,
        'isMultiLevel' : False,
    }

    step = 64
    SubBlocks = block_visualization(C,params) 

    # Typical GFT
    T_og = {'GFTs': [], 'Gfreqs' : [], 'Ahats'  : [], 'Vblocks' : [] }
    for Block in SubBlocks:
        Vblock = Block['Vblock']
        Cblock = Block['Ablock']
        W,edge = compute_graph_MSR(Vblock)
        GFT, Gfreq, Ahat = compute_GFT_noQ(W,Cblock)        # Cblock is just for getting Ahat
        T_og['GFTs'].append(GFT)
        T_og['Gfreqs'].append(Gfreq)
        T_og['Ahats'].append(Ahat)
        T_og['Vblocks'].append(Vblock)

    # Inverse GFT

    RecBlocks = {'Vblock':[], 'Ablock':[]}
    _,_,Ahat_vals,Vblocks = T_og.values()
    for id,Ahat_val in enumerate(Ahat_vals):
        Vblock = Vblocks[id]
        Vblock, Arec = compute_iGFT_noQ(Vblock, Ahat_val)
        RecBlocks['Vblock'].append(Vblock)
        RecBlocks['Ablock'].append(Arec)

    

    # Concatenate
    # print(f"Shape for {np.shape(RecBlocks['Vblock'][0])}")
    # print(f"Shape for {np.shape(RecBlocks['Ablock'][0])}")
