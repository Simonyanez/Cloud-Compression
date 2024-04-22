# Full Import
import numpy as np
import matplotlib.pyplot as plt
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

    error_1 = []
    
    for i,Block in enumerate(SubBlocks): 
        Vblock = Block['Vblock']
        Cblock = Block['Ablock']
        W,_ = compute_graph_MSR(Vblock)
        _, _, Ahat = compute_GFT_noQ(W,Cblock)        # Cblock is just for getting Ahat
        if i == 10:
            fig1 = hist_plot(Ahat,"Value distribution")
        _, Arec = compute_iGFT_noQ(Vblock, Ahat)
        mse = np.square(Cblock - Arec).mean(axis=None)
        error_1.append(mse)

    fig2 = hist_plot(error_1,"MSE between Original and Recovered")
    plt.show()
    

    # print(f"Shape for {np.shape(RecBlocks['Vblock'][0])}")
    # print(f"Shape for {np.shape(RecBlocks['Ablock'][0])}")
