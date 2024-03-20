
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

from utils import ply
from utils.visualization import *
from utils.color import *

if __name__ == "__main__":
    V,C_rgb,J = ply.ply_read8i("res/longdress_vox10_1051.ply")         # Read
    #N = V.shape()[0]
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
    
    SizeSortedBlocks = sorted(SubBlocks, key =lambda x: len(x['Vblock'].flatten()))
    print(f"This is number of blocks {len(SizeSortedBlocks)}")
    
    biggest_block = SizeSortedBlocks[-1]
    Vbiggest, Cbiggest, Cmean, Cstd = (biggest_block['Vblock'],
                          biggest_block['Ablock'],
                          biggest_block['Amean'],
                          biggest_block['Astd']
    )

    print(f"This block size {np.shape(Vbiggest)}")
    print(f"This block attributes size {np.shape(Cbiggest)}")
    
    Blocks_Mean = [x['Amean'] for x in SubBlocks]
    Blocks_STD = [x['Astd'] for x in SubBlocks]
    
    means_fig = hist_plot(Blocks_Mean,'Mean')
    stds_fig = hist_plot(Blocks_STD,'Standard Deviation')
    aspect_ratio,block_fig = visualization(Vbiggest,Cbiggest,Cmean,Cstd,None)
    plt.show()