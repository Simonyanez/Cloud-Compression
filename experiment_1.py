
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
    
    Blocks_Color_Mean = [x['Amean'] for x in SubBlocks]
    Blocks_Color_STD = [x['Astd'] for x in SubBlocks]
    Blocks_Diff_Mean = [x['Dmean'] for x in SubBlocks]
    Blocks_Diff_STD = [x['Dstd'] for x in SubBlocks]

    means_fig = hist_plot(Blocks_Color_Mean,'Mean')
    stds_fig = hist_plot(Blocks_Color_STD,'Standard Deviation')
    aspect_ratio,block_fig = visualization(Vbiggest,Cbiggest,Cmean,Cstd,None)
    
    Features = np.column_stack((Blocks_Color_Mean, 
                                Blocks_Color_STD, 
                                Blocks_Diff_Mean, 
                                Blocks_Diff_STD))
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Normalize the features
    Features_normalized = scaler.fit_transform(Features)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(Features_normalized)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    cluster_fig1, cluster_fig2 = cluster_visualization3d(Features_normalized, centroids, labels)
    plt.show()

