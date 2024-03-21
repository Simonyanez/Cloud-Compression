import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt
from graph.properties import block_indices
from utils.color import YUVtoRGB
from graph.properties import direction,gradient
from graph.create import compute_graph_sl

def visualization(Vblock, Ablock, Amean, Astd, method, *varargin):
    """
    Visualization of 3D point cloud with custom colors.
    
    Args:
        Vblock (numpy.ndarray): Nx3 array of point cloud coordinates.
        Ablock (numpy.ndarray): Nx3 array of attributes (e.g., colors).
        im_num (int): Image number.
        method (str): Method used for visualization.
        *varargin: Additional optional arguments, e.g., 'GFT' and GFT data.
        
    Returns:
        aspect_ratio (tuple): Aspect ratio of the plot.
    """
    # Extract coordinates
    X_block, Y_block, Z_block = Vblock[:, 0], Vblock[:, 1], Vblock[:, 2]
    
    # Calculate mean coordinates
    X_mean, Y_mean, Z_mean = np.mean(Vblock, axis=0)
    
    # Convert YUV attributes to RGB
    RGB_block = YUVtoRGB(Ablock)/256
    RGB_block_double = RGB_block.astype(float)
    
    # Create a figure
    fig = plt.figure()
    fig.set_facecolor([0.7, 0.7, 0.7])
    
    # Set axis properties
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.grid(True)
    
    # Check if 'GFT' is provided as an argument and if GFT data is provided
    if 'GFT' in varargin and len(varargin) > 1:
        idx = varargin.index('GFT')

        if idx < len(varargin) - 1:
            GFT = varargin[idx + 1]

            # Scatter plot with GFT as colormap
            ax.scatter3D(X_block, Y_block, Z_block, c=GFT, s=20, cmap='hot')
            plt.colorbar()
        else:
            print("No value provided for 'GFT' property.")
    else:
        # Scatter plot with RGB attributes
        ax.scatter3D(X_block, Y_block, Z_block, c=RGB_block_double, s=20)
    
    # Overlay additional points (e.g., mean)
    ax.scatter3D(X_mean, Y_mean, Z_mean, c='black', s=100)
    
    # Set axis labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'3D Point Cloud Plot with Custom Colors - STD: {round(Astd,2)} Mean: {round(Amean,2)} Method {method}')
    
    # Set view angle
    ax.view_init(elev=60, azim=30)

    #plt.show()

    # Get aspect ratio
    aspect_ratio = ax.get_box_aspect()
    
    return aspect_ratio,fig

def hist_plot(data, stat):
    """
    Plot a histogram of the given data with a specified title.
    
    Args:
        data: Input data to plot the histogram for.
        stat (str): Additional information to include in the title.
        
    Returns:
        fig: The figure object containing the histogram.
    """
    fig, ax = plt.subplots()
    ax.hist(data)
    ax.set_title(f"Histogram of Blocks {stat}")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    
    return fig

def cluster_visualization(X, centroids, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='red')
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Color Mean')
    ax.set_ylabel('Color Standard Deviation')
    return fig

def cluster_visualization3d(X, centroids, labels):
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111, projection='3d')
    
    # Scatter plot for data points
    scatter = ax_1.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
    
    # Centroids plot
    ax_1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, color='red')
    
    ax_1.set_title('K-Means Clustering')
    ax_1.set_xlabel('Color Mean')
    ax_1.set_ylabel('Color Standard Deviation')
    ax_1.set_zlabel('Graph Color Diff Mean')
    
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111, projection='3d')
    
    # Scatter plot for data points
    scatter = ax_2.scatter(X[:, 0], X[:, 1], X[:, 3], c=labels, cmap='viridis')
    
    # Centroids plot
    ax_2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, color='red')
    
    ax_2.set_title('K-Means Clustering')
    ax_2.set_xlabel('Color Mean')
    ax_2.set_ylabel('Color Standard Deviation')
    ax_2.set_zlabel('Graph Color Diff Standard Deviation')

    return fig_1, fig_2

def block_visualization(A, params):
    V = params['V']
    b = params['bsize']
    J = params['J']
    isMultiLevel = params['isMultiLevel']
    N = V.shape[0]

    if isinstance(b, int) or len(b) == 1:
        if isMultiLevel:
            base_bsize = np.log2(b)
            if not np.all(np.floor(base_bsize) == base_bsize):
                raise ValueError('block size b should be a power of 2')
            L = J // base_bsize
            if L != np.floor(L):
                raise ValueError('block size does not match number of levels')
            bsize = np.ones(L) * b
        else:
            base_bsize = np.log2(b)
            if not np.all(np.floor(base_bsize) == base_bsize):
                raise ValueError('block size b should be a power of 2')
            L = 1
            bsize = np.array([b])
    else:
        bsize = np.array(b)
        L = len(bsize)
        base_bsize = np.log2(b)
        if not np.all(np.floor(base_bsize) == base_bsize):
            raise ValueError('entries of block size should be a power of 2')
        if np.sum(base_bsize) > J:
            raise ValueError('block sizes do not match octree depth J')

    Ahat = None
    Vcurr = V
    Acurr = A
    Qin = np.ones(N)
    Gfreq_curr = np.zeros(N)
    freqs = []
    weights = []
    Sorted_Blocks = []

    for level in range(L, 0, -1):
        # Start and ending index of the constructed blocks
        start_indices = block_indices(Vcurr, bsize[level - 1])  
        Nlevel = Vcurr.shape[0]                                 
        end_indices = np.concatenate((start_indices[1:] - 1, np.array([Nlevel - 1])))
        
        ni = end_indices - start_indices + 1 # Points per block
        to_change = np.where(ni != 1)[0] # Number of blocks with more than one point 
        print(f"Cantidad de bloques con más de un punto {np.shape(to_change)}")

        # Initialization of parameters to modify (remnants of old code)
        Acurr_hat = Acurr   
        Qout = Qin.copy()
        Gfreq_curr = np.zeros(N)

        # Dictionary for useful information
        SubBlocks = []

        for currblock in to_change:         # Iteración por bloque
            first_point = start_indices[currblock]      # First point of current block
            last_point = end_indices[currblock]         # Last point of current block
            Vblock = Vcurr[first_point:last_point + 1, :]   # Block vertex
            Qin_block = Qin[first_point:last_point + 1]     # Remnant of other implementation
            Ablock = Acurr[first_point:last_point + 1, :]   # Block attributes (tipicaly color)

            # Clustering
            # Add clustering logic here
            #_,_, distance_vectors, weights = direction(Vblock,Ablock)
            GraphDiffsMatrix = gradient(Vblock, Ablock)

            # Metrics
            AttributeMean = np.mean(Ablock)
            AttributeSTD = np.std(Ablock)
            GraphDiffsMean = np.mean(GraphDiffsMatrix)
            GraphDiffsSTD = np.std(GraphDiffsMatrix)

            # For now this is the block data
            block_data = {
                        'Vblock': Vblock,
                        'Ablock': Ablock,
                        'Amean':AttributeMean,
                        'Astd':AttributeSTD,
                        'Dmatrix':GraphDiffsMatrix,
                        'Dmean':GraphDiffsMean,
                        'Dstd':GraphDiffsSTD
                            }
            
            SubBlocks.append(block_data)
    # OLD -> return Ahat, freqs, weights, Vblock, Ablock, Sorted_Blocks
    # Returning last block
    return SubBlocks

def coeff_visualization(Ahat_orig, Ahat_mod, distance, cluster):
    Y_og = abs(Ahat_orig[:, 0])
    U_og = abs(Ahat_orig[:, 1])
    V_og = abs(Ahat_orig[:, 2])

    Y_mod = abs(Ahat_mod[:, 0])
    U_mod = abs(Ahat_mod[:, 1])
    V_mod = abs(Ahat_mod[:, 2])

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(range(1, len(Y_og) + 1), Y_og, marker='s')
    ax.scatter(range(1, len(Y_mod) + 1), Y_mod, marker='d')
    ax.legend(['Original', 'Modified'])
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Transform coefficients Y - Distance = {distance} Cluster {cluster}')
    ax.set_xlim(0.5, len(Y_og) + 0.5)
    ax.set_ylim(min(Y_og) - 1, max(Y_og) + 1)
    ax.axis('tight')

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(range(1, len(U_og) + 1), U_og, marker='s')
    ax.scatter(range(1, len(U_mod) + 1), U_mod, marker='d')
    ax.legend(['Original', 'Modified'])
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Transform coefficients U - Distance = {distance} Cluster {cluster}')
    ax.set_xlim(0.5, len(U_og) + 0.5)
    ax.set_ylim(min(U_og) - 1, max(U_og) + 1)
    ax.axis('tight')

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(range(1, len(V_og) + 1), V_og, marker='s')
    ax.scatter(range(1, len(V_mod) + 1), V_mod, marker='d')
    ax.legend(['Original', 'Modified'])
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Transform coefficients V - Distance = {distance} Cluster {cluster}')
    ax.set_xlim(0.5, len(V_og) + 0.5)
    ax.set_ylim(min(V_og) - 1, max(V_og) + 1)
    ax.axis('tight')

    plt.show()