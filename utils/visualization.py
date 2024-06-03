import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from graph.properties import block_indices
from utils.color import YUVtoRGB
from graph.properties import direction,gradient
from graph.create import compute_graph_sl

def min_max_normalize(vector_values):
    """
    Perform min-max normalization on a vector.

    Args:
        vector_values (numpy.ndarray): 1D array of vector values.

    Returns:
        numpy.ndarray: Min-max normalized vector.
    """
    min_val = np.min(vector_values)
    max_val = np.max(vector_values)
    normalized_values = (vector_values - min_val) / (max_val - min_val)
    return normalized_values

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
    ax.set_title(f'Color change direction by node for block with {len(Vblock[:, 0])} points')
    
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
    # This function does block octotree partition and clusterization features
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
            _,_, DistVecs, Weights = direction(Vblock,Ablock)
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
                        'Dstd':GraphDiffsSTD,
                        'DistVecs':DistVecs,
                        'Wblock':Weights
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

    return fig


def direction_visualization(Vblock, distance_vectors):
    x = Vblock[:, 0]
    y = Vblock[:, 1]
    z = Vblock[:, 2]
    u = distance_vectors[:, 0]
    v = distance_vectors[:, 1]
    w = distance_vectors[:, 2]
    
    # Calculate magnitudes
    magnitudes = np.linalg.norm(distance_vectors, axis=1)
    
    # Check for zero magnitudes to avoid division by zero
    zero_magnitudes = magnitudes == 0
    if np.any(zero_magnitudes):
        # Replace zero magnitudes with small values to avoid division by zero
        magnitudes[zero_magnitudes] = 1e-6
    
    # Normalize vectors
    u_unit = u / magnitudes
    v_unit = v / magnitudes
    w_unit = w / magnitudes
    
    # Calculate mean vector
    r = np.mean(distance_vectors, axis=0)
    r_norm = np.linalg.norm(r)
    
    # Check for zero mean vector to avoid division by zero
    if r_norm == 0:
        r_norm = 1e-6  # Replace zero norm with small value to avoid division by zero
    
    r = r / r_norm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, u_unit, v_unit, w_unit)
    ax.quiver(np.mean(Vblock[:,1]),np.mean(Vblock[:,2]),np.mean(Vblock[:,3]), r[0], r[1], r[2], color='r', linewidth=6, arrow_length_ratio=0.1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Color change direction by node for block with {len(Vblock[:, 0])} points')
    ax.grid(True)
    ax.set_xlim([np.min(x) - 1, np.max(x) + 1])
    ax.set_ylim([np.min(y) - 1, np.max(y) + 1])
    ax.set_zlim([np.min(z) - 1, np.max(z) + 1])
    return fig

def border_visualization(Vblock, Ablock, borders_idx):
    """
    Visualization of 3D point cloud with custom colors.
    
    Args:
        Vblock (numpy.ndarray): Nx3 array of point cloud coordinates.
        Ablock (numpy.ndarray): Nx3 array of attributes (e.g., colors).
        borders_idx (numpy.ndarray): Indices of border points.
        
    Returns:
        aspect_ratio (tuple): Aspect ratio of the plot.
        fig (matplotlib.figure.Figure): The resulting figure object.
    """
    # Extract coordinates
    X_block, Y_block, Z_block = Vblock[:, 0], Vblock[:, 1], Vblock[:, 2]
    
    # Calculate mean coordinates
    X_mean, Y_mean, Z_mean = np.mean(Vblock, axis=0)
    
    # Convert YUV attributes to RGB
    RGB_block = Ablock / 255.0
    
    # Create a figure
    fig = plt.figure()
    # Set axis properties
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('black')
    # Scatter plot with RGB attributes
    y_values = Ablock[:, 0]/255.0
    sc = ax.scatter3D(X_block, Y_block, Z_block, c=y_values, cmap='inferno', s=50, alpha=0.8)
    
    cb = plt.colorbar(sc)
    cb.set_label('Y values', color="white")
    # Customize colorbar ticks
    cb.ax.yaxis.set_tick_params(color='white')  # Set tick color
    cb.outline.set_edgecolor('white') 
    # Set colorbar tick labels color
    cb.ax.tick_params(colors='white')
    # Set colorbar tick values color
    cb.ax.yaxis.offsetText.set_color('white')

    # Overlay additional points (e.g., mean)
    ax.scatter3D(X_mean, Y_mean, Z_mean, c='white', s=100, edgecolors='black')
    
    # Add borders
    border_points = Vblock[borders_idx]
    ax.scatter3D(border_points[:, 0], border_points[:, 1], border_points[:, 2], color='cyan', s=150, alpha=0.1, edgecolors='black')

    # Annotate index positions
    for idx, (x, y, z) in zip(borders_idx, border_points):
        ax.text(x, y, z + 0.01, str(idx), color='white')
    
    # Set axis labels and title
    ax.set_xlabel('X-axis', color='white')
    ax.set_ylabel('Y-axis', color='white')
    ax.set_zlabel('Z-axis', color='white')
    ax.set_title(f'Block with {len(Vblock)} points and border points', color='white')
    
    # Set view angle
    ax.view_init(elev=30, azim=45)

    # Set background color
    fig.set_facecolor('black')
    

    
    return y_values, fig

def component_visualization(Vblock, base,version,reference_base=None):
    """
    Visualization of 3D point cloud with base colors.
    
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


    # Create a figure
    fig_1 = plt.figure()

    # Set axis properties
    ax_1 = fig_1.add_subplot(111, projection='3d')
    ax_1.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax_1.grid(True)
    abs_base = np.abs(base)
        
    norm = Normalize(vmin=np.min(base), vmax=np.max(base))
    # Scatter plot with RGB attributes
    sc = ax_1.scatter3D(X_block, Y_block, Z_block, c=base, cmap='inferno', norm=norm,s=50, alpha=0.8)

    # Overlay additional points (e.g., mean)
    ax_1.scatter3D(X_mean, Y_mean, Z_mean, c='black', s=50)
    plt.gca().set_facecolor('black')
    
    cb = plt.colorbar(sc)
    cb.set_label('Y values', color="white")
    # Customize colorbar ticks
    cb.ax.yaxis.set_tick_params(color='white')  # Set tick color
    cb.outline.set_edgecolor('white') 
    # Set colorbar tick labels color
    cb.ax.tick_params(colors='white')
    # Set colorbar tick values color
    cb.ax.yaxis.offsetText.set_color('white')
    # Set axis labels and title
    ax_1.set_xlabel('X-axis',color="white")
    ax_1.set_ylabel('Y-axis',color='white')
    ax_1.set_zlabel('Z-axis', color='white')
    ax_1.set_title(f'Base colormap projection for block with {len(Vblock[:, 0])} points and {version} method',color="white")

    # Set view angle
    ax_1.view_init(elev=60, azim=30)
    fig_1.set_facecolor('black')


    return fig_1

def Yvisualization(Vblock,Ablock):
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
    Y = Ablock[0]/255.0
    
    # Create a figure
    fig = plt.figure()
    
    # Set axis properties
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.grid(True)
    
    
    # Scatter plot with RGB attributes
    ax.scatter3D(X_block, Y_block, Z_block, c=Y, s=20)
    
    # Overlay additional points (e.g., mean)
    ax.scatter3D(X_mean, Y_mean, Z_mean, c='black', s=50)

    # Set axis labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Color change direction by node for block with {len(Vblock[:, 0])} points')
    
    # Set view angle
    ax.view_init(elev=60, azim=30)

    #plt.show()

    # Get aspect ratio
    aspect_ratio = ax.get_box_aspect()
    
    return aspect_ratio,fig

def base_plot(base,choosed_weights=None):
    fig, ax = plt.subplots()
    ax.plot(base, marker='o', linestyle='-')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Base Plot')
    # Adding a LaTeX formatted title with dynamic values
    title = r'Base Plot $\alpha={}$ $\beta={}$'.format(choosed_weights[0], choosed_weights[1])
    ax.set_title(title)
    ax.grid(True)
    return fig