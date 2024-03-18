import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt
from utils.color import YUVtoRGB

def visualization(Vblock, Ablock, im_num, method, *varargin):
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
    ax.set_title(f'3D Point Cloud Plot with Custom Colors - Image Number: {im_num} Method: {method}')
    
    # Set view angle
    ax.view_init(elev=60, azim=30)
    fig.show()
    # Get aspect ratio
    aspect_ratio = ax.get_box_aspect()
    
    return aspect_ratio