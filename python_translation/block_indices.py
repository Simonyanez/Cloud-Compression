import numpy as np

def block_indices(V, bsize):
    # V is Nx3, a point cloud, each row are the xyz coordinates of a point, 
    # each coordinate x,y,z is an integer
    # this assumes point cloud is morton ordered
    
    V_coarse = np.floor(V / bsize) * bsize

    variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)

    variation = np.concatenate(([1], variation))

    indices = np.nonzero(variation)[0]

    return indices
