import numpy as np

def gradient(V, C):
    #TODO: Make this function work from a constructed graph, don't do it again
    """
    Computes the gradient between points in a point cloud build graph
    based on their Euclidean distance and color difference.

    Parameters:
    - V (ndarray): Array of 3D points.
    - C (ndarray): Array of RGB colors.

    Returns:
    - G_vec (ndarray): Gradient vector.
    - W (ndarray): Weight matrix.
    - edge (ndarray): Indices of connected nodes.
    """
    N = V.shape[0]

    squared_norms = np.sum(V**2, axis=1)   # Squared distance of each row of V
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))  # Squared distances between all points

    th = np.sqrt(3) + 0.00001  # Maximum distance threshold of points

    iD = np.zeros_like(D) 
    non_zero_mask = (D > 0) & (D <= th)
    iD[non_zero_mask] = 1 / D[non_zero_mask]
    iD[D > th] = 0
    iD[D == 0] = 0

    idx = np.where(iD != 0)[0]
    I, J = np.unravel_index(idx, D.shape)  # Identify connected nodes

    edge = np.column_stack((I, J))
    D_aux = D.copy()  # Auxiliary set to weight distances
    YUV_block_double = C.astype(float)  # Block in YUV format
    YUV_block_normed = YUV_block_double / 256  # Normalization

    G_vec = np.zeros(D.shape)
    for id in range(edge.shape[0]):
        i, j = edge[id]
        D_aux[i, j] *= (1 + abs(YUV_block_normed[i, 0] - YUV_block_normed[j, 0]))  # Use only luminance information. Weight the difference
        G_vec[i, j] = abs(YUV_block_normed[j, 0] - YUV_block_normed[i, 0])  # between nodes as a parameter for weights,
                                                                            # decreases distance, increases relevance if color change is more abrupt.

    iD_aux = np.zeros_like(D_aux) 
    non_zero_mask = (D_aux > 0) & (D_aux <= th)
    iD[non_zero_mask] = 1 / D[non_zero_mask] # If the distance is large (< similarity) weighted by if the color change is large (> relevance)
    iD_aux[D > th] = 0
    iD_aux[D == 0] = 0
    W = iD_aux.T + iD_aux

    return G_vec

import matplotlib.pyplot as plt

def direction(V, C, aspect_ratio=None):
    """
    Compute direction vectors between points in a point cloud.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        C (numpy.ndarray): Color information corresponding to each point in V.
        aspect_ratio (tuple): Aspect ratio for plotting (optional).

    Returns:
        numpy.ndarray: Direction vectors.
        numpy.ndarray: Edge list.
        numpy.ndarray: Distance vectors.
        numpy.ndarray: Weights.
    """
    # Structural graph construction
    N = V.shape[0]

    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))

    th = np.sqrt(3) + 0.00001

    iD = np.zeros_like(D) 
    non_zero_mask = (D > 0) & (D <= th)
    iD[non_zero_mask] = 1 / D[non_zero_mask]
    iD[D > th] = 0
    iD[D == 0] = 0

    idx = np.nonzero(iD)
    I, J = np.unravel_index(idx, D.shape)
    edge = np.column_stack((I, J))

    # Attribute information
    YUV_block_double = C.astype(float) / 256
    c_len = C.shape[0]

    # Vector of color difference between connected nodes
    G_vec = np.zeros((c_len, c_len))
    for id in range(len(J)):
        i = I[id]
        j = J[id]
        G_vec[i, j] = YUV_block_double[j, 0] - YUV_block_double[i, 0]

    # Matrix to store direction
    distance_vectors = np.zeros((G_vec.shape[0], 3))
    distance_indexes = np.zeros((G_vec.shape[0], 2))
    weights = np.zeros(G_vec.shape[0])

    for iter in range(G_vec.shape[0]):
        min_val = np.min(G_vec[:, iter])
        min_index = np.argmin(G_vec[:, iter])
        if np.abs(min_val) > 0.02:
            distance_indexes[iter, :] = [min_index, iter]
            weights[iter] = np.abs(G_vec[min_index, iter])
            dis_vec = V[iter, :] - V[min_index, :]
        else:
            dis_vec = np.array([0, 0, 0])
        
        if np.linalg.norm(dis_vec) != 0:
            dis_vec = dis_vec / np.linalg.norm(dis_vec)
            distance_vectors[iter, :] = dis_vec
        else:
            distance_vectors[iter, :] = dis_vec

    if aspect_ratio is not None:
        x = V[:, 0]
        y = V[:, 1]
        z = V[:, 2]
        u = distance_vectors[:, 0]
        v = distance_vectors[:, 1]
        w = distance_vectors[:, 2]
        magnitudes = np.sqrt(u**2 + v**2 + w**2)
        u_unit = u / magnitudes
        v_unit = v / magnitudes
        w_unit = w / magnitudes
        r = np.mean(distance_vectors, axis=0)
        r = r / np.linalg.norm(r)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u_unit, v_unit, w_unit)
        ax.quiver(0, 0, 0, r[0], r[1], r[2], color='r', linewidth=6, arrow_length_ratio=0.1)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Unit Vectors in 3D')
        ax.grid(True)
        ax.set_box_aspect(aspect_ratio)
        ax.set_xlim([np.min(x) - 1, np.max(x) + 1])
        ax.set_ylim([np.min(y) - 1, np.max(y) + 1])
        ax.set_zlim([np.min(z) - 1, np.max(z) + 1])
        plt.show()

    return G_vec, edge, distance_vectors, weights

def block_indices(V, bsize):
    # V is Nx3, a point cloud, each row are the xyz coordinates of a point, 
    # each coordinate x,y,z is an integer
    # this assumes point cloud is morton ordered
    
    V_coarse = np.floor(V / bsize) * bsize

    variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)

    variation = np.concatenate(([1], variation))

    indices = np.nonzero(variation)[0]

    return indices
