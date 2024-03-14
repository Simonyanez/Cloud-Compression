import numpy as np

def gradient(V, C):
    """
    Computes the gradient between points in a point cloud based on their Euclidean distance and color difference.

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

    iD = 1 / D  # Scalar inverse of matrix elements, weights are the inverse of the distance
    iD[D > th] = 0  # Set distances greater than the threshold to 0
    iD[D == 0] = 0  # Set null distances to 0

    idx = np.where(iD != 0)[0]
    I, J = np.unravel_index(idx, D.shape)  # Identify connected nodes

    edge = np.column_stack((I, J))
    D_aux = D.copy()  # Auxiliary set to weight distances
    YUV_block_double = C.astype(float)  # Block in YUV format
    YUV_block_normed = YUV_block_double / 256  # Normalization

    G_vec = np.zeros_like(C[:, 0])
    for id in range(edge.shape[0]):
        i, j = edge[id]
        D_aux[i, j] *= (1 + abs(YUV_block_normed[i, 0] - YUV_block_normed[j, 0]))  # Use only luminance information. Weight the difference
        G_vec[i, j] = abs(YUV_block_normed[j, 0] - YUV_block_normed[i, 0])  # between nodes as a parameter for weights,
                                                                            # decreases distance, increases relevance if color change is more abrupt.

    iD_aux = 1 / D_aux  # If the distance is large (< similarity) weighted by if the color change is large (> relevance)
    iD_aux[D > th] = 0
    iD_aux[D == 0] = 0
    W = iD_aux.T + iD_aux

    return G_vec, W, edge
