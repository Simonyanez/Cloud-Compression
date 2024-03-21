import numpy as np

def bf_graph(V, C):
    N = V.shape[0]
    
    squared_norms = np.sum(V**2, axis=1)  # Squared distance of each row of V
    D = np.sqrt(np.tile(squared_norms, (N, 1)).T + np.tile(squared_norms, (N, 1)) - 2 * np.dot(V, V.T))  # Distances between all points
    
    th = np.sqrt(3) + 0.00001  # Maximum distance threshold of points
    
    iD = 1 / D  # Scalar inverse of the elements of the matrix, weights are the inverse of the distance
    iD[D > th] = 0  # Find all distances greater than the threshold and evaluate them to 0
    iD[D == 0] = 0  # Also find those that are null, i.e., self-connections
    
    idx = np.where(iD != 0)
    I, J = idx[0], idx[1]  # Identify connected nodes
    
    edge = np.column_stack((I, J))
    W_aux = D.copy()  # Auxiliary set to weight distances
                      # Square matrix of distances between nodes
    
    YUV_block_double = np.array(C, dtype=float)  # Block in YUV format
    YUV_block_normed = YUV_block_double / 256  # Normalization
    
    for id in range(J.shape[0]):
        i = I[id]
        j = J[id]
        W_aux[i, j] = np.exp(-(D[i, j]**2) / (2 * np.std(D)**2)) * np.exp(-(YUV_block_normed[i, 0] - YUV_block_normed[j, 0])**2) / (2 * np.std(YUV_block_normed[:, 0])**2)  # We only use luminance information. We weigh the difference
                                                                                                                                  # between nodes as a parameter for the weights,
                                                                                                                                  # decreases the distance, increases the relevance if the color change is more abrupt.
    
    W = W_aux.T + W_aux
    
    return W, edge

def complete_graph(V):
    """
    Computes a complete graph with edge weights 1/sqrt(distance(vi, vj)).
    
    Parameters:
        V (numpy.ndarray): nx3 array. n points, where vi is the i-th row of V.
        
    Returns:
        numpy.ndarray: Weight matrix representing the complete graph.
    """
    N = V.shape[0]  # Number of points

    # Compute Euclidean Distance Matrix (EDM)
    squared_norms = np.sum(V**2, axis=1)  # Squared norms of each point
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))

    # Compute inverse distances (edge weights)
    iD = 1 / D
    iD[np.where(D == 0)] = 0  # Set elements where distance is zero to zero to avoid division by zero

    # Construct weight matrix by adding transpose of inverse distances
    W = iD.T + iD

    return W

def compute_graph_MSR(V, th=None):
    """
    Compute distance-based graph from Zhang et al., ICIP 2014.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        th (float): Threshold to construct the graph (optional).

    Returns:
        numpy.ndarray: Weight matrix representing the graph.
        numpy.ndarray: Edge list.
    """
    N = V.shape[0]

    if th is None:
        th = np.sqrt(3) + 0.00001

    # Compute Euclidean Distance Matrix (EDM)
    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))

    iD = 1 / D
    iD[np.where(D > th)] = 0
    iD[np.where(D == 0)] = 0

    W = iD.T + iD

    idx = np.nonzero(iD)

    I, J = np.unravel_index(idx, D.shape)

    edge = np.column_stack((I, J))

    return W, edge

def compute_graph_sl(V, distance_vectors, weights, th=None):
    """
    Compute distance-based graph from Zhang et al., ICIP 2014.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        distance_vectors (numpy.ndarray): Distance vectors.
        weights (numpy.ndarray): Weights.
        th (float): Threshold to construct the graph (optional).

    Returns:
        numpy.ndarray: Weight matrix representing the graph.
        numpy.ndarray: Edge list.
        numpy.ndarray: Degree vector.
        numpy.ndarray: Inverse distance matrix.
        numpy.ndarray: Indices of the closest points along the mean direction.
    """
    N = V.shape[0]

    if th is None:
        th = np.sqrt(3) + 0.00001

    mean_X = np.mean(V[:, 0])
    mean_Y = np.mean(V[:, 1])
    mean_Z = np.mean(V[:, 2])
    mean_point_cloud = np.array([mean_X, mean_Y, mean_Z])

    centered_V = V - mean_point_cloud

    mean_direction = np.sum(distance_vectors * weights) / np.sum(weights)
    mean_direction /= np.linalg.norm(mean_direction)

    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))
    iD = 1 / D
    iD[D > th] = 0
    iD[D == 0] = 0

    degrees = np.sum(iD, axis=1)
    degrees /= np.linalg.norm(degrees)

    sorted_indices = np.argsort(degrees)
    below_threshold = int(0.2 * len(sorted_indices))
    idx_closest_original = sorted_indices[:below_threshold]

    selected_vectors = centered_V[idx_closest_original]
    dot_products_degreed = np.dot(selected_vectors, mean_direction)

    idx_closest = np.argsort(dot_products_degreed)[:below_threshold]
    idx_closest = idx_closest_original[idx_closest[0]] if below_threshold > 0 else None

    W = iD.T + iD

    idx = np.nonzero(iD)
    I, J = np.unravel_index(idx, D.shape)
    edge = np.column_stack((I, J))

    return W, edge, degrees, iD, idx_closest