import numpy as np

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
