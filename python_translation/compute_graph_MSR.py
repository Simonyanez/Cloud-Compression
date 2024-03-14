import numpy as np

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
