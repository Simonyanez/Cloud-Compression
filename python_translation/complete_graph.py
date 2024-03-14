import numpy as np

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