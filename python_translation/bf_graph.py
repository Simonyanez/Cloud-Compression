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