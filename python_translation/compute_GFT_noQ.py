def compute_GFT_noQ(Adj, A, idx_closest=None):
    """
    Compute the Graph Fourier Transform (GFT) without using the quality matrix.

    Parameters:
        Adj (numpy.ndarray): Adjacency matrix of the graph.
        A (numpy.ndarray): Attribute matrix.
        idx_closest (numpy.ndarray or None): Index of the closest points (optional).

    Returns:
        numpy.ndarray: Graph Fourier Transform.
        numpy.ndarray: Eigenvalues of the Laplacian matrix (sorted in ascending order).
        numpy.ndarray: Transformed attribute matrix.
    """
    if idx_closest is not None:
        L = w2l(Adj, idx_closest)
    else:
        L = w2l(Adj)

    GFT, D = np.linalg.eig(L)
    idxSorted = np.argsort(np.diag(D))

    GFT = GFT[:, idxSorted]
    GFT[:, 0] = np.abs(GFT[:, 0])
    GFT = GFT.T
    Gfreq = np.abs(np.diag(D))
    Gfreq[0] = np.abs(Gfreq[0])

    Ahat = np.dot(GFT, A)

    return GFT, Gfreq, Ahat