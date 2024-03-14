import numpy as np

def compute_GFT(Adj, Q):
    """
    Compute the Graph Fourier Transform (GFT).

    Parameters:
        Adj (numpy.ndarray): Adjacency matrix of the graph.
        Q (numpy.ndarray): Quality matrix (assumed to be a vector).

    Returns:
        numpy.ndarray: Graph Fourier Transform.
        numpy.ndarray: Eigenvalues of the Laplacian matrix (sorted in ascending order).
    """
    Qm = np.diag(Q**(-1/2))  # Assume Q is a vector
    L = np.dot(np.dot(Qm, w2l(Adj)), Qm)

    GFT, D = np.linalg.eig(L)
    idxSorted = np.argsort(np.diag(D))

    GFT = GFT[:, idxSorted]
    GFT[:, 0] = np.abs(GFT[:, 0])
    GFT = GFT.T
    Gfreq = np.abs(np.diag(D))
    Gfreq[0] = np.abs(Gfreq[0])

    return GFT, Gfreq