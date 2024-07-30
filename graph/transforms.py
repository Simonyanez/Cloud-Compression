import numpy as np
from graph.create import *
from scipy.linalg import eigh

def w2l(W, idx_closest_map=None):
    """
    Convert weight matrix to Laplacian matrix.

    Args:
        W (numpy.ndarray): Weight matrix.
        idx_closest (numpy.ndarray): Indices of the closest points (optional).

    Returns:
        L (numpy.ndarray): Laplacian matrix.
    """
    sz_W = W.shape
    C = np.zeros(sz_W)

    if np.any(W < 0):
        # raise ValueError('W is not a valid weight matrix')
        pass  # Handle negative weights differently if needed

    if idx_closest_map is not None:
        for idx in idx_closest_map.keys():
            #C[np.ix_(idx, idx)] = idx_closest_map[idx]
            C[idx,idx] = idx_closest_map[idx]
    L = C + np.diag(np.sum(W, axis=1)) - W + np.diag(np.diag(W))

    return L

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

    D, GFT = eigh(L) # D eigen values and GFT eigenvectors
    idxSorted = np.argsort(D)      # Order of the eigenvalues

    GFT = GFT[:,idxSorted]    # GFT ordered by eigenvalues order first less
    GFT[:,0] = np.abs(GFT[:,0])
    #GFT = GFT.T
    Gfreq = np.sort(D)

    Gfreq[0] = np.abs(Gfreq[0])

    Ahat = np.dot(GFT.T, A)
    return GFT, Gfreq, Ahat


def compute_iGFT_noQ(Adj, Ahat_val, idx_closest=None):
    if idx_closest is not None:
        L = w2l(Adj, idx_closest)
    else:
        L = w2l(Adj)

    D, GFT = eigh(L) # D eigen values and GFT eigenvectors
    idxSorted = np.argsort(D)      # Order of the eigenvalues
    GFT = GFT[:,idxSorted]    # GFT ordered by eigenvalues order
    GFT[:,0] = np.abs(GFT[:,0])
    GFT = GFT.T

    GFT_inv = np.linalg.inv(GFT)

    Arec = np.dot(GFT_inv, Ahat_val)
    return GFT_inv, Arec

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