
# Importing Parent Folder to Path
import os
main_folder = os.getcwd()
import sys
sys.path.insert(0, main_folder)

# Other imports
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

    D = np.diag(np.sum(W, axis=0))

    # Be careful that C = np.diag(np.diag(W)) if the self-loops are originally at the structure of the graph

    L = C + D - W + np.diag(np.diag(W))
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

    # L is normalized by the way it's build
    D, GFT = eigh(L) # D eigen values and GFT eigenvectors
    idxSorted = np.argsort(D)      # Order of the eigenvalues

    GFT = GFT[:,idxSorted]    # GFT ordered by eigenvalues order first less
    GFT[:,0] = np.abs(GFT[:,0])
    GFT = GFT.T         # Because the matrix that do the transform is this one
    Gfreq = np.sort(D)

    Gfreq[0] = np.abs(Gfreq[0])

    Ahat = np.matmul(GFT, A)      # @ is a shortcut for matmul, yet i dont like it
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

    Arec = np.matmul(GFT_inv, Ahat_val)
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


if __name__ == "__main__":
    W = np.array([[0,1,2],[5,0,9],[6,4,0]])
    print(f"This is the W matrix \n {W}")
    L = w2l(W)
    print(f"This is the resulting L matrix \n {L}")
    A = [[4,8,1],[4,8,1],[4,8,1]]
    print(A[1:3])
    
    GFT, Gfreq, Ahat=compute_GFT_noQ(W,A)
    print(Ahat)