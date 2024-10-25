
# Importing Parent Folder to Path
import os
main_folder = os.getcwd()
import sys
sys.path.insert(0, main_folder)

# Other imports
import matplotlib.pyplot as plt
import numpy as np
from graph.create import *
#from scipy.linalg import eigh

def w2l(W, idx_closest_map=None, iter=None):
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
        if iter is not None:
            print(f"This is the block that has complex values {iter}")
        # Handle negative weights differently if needed

    if idx_closest_map is not None:
        for idx in idx_closest_map.keys():
            #C[np.ix_(idx, idx)] = idx_closest_map[idx]
            W[idx,idx] = idx_closest_map[idx]

    D = np.diag(np.sum(W, axis=0))

    # Be careful that C = np.diag(np.diag(W)) if the self-loops are originally at the structure of the graph

    L = D - W + np.diag(np.diag(W))
    return L

def compute_GFT_noQ(Adj, A, idx_closest=None, iter=None):
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

    if Adj.shape[0] > 1:
        if idx_closest is not None:
            L = w2l(Adj, idx_closest, iter = iter)
        else:
            L = w2l(Adj, iter = iter)

        # L is normalized by the way it's build
        D, GFT = np.linalg.eigh(L) # D eigen values and GFT eigenvectors
        idxSorted = np.argsort(D)      # Order of the eigenvalues. # np.abs(D) 
        GFT = GFT[:,idxSorted]         # GFT ordered by eigenvalues order first less
        for i in range(GFT.shape[1]):
            if GFT[0,i] < 0:
                GFT[:,i] =  GFT[:,i]*(-1) 
        # GFT[:,0] = np.abs(GFT[:,0])
        # GFT = GFT.T         # Because the matrix that do the transform is this one
        Gfreq = np.sort(D)

        Gfreq[0] = np.abs(Gfreq[0])
        
        Ahat = np.matmul(GFT.T, A)      # @ is a shortcut for matmul, yet i dont like it
        if np.iscomplexobj(Ahat) and iter is not None:
            print(f"This is the block that has complex values {iter}")

    else:  # 1D-case, DC only
        GFT = np.array([1.0])
        Gfreq = np.array([0.0])
        Ahat = np.matmul(GFT.T, A)

    return GFT, Gfreq, Ahat


def compute_iGFT_noQ(Adj, Ahat_val, idx_closest=None):
    if idx_closest is not None:
        L = w2l(Adj, idx_closest)
    else:
        L = w2l(Adj)

    D, GFT = np.linalg.eig(L) # D eigen values and GFT eigenvectors
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
    #GFT = GFT.T
    Gfreq = np.abs(np.diag(D))
    Gfreq[0] = np.abs(Gfreq[0])

    return GFT, Gfreq


if __name__ == "__main__":
    from create import *
    indexes = get_block_indexes(V,4)
    V = np.load('V_longdress.npy')
    C_rgb = np.load('C_longdress.npy')
    W, edge = compute_graph_MSR(V)
    GFT, Gfreq, Ahat = compute_GFT(W)
    Ahat_2 = np.load('Coeff_quant.npy')

    # Squared diff
    SQQ_b4 = (Ahat - Ahat_2)**2
    # SQQ_b8 = (Coeff_b8_edu - Coeff_b8_simon)**2
    # SQQ_b16 = (Coeff_b16_edu - Coeff_b16_simon)**2

    # MSE
    MSE_b4 = (SQQ_b4).mean(axis=1)
    MSE_b8 = (SQQ_b8).mean(axis=1)
    MSE_b16 = (SQQ_b16).mean(axis=1)
