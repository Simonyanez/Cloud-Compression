# Authors - Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
# Copyright Eduardo Pavez, University of Southern California
# Los Angeles, USA, 05/30/2020
# E. Pavez, B. Girault, A. Ortega, and P. A. Chou.
# "Region adaptive graph Fourier transform for 3D point clouds".
# IEEE International Conference on Image Processing (ICIP), 2020
# https://arxiv.org/abs/2003.01866

import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.linalg import block_diag

def iRegion_Adaptive_GFT(Coeff, params):
    """
    Computes the Region Adaptive Graph Fourier Transform (RA-GFT) for 3D point clouds.

    Args:
    Coeff (ndarray): Coefficients matrix.
    params (dict): Parameters dictionary.

    Returns:
    tuple: Tuple containing start indices, end indices, transformed matrix V_MR, and auxiliary coefficients.
    """

    V = params['V']
    b = params['bsize']
    J = params['J']
    isMultiLevel = params['isMultiLevel']
    N = V.shape[0]

    # Check consistency of block sizes, resolution levels, and octree depth
    if isinstance(b, int):
        if isMultiLevel:
            base_bsize = np.log2(b)
            if not np.floor(base_bsize) == base_bsize:
                raise ValueError('Block size bsize should be a power of 2')
            L = J // base_bsize
            if L != np.floor(L):
                raise ValueError('Block size does not match number of levels')
            bsize = np.ones(L) * b
        else:
            base_bsize = np.log2(b)
            if not np.floor(base_bsize) == base_bsize:
                raise ValueError('Block size bsize should be a power of 2')
            L = 1
            bsize = np.array([b])
    else:
        bsize = np.array(b)
        L = len(bsize)
        if not np.all(np.floor(np.log2(bsize)) == np.log2(bsize)):
            raise ValueError('Entries of block size should be a power of 2')
        if np.sum(np.log2(bsize)) > J:
            raise ValueError('Block sizes do not match octree depth J')

    starti = []
    Q = []
    endi = []
    V_MR = []
    Vcurr = V.copy()
    V_MR.append(Vcurr)
    Q.append(np.ones(N))
    for level in range(L-1, -1, -1):
        start_indices = block_indices(Vcurr, bsize[level])
        Nlevel = Vcurr.shape[0]
        end_indices = np.concatenate((start_indices[1:] - 1, np.array([Nlevel])))
        Vcurr = np.floor_divide(Vcurr[start_indices, :], bsize[level])
        starti.append(start_indices)
        endi.append(end_indices)
        if level > 0:
            V_MR.append(Vcurr)
            Q.append(compute_Q_lower_resolution(Q[-1], start_indices, end_indices))

    Coeff_aux = Coeff.copy()

    for level in range(L):
        start_indices = starti[level]
        end_indices = endi[level]
        level_indices = np.arange(start_indices[0], end_indices[-1] + 1)
        Nlevel = len(level_indices)

        ni = end_indices - start_indices + 1
        to_change = np.nonzero(ni != 1)[0]

        coeff_level = Coeff_aux[level_indices, :]
        coeff_level_permuted = invert_level_permutation(start_indices, coeff_level)

        Vcurr = V_MR[level]
        Qin = Q[level]

        for currblock in to_change:
            first_point = start_indices[currblock]
            last_point = end_indices[currblock]
            Vblock = Vcurr[first_point:last_point+1, :]
            Qin_block = Qin[first_point:last_point+1]
            Ahatblock = coeff_level_permuted[first_point:last_point+1, :]

            Ablock = invert_block_coeffs(Vblock, Ahatblock, Qin_block, bsize[level])
            coeff_level_permuted[first_point:last_point+1, :] = Ablock

        Coeff_aux[level_indices, :] = coeff_level_permuted

    return starti, endi, V_MR, Coeff_aux


def invert_level_permutation(start_indices, coeff_level):
    """
    Inverts the level permutation of coefficients.

    Args:
    start_indices (list): Start indices of blocks.
    coeff_level (ndarray): Coefficients at the current level.

    Returns:
    ndarray: Permuted coefficients.
    """
    coeff_level_permuted = np.zeros_like(coeff_level)

    ndc = len(start_indices)
    level_indices = np.arange(1, coeff_level.shape[0] + 1)
    level_indices_high = level_indices.copy()
    level_indices_high[start_indices] = -1
    level_indices_high = level_indices_high[level_indices_high != -1]
    coeff_level_permuted[start_indices, :] = coeff_level[:ndc, :]
    coeff_level_permuted[level_indices_high, :] = coeff_level[ndc:, :]

    return coeff_level_permuted


def compute_Q_lower_resolution(Qin, start_indices, end_indices):
    """
    Computes Q for lower resolution.

    Args:
    Qin (ndarray): Input Q.
    start_indices (list): Start indices of blocks.
    end_indices (list): End indices of blocks.

    Returns:
    ndarray: Lower resolution Q.
    """
    Qout = np.zeros_like(Qin)

    for i, starti in enumerate(start_indices):
        Qout[i] = np.sum(Qin[starti:end_indices[i] + 1])

    return Qout


def invert_block_coeffs(Vblock, Ahat, Q, bsize):
    """
    Inverts block coefficients.

    Args:
    Vblock (ndarray): Block of points.
    Ahat (ndarray): Block coefficients.
    Q (ndarray): Q matrix.
    bsize (int): Block size.

    Returns:
    ndarray: Inverted coefficients.
    """
    W = complete_graph(Vblock)
    if bsize == 2:
        A = inverse_RAGFT_connected_graph(W, Ahat, Q)
    else:
        p, _, r, _ = csr_matrix(W + np.eye(W.shape[0])).permute_generalized_nested_dissection()
        numConnComp = len(r) - 1
        if numConnComp == 1:
            A = inverse_RAGFT_connected_graph(W, Ahat, Q)
        else:
            A = inverse_RAGFT_disconnected_graph(W, Ahat, Q, Vblock, numConnComp, p, r)

    return A


def inverse_RAGFT_connected_graph(W, Ahat, Q):
    """
    Inverts coefficients for connected graph.

    Args:
    W (ndarray): Weight matrix.
    Ahat (ndarray): Coefficients matrix.
    Q (ndarray): Q matrix.

    Returns:
    ndarray: Inverted coefficients.
    """
