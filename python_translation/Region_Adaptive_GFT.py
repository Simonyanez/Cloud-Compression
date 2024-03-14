import numpy as np

def Region_Adaptive_GFT(A, params):
    """
    This function implements the Region adaptive graph Fourier transform (RA-GFT)
    for point cloud attributes of voxelized point clouds.

    Args:
    A (ndarray): Attribute matrix.
    params (dict): Parameters including 'V' for point cloud coordinates, 'bsize' for block size,
                   'J' for depth of octree, and 'isMultiLevel' for multilevel flag.

    Returns:
    tuple: Tuple containing Ahat, freqs, and weights.
    """
    V = params['V']
    b = params['bsize']
    J = params['J']
    isMultiLevel = params['isMultiLevel']
    N = V.shape[0]

    if isinstance(b, int) or isinstance(b, float):
        if isMultiLevel:
            base_bsize = np.log2(b)
            if np.floor(base_bsize) != base_bsize:
                raise ValueError('block size bsize should be a power of 2')
            L = int(J / base_bsize)
            if L != np.floor(L):
                raise ValueError('block size does not match number of levels')
            bsize = np.ones(L) * b
        else:
            base_bsize = np.log2(b)
            if np.floor(base_bsize) != base_bsize:
                raise ValueError('block size bsize should be a power of 2')
            L = 1
            bsize = np.array([b])
    else:
        bsize = np.array(b)
        L = len(bsize)
        base_bsize = np.log2(bsize)
        if not np.all(base_bsize == np.floor(base_bsize)):
            raise ValueError('entries of block size should be a power of 2')
        if np.sum(base_bsize) > J:
            raise ValueError('block sizes do not match octree depth J')

    Ahat = []
    Vcurr = V
    Acurr = A
    Qin = np.ones(N)
    Gfreq_curr = np.zeros(N)
    freqs = []
    weights = []

    for level in range(L, 0, -1):
        start_indices = block_indices(Vcurr, bsize[level - 1])
        Nlevel = Vcurr.shape[0]
        end_indices = np.append(start_indices[1:] - 1, Nlevel)

        ni = end_indices - start_indices + 1
        to_change = np.where(ni != 1)[0]

        Acurr_hat = np.copy(Acurr)
        Qout = Qin
        Gfreq_curr = np.zeros_like(Qin)

        for currblock in to_change:
            first_point = start_indices[currblock]
            last_point = end_indices[currblock]
            Vblock = Vcurr[first_point:last_point]
            Qin_block = Qin[first_point:last_point]
            Ablock = Acurr[first_point:last_point]

            Ahatblock, Gfreq_block, weights_block = block_coeffs(Vblock, Ablock, Qin_block, bsize[level - 1])
            Acurr_hat[first_point:last_point] = Ahatblock
            Qout[first_point:last_point] = weights_block
            Gfreq_curr[first_point:last_point] = Gfreq_block

        Vcurr = np.floor(Vcurr[start_indices] / bsize[level - 1])
        Acurr = Acurr_hat[start_indices]
        Qin = Qout[start_indices]

        Coeff_high = Acurr_hat[start_indices]
        Qout_high = Qout[start_indices]
        Gfreq_high = Gfreq_curr[start_indices]

        Ahat = np.vstack((Acurr_hat[start_indices:], Ahat))
        freqs = np.hstack((Gfreq_high, freqs))
        weights = np.hstack((Qout_high, weights))

        if level == 1:
            Gfreq_low = Gfreq_curr[start_indices]
            Ahat = np.vstack((Acurr, Ahat))
            freqs = np.hstack((Gfreq_low, freqs))
            weights = np.hstack((Qin, weights))

    return Ahat, freqs, weights

def block_coeffs(Vblock, A, Q, bsize):
    W = compute_graph_MSR(Vblock)
    if bsize == 2:
        Ahat, Gfreq, weights = RAGFT_connected_graph(W, A, Q)
    else:
        p, _, r, _ = dmperm(W + np.eye(W.shape[0]))
        numConnComp = r.shape[1] - 1
        if numConnComp == 1:
            Ahat, Gfreq, weights = RAGFT_connected_graph(W, A, Q)
        else:
            Ahat, Gfreq, weights = RAGFT_disconnected_graph(W, A, Q, Vblock, numConnComp, p, r)
    return Ahat, Gfreq, weights

def RAGFT_connected_graph(W, A, Q):
    GFT, Gfreq = compute_GFT(W, Q)
    weights = np.tile(np.sum(Q), (A.shape[0], 1))
    Coeff = GFT @ A
    return Coeff, Gfreq, weights

def RAGFT_disconnected_graph(Wcurr, A, Qcurr, Vblock, numDCs, p, r):
    U = np.zeros((Wcurr.shape[0], 0))
    isDC
