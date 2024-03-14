import numpy as np

def w2l(W, idx_closest=None):
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

    if idx_closest is not None:
        C[np.ix_(idx_closest, idx_closest)] = 2

    L = C + np.diag(np.sum(W, axis=1)) - W + np.diag(np.diag(W))

    return L
