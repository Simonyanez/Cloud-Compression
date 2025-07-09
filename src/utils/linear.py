import numpy as np

def is_linearly_independent(v1, v2):
    """Check if two vectors are linearly independent."""
    if np.all(v1 == 0) or np.all(v2 == 0):
        return False
    return not np.allclose(v1 / v1[np.nonzero(v1)[0][0]], v2 / v2[np.nonzero(v2)[0][0]])
