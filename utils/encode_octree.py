import numpy as np

def get_morton_code(V, J):
    """
    Computes the Morton code for a voxelized point cloud.

    Parameters:
    V (numpy.ndarray): Nx3 point cloud with integer coordinates.
    J (int): Octree depth.

    Returns:
    numpy.ndarray: Morton codes for the input points.
    """

    N = V.shape[0]
    M = np.zeros(N, dtype=np.uint64)
    tt = np.array([1, 2, 4], dtype=np.uint64)

    for i in range(1, J + 1):
        V_bits = (V >> (i - 1)) & 1
        V_bits = V_bits.astype(np.uint8)
        V_unpacked = np.fliplr(V_bits)
        M += np.dot(V_unpacked, tt)
        tt *= 8

    return M

def octree_byte_count(V, maxDepth):
    """
    This function computes the number of bits used by an octree representation
    of a voxelized point cloud without actually finding the octree.
    Inspired by the formula from Queiroz, Chou, Region adaptive hierarchical
    transform, IEEE TIP.

    Parameters:
    V (numpy.ndarray): The voxelized point cloud (Nx3 array).
    maxDepth (int): The maximum depth of the octree.

    Returns:
    nbits (int): The total number of bits used by the octree.
    bytes (numpy.ndarray): The number of bytes at each level.
    """

    width = 2 ** (-maxDepth)

    # First, make sure points are voxelized
    Vint = np.floor(V / width) * width
    nbits = 0
    bytes_array = np.zeros(maxDepth)

    # The code goes from leaves of the octree to the root, counting
    # the number of points at each level, and adding up bytes
    for j in range(1, maxDepth + 1):
        Vj = np.floor(Vint / (2 ** j)) * (2 ** j)
        Vj = Vj.astype(np.uint64)
        Mj = get_morton_code(Vj,maxDepth)
        bytes_array[j - 1] = np.unique(Mj, axis=0).shape[0]

    nbits = 8 * np.sum(bytes_array)

    return int(nbits), bytes_array