import numpy as np
import matplotlib.pyplot as plt

def direction(V, C, aspect_ratio=None):
    """
    Compute direction vectors between points in a point cloud.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        C (numpy.ndarray): Color information corresponding to each point in V.
        aspect_ratio (tuple): Aspect ratio for plotting (optional).

    Returns:
        numpy.ndarray: Direction vectors.
        numpy.ndarray: Edge list.
        numpy.ndarray: Distance vectors.
        numpy.ndarray: Weights.
    """
    N = V.shape[0]

    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))

    th = np.sqrt(3) + 0.00001

    iD = 1 / D
    iD[D > th] = 0
    iD[D == 0] = 0

    idx = np.nonzero(iD)
    I, J = np.unravel_index(idx, D.shape)
    edge = np.column_stack((I, J))

    YUV_block_double = C.astype(float) / 256
    c_len = C.shape[0]
    G_vec = np.zeros((c_len, c_len))
    for id in range(len(J)):
        i = I[id]
        j = J[id]
        G_vec[i, j] = YUV_block_double[j, 0] - YUV_block_double[i, 0]

    distance_vectors = np.zeros((G_vec.shape[0], 3))
    distance_indexes = np.zeros((G_vec.shape[0], 2))
    weights = np.zeros(G_vec.shape[0])

    for iter in range(G_vec.shape[0]):
        min_val = np.min(G_vec[:, iter])
        min_index = np.argmin(G_vec[:, iter])
        if np.abs(min_val) > 0.02:
            distance_indexes[iter, :] = [min_index, iter]
            weights[iter] = np.abs(G_vec[min_index, iter])
            dis_vec = V[iter, :] - V[min_index, :]
        else:
            dis_vec = np.array([0, 0, 0])
        
        if np.linalg.norm(dis_vec) != 0:
            dis_vec = dis_vec / np.linalg.norm(dis_vec)
            distance_vectors[iter, :] = dis_vec
        else:
            distance_vectors[iter, :] = dis_vec

    if aspect_ratio is not None:
        x = V[:, 0]
        y = V[:, 1]
        z = V[:, 2]
        u = distance_vectors[:, 0]
        v = distance_vectors[:, 1]
        w = distance_vectors[:, 2]
        magnitudes = np.sqrt(u**2 + v**2 + w**2)
        u_unit = u / magnitudes
        v_unit = v / magnitudes
        w_unit = w / magnitudes
        r = np.mean(distance_vectors, axis=0)
        r = r / np.linalg.norm(r)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u_unit, v_unit, w_unit)
        ax.quiver(0, 0, 0, r[0], r[1], r[2], color='r', linewidth=6, arrow_length_ratio=0.1)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Unit Vectors in 3D')
        ax.grid(True)
        ax.set_box_aspect(aspect_ratio)
        ax.set_xlim([np.min(x) - 1, np.max(x) + 1])
        ax.set_ylim([np.min(y) - 1, np.max(y) + 1])
        ax.set_zlim([np.min(z) - 1, np.max(z) + 1])
        plt.show()

    return G_vec, edge, distance_vectors, weights
