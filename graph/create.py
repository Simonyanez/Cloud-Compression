import numpy as np


def get_block_indexes(V, bsize):    

    if not np.log2(bsize).is_integer():
        raise ValueError

    # Quantize coordinates
    V_coarse = np.floor(V/bsize)*bsize

    # Find block limits
    variation = np.sum(np.abs(V_coarse[1:, :] - V_coarse[0:-1, :]), axis=1)
    variation = np.concatenate(([1], variation))

    idx_start = np.argwhere(variation)
    idx_stop = np.row_stack((idx_start[1:], [V.shape[0]]))

    idx_start = idx_start.flatten()
    idx_stop = idx_stop.flatten()

    indexes = list(zip(idx_start,idx_stop))
    return indexes

def get_block_npoints(indexes,iter):
    index = indexes[iter]
    npoints = index[1] - index[0]
    return npoints


def complete_graph(V):
    """
    Computes a complete graph with edge weights 1/sqrt(distance(vi, vj)).
    
    Parameters:
        V (numpy.ndarray): nx3 array. n points, where vi is the i-th row of V.
        
    Returns:
        numpy.ndarray: Weight matrix representing the complete graph.
    """
    N = V.shape[0]  # Number of points

    # Compute Euclidean Distance Matrix (EDM)
    squared_norms = np.sum(V**2, axis=1)  # Squared norms of each point
    D = np.sqrt(np.maximum(0, np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T)))

    # Avoid division by zero: set diagonal elements of D to a small value (so we don't divide by zero)
    np.fill_diagonal(D, np.nan)  # We don't want to divide by zero for the diagonal, set them to NaN

    # Compute inverse distances (edge weights)
    iD = np.power(D, -1)  # Equivalent to D.^(-1) in MATLAB

    # Replace NaN values with zeros (to handle the division by zero)
    iD = np.nan_to_num(iD, nan=0.0)  # NaN values become 0, so no NaN in the weight matrix

    # Construct weight matrix by adding transpose of inverse distances
    W = iD.T + iD

    return W

def compute_graph_MSR(V, th=None):
    """
    Compute distance-based graph from Zhang et al., ICIP 2014.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        th (float): Threshold to construct the graph (optional).

    Returns:
        numpy.ndarray: Weight matrix representing the graph.
        numpy.ndarray: Edge list.
    """
    N = V.shape[0]

    if th is None:
        th = np.sqrt(3) + 0.00001

    # Compute Euclidean Distance Matrix (EDM)
    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))
    iD = np.zeros_like(D) 
    non_zero_mask = (D > 0) & (D <= th)
    iD[non_zero_mask] = 1 / D[non_zero_mask]
    iD[np.where(D > th)] = 0
    iD[np.where(D == 0)] = 0
    W = iD.T + iD

    idx = np.nonzero(iD)
    #I, J = np.unravel_index(idx, D.shape)

    edge = np.column_stack(( idx[1], idx[0]))

    return W, edge


def compute_graph_sl(V, distance_vectors, weights, th=None):
    """
    Compute distance-based graph from Zhang et al., ICIP 2014.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        distance_vectors (numpy.ndarray): Distance vectors.
        weights (numpy.ndarray): Weights.
        th (float): Threshold to construct the graph (optional).

    Returns:
        numpy.ndarray: Weight matrix representing the graph.
        numpy.ndarray: Edge list.
        numpy.ndarray: Degree vector.
        numpy.ndarray: Inverse distance matrix.
        numpy.ndarray: Indices of the closest points along the mean direction.
    """
    N = V.shape[0]

    if th is None:
        th = np.sqrt(3) + 0.00001

    # Find the mean of the Cloud 
    mean_X = np.mean(V[:, 0])
    mean_Y = np.mean(V[:, 1])
    mean_Z = np.mean(V[:, 2])
    mean_point_cloud = np.array([mean_X, mean_Y, mean_Z])

    # Center de Cloud by its mean value
    centered_V = V - mean_point_cloud
    
    # Mean direction normalized over all color change directions
    mean_direction = np.sum(np.dot(weights,distance_vectors)) / np.sum(weights)
    mean_direction /= np.linalg.norm(mean_direction)

    # Standard weights calculation
    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))
    iD = np.zeros_like(D) 
    non_zero_mask = (D > 0) & (D <= th)
    iD[non_zero_mask] = 1 / D[non_zero_mask]
    iD[np.where(D > th)] = 0
    iD[np.where(D == 0)] = 0

    # Find degree property of nodes for ev
    degrees = np.sum(iD, axis=1)
    degrees /= np.linalg.norm(degrees)

    # Find the 20% less degree indexs (edges)
    sorted_indices = np.argsort(degrees)
    first_threshold = int(0.2 * len(sorted_indices))
    idx_closest_original = sorted_indices[:first_threshold]

    # The selected vectors dot product to the mean direction
    selected_vectors = centered_V[idx_closest_original]
    dot_products_degreed = np.dot(selected_vectors, mean_direction)

    # Use the new indices to reorder the original indexes (start edge for added weight)
    second_threshold = int(0.2 * len(dot_products_degreed))
    idx_closest = np.argsort(dot_products_degreed)[:second_threshold]
    idx_closest = idx_closest_original[idx_closest[0]] if second_threshold > 0 else None

    W = iD.T + iD

    idx = np.nonzero(iD)
    I, J = np.unravel_index(idx, D.shape)
    edge = np.column_stack((I, J))

    return W, edge, idx_closest #,degrees, iD, idx_closest

def compute_graph_unit(V,th=None):

    """
    Compute distance-based graph from Zhang et al., ICIP 2014.

    Parameters:
        V (numpy.ndarray): nx3 array. n points.
        th (float): Threshold to construct the graph (optional).

    Returns:
        numpy.ndarray: Weight matrix representing the graph.
        numpy.ndarray: Edge list.
    """
    N = V.shape[0]

    if th is None:
        th = np.sqrt(3) + 0.00001

    # Compute Euclidean Distance Matrix (EDM)
    squared_norms = np.sum(V**2, axis=1)
    D = np.sqrt(np.tile(squared_norms, (N, 1)) + np.tile(squared_norms[:, np.newaxis], (1, N)) - 2 * np.dot(V, V.T))

    iD = np.zeros_like(D) 
    non_zero_mask = (D > 0) & (D <= th)
    iD[non_zero_mask] = 1 #/ D[non_zero_mask]
    iD[np.where(D > th)] = 0
    iD[np.where(D == 0)] = 0

    W = iD.T + iD

    idx = np.nonzero(iD)
    
    

    edge = np.column_stack((I, J))

    return W, edge

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import os
    main_path = os.getcwd()
    sys.path.append(main_path)
    import utils.color as clr
    import graph.transforms as cr
    import scipy.io as sio

    def mean_square_error(A, B, absolute = False, just_DC = False, mean_AC=False):
        if absolute:
            A = np.abs(A)
            B = np.abs(B)

        if just_DC:
            A = A[0,:]
            B = B[0,:]

        new_A = np.zeros((2,3))
        new_B = np.zeros((2,3))
        if mean_AC and A.shape[0] > 1:
            new_A[0,:] = A[0,:]
            new_A[1,:] = np.mean(A[1:,:], axis=0)
            new_B[0,:] = B[0,:]
            new_B[1,:] = np.mean(B[1:,:], axis=0)
            return np.mean((new_A - new_B) ** 2, axis=0)
        return np.mean((A - B) ** 2, axis=0)
    

    def whole_test(compute = True, absolute = False, just_DC = False, mean_AC = False):
        V = np.load('V_longdress.npy')
        C_rgb = np.load('C_longdress.npy')
        A = clr.RGBtoYUV(C_rgb)
        indexes = get_block_indexes(V,8)
        total_points = 0
        one_points = []
        mat_number_of_points = sio.loadmat('matlab_code/number_of_points.mat')
        block_data_mat = sio.loadmat('matlab_code/block_data.mat')
        matlab_transform_data = sio.loadmat('matlab_code/tranform_data.mat')
        mse_results = np.zeros((len(indexes),3),dtype=np.float64)
        if compute:
            Coeff = np.zeros(C_rgb.shape, dtype=np.float64)
        else:
            Coeff = np.load('python_Coeff.npy')
        for iter in range(len(indexes)):
            index = indexes[iter]
            start_idx, end_idx = index[0], index[1]
            # Number of points check
            npoints = get_block_npoints(indexes,iter)
            if npoints == 1 or npoints == 0:
                one_points.append(iter)
            if npoints != mat_number_of_points['ni'][iter]:
                print(f"Block Nº {iter+1} has different number of points than matlab")
                break        
            total_points += npoints
            #print(f"Block Nº {iter+1} Nº Points {npoints}")
            Vblock = V[start_idx:end_idx]
            Ablock = A[start_idx:end_idx]
            
            W,edge = compute_graph_MSR(Vblock)
            # Edges check
            if npoints != 1:
                
                
                mat_edges = block_data_mat['block_data_save'][0][iter][0][1][:]    
                mat_edges_python_indexing = mat_edges
                if mat_edges.shape[0] != 0:
                    mat_edges_python_indexing = mat_edges + np.array([[-1,-1]])
                if not np.array_equal(edge, mat_edges_python_indexing):
                    print(f"Block number {iter} has mitmatch with matlab")
                    print(f"Mat edges {mat_edges_python_indexing, mat_edges_python_indexing.shape}")
                    print(f"Python edges {edge, edge.shape}")

            # GFT check
                # Python execution
            if compute == True:
                GFT, Gfreq, Ablockhat = cr.iterative_GFT(W, Ablock, Vblock, iter)
                Coeff[start_idx:end_idx,:] = Ablockhat
            else:
                Ablockhat = Coeff[start_idx:end_idx,:]
            # Matlab execution
            Ahat_mat = matlab_transform_data['transform_data'][0][iter][0][0][:]
            Gfreq_mat = matlab_transform_data['transform_data'][0][iter][0][1][:]
            GFT_mat = matlab_transform_data['transform_data'][0][iter][0][2][:]
            weights_mat = matlab_transform_data['transform_data'][0][iter][0][3][:]

            

            mse_results[iter] = mean_square_error(np.round(Ahat_mat), np.round(Ablockhat), absolute, just_DC, mean_AC)
        if compute == True:
            np.save('python_Coeff.npy',Coeff)
        np.save('mse_results.npy', mse_results)
        plt.plot(list(range(len(mse_results))),np.sort(np.mean(mse_results, axis=1)))  
        plt.show()     
        # Compare the indexes from both methods
        # for i in range(min_length):
        #     if i+1 == min_length:
        #         break
        #     index = indexes[i]
        #     index_2 = indexes_2[i]
            
        #     if index[0] != index_2 or index[1] != indexes_2[i+1]-1:
        #         print(f"Old: {index}    New: {index_2,indexes_2[i+1]}")
        #     else:
        #         continue
                # print(f"Number of points of block {index[1]- index[0] + 1}")

        # npoints = get_block_npoints(indexes,4)
        # Vblock = V[indexes[4][0]:indexes[4][1]+1]
        # print(f"Vblock size {Vblock.shape} and number of points {npoints}")
        # W,edge = compute_graph_MSR(Vblock)
        # W_2, edge_2 = compute_graph_MSR_v2(Vblock)
        # print(edge.shape, edge_2.shape)

        # I = edge_2[:,0]
        # J = edge_2[:,1]

        # for k in range(len(I)):
        #     print(f"Edge: {I[k],J[k]} \n Weight {W_2[I[k],J[k]],W_2[J[k],I[k]]} \n Position {Vblock[I[k],:], Vblock[J[k],:]}")

    def minor_test():
        # Matlab
        block_data_mat = sio.loadmat('matlab_code/block_data.mat')
        matlab_transform_data = sio.loadmat('matlab_code/tranform_data.mat')
        mse_results = np.load('mse_results.npy')
        mse_all_channels_mean = np.mean(mse_results,axis=1)
        worst_block = np.argsort(mse_all_channels_mean)[-1]
        Ahat = matlab_transform_data['transform_data'][0][worst_block][0][0][:]
        Gfreq_mat = matlab_transform_data['transform_data'][0][worst_block][0][1][:]
        GFT_mat = matlab_transform_data['transform_data'][0][worst_block][0][2][:]
        weights = matlab_transform_data['transform_data'][0][worst_block][0][3][:]
        W_mat =  block_data_mat['block_data_save'][0][worst_block][0][2][:]
        Ablock_mat = block_data_mat['block_data_save'][0][worst_block][0][3][:]
        L_mat = matlab_transform_data['transform_data'][0][worst_block][0][4][:]
        print(75*"=" + f"\n Matlab Coeff: {np.round(Ahat)} \n W: {W_mat} \n GFT: {GFT_mat} \n Gfreq: {Gfreq_mat} \n L: {L_mat} \n {Ablock_mat}")

        # Python
        V = np.load('V_longdress.npy')
        C_rgb = np.load('C_longdress.npy')
        A = clr.RGBtoYUV(C_rgb)
        indexes = get_block_indexes(V,8)
        Coeff = np.load('python_Coeff.npy')
        print(f"This is the worst block {worst_block}")
        worst_indexes = indexes[worst_block]
        #Ablockhat = Coeff[worst_indexes[0]:worst_indexes[1],:]
        Vblock = V[worst_indexes[0]:worst_indexes[1],:]
        Ablock = A[worst_indexes[0]:worst_indexes[1],:]
        W,_ = compute_graph_MSR(Vblock) 
        #GFT,Gfreq,_ = cr.iterative_GFT(W,Ablock,Vblock,worst_block,debug=True)
        GFT,Gfreq,Ablockhat = cr.compute_GFT_noQ(W,Ablock)
        #GFT_new,Ablockhat_new = cr.optimize_rotation(Gfreq, GFT, Ablock)
        L = cr.w2l(W)
        print(75*"=" + f"\n Python Coeff: {np.round(Ablockhat)} \n W: {W} \n GFT: {GFT.T} \n Gfreq: {Gfreq} \n L: {L} \n Ablock {Ablock}")
        #print(f"New GFT {GFT_new.T} and New Ablockhat {Ablockhat_new} ")
        # for bad_block in np.argsort(mse_all_channels_mean)[-20:]:
        #     Gfreq_mat = matlab_transform_data['transform_data'][0][bad_block][0][1][:]
        #     print(f"Matlab Freqs {Gfreq_mat}")
        #     bad_indexes = indexes[bad_block]
        #     Vblock = V[bad_indexes[0]:bad_indexes[1],:]
        #     Ablock = A[bad_indexes[0]:bad_indexes[1],:]
        #     W,_ = compute_graph_MSR(Vblock) 
        #     #GFT,Gfreq,_ = cr.iterative_GFT(W,Ablock,Vblock,worst_block,debug=True)
        #     GFT,Gfreq,Ablockhat = cr.compute_GFT_noQ_v2(W,Ablock)
        #     print(f"Python Freqs {Gfreq}")
        print(mean_square_error(np.round(Ablockhat), np.round(Ahat), absolute=True, mean_AC=True))

        
        print(f" This many blocks have mean mse of all channels greater than 10: {np.sum(mse_all_channels_mean > 10)}")
        
        # print(75*"=" + f"\n {Gfreq}")
        # print(75*"=" + f"\n {GFT}")
        # print(75*"=" + f"\n {weights}")

    whole_test(compute=True, absolute=True, just_DC=False, mean_AC=True)
    
    minor_test()

# NOTE: DIFFERENT AND NOT CONSTANT HIFREQ SIGNS (THINK IS NORMAL)
# NOTE: DC VALUES ARE EQUAL