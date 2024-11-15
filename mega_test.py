import os
import numpy as np
from utils import ply
from src.encoder import *
from graph.create import *
from graph.transforms import * 
from encode.encode import *
import matplotlib.pyplot as plt

def single_vis(V,A, iter):
    Vblock = V[iter[0]:iter[1]]
    Ablock = A[iter[0]:iter[1]]
    Wblock,_ = compute_graph_MSR(Vblock)
    GFT, Gfreqs, Ablockhat = compute_GFT_noQ(Wblock, Ablock)
    plt.matshow(GFT.T, cmap = plt.cm.Blues)
    plt.show()

    _, vis_fig = visual.visualization(Vblock,Ablock,1,1,"Normal")
    plt.show()
    Coeff = np.zeros(A.shape)
    Coeff[iter[0]:iter[1]] = Ablockhat
    plt.plot(Coeff[:iter[1]+100])
    plt.show()
    Coeff_sort = sort_gft_coeffs(Coeff,indexes,1)
    plt.plot(Coeff_sort[:iter[1]+100])
    plt.show()

if __name__ == "__main__":
    Ypositions_4 = np.load('res/Ypositions_4.npy')
    print(Ypositions_4.shape)
    V = np.load('V_longdress.npy')
    A = np.load('C_longdress.npy')

    indexes = get_block_indexes(V,4)
    indexes_2 = block_indices_v2(V,4)

    # Ensure both index lists have the same length
    min_length = min(len(indexes), len(indexes_2))

    # Compare the indexes from both methods
    for i in range(min_length):
        if i+1 == min_length:
            break
        index = indexes[i]
        index_2 = indexes_2[i]
        
        if index[0] != index_2 or index[1] != indexes_2[i+1]-1:
            # print(f"Old: {index}    New: {index_2,indexes_2[i+1]}")
            pass
        else:
            continue
            # print(f"Number of points of block {index[1]- index[0] + 1}")
    # Test block
    npoints = get_block_npoints(indexes,4)
    Vblock = V[indexes[4261][0]:indexes[4261][1]]
    Ablock = A[indexes[4261][0]:indexes[4261][1]]
    # print(f"Vblock size {Vblock.shape} and number of points {npoints}")
    Wblock,edge = compute_graph_MSR(Vblock)
    Wblock_2, edge_2 = compute_graph_MSR_v2(Vblock)
    # print(edge.shape, edge_2.shape)

    I = edge_2[:,0]
    J = edge_2[:,1]

    for k in range(len(I)):
        # print(f"Edge: {I[k],J[k]} \n Weight {Wblock_2[I[k],J[k]],Wblock_2[J[k],I[k]]} \n Position {Vblock[I[k],:], Vblock[J[k],:]}")
        pass

    GFT, Gfreqs, Ablockhat = compute_GFT_noQ(Wblock_2, Ablock)
    print(Gfreqs)
    print(GFT.T)
    plt.matshow(GFT.T, cmap = plt.cm.Blues)
    plt.show()
    
    print(Ablockhat.shape)
    print(Ablockhat)
    Coeff = np.zeros(A.shape)
    # Coeff[indexes[4][0]:indexes[4][1]] = Ablockhat
    plt.plot(Coeff[:60000])
    plt.show()
    # Coeff_sort = sort_gft_coeffs(Coeff,indexes,1)
    # plt.plot(Coeff_sort[:60000])
    plt.show()
    print(indexes[4][0])
    # print(indexes)

    rare_blocks = []

    # Flatten Ypositions_4 to a 1D array
    Ypositions_flat = Ypositions_4.flatten()

    # All blocks
    Coeff = np.zeros(A.shape)
    DC_positions = []
    for index in indexes:
        Vblock = V[index[0]:index[1]]
        Ablock = A[index[0]:index[1]]
        Wblock,edge = compute_graph_MSR(Vblock)
        GFT, Gfreqs, Ablockhat, DC_pos = iterative_GFT(Wblock, Ablock)
        Coeff[index[0]:index[1]] = Ablockhat
        DC_positions = DC_positions +  [sum(x) for x in zip([index[0]] * len(DC_pos), DC_pos)]
    Coeff_sort = sort_gft_coeffs(Coeff,DC_positions,1, debug=False)
    plt.plot(Coeff_sort)
    plt.show()
    # for bad_block in bad_blocks:
    #     bad_iter = indexes[bad_block]
    #     Vblock = V[bad_iter[0]:bad_iter[1]]
    #     print(f"The block number {bad_block} has wrong cuofficients")
    #     Wblock,edge = compute_graph_MSR(Vblock)
    #     print(Wblock)
    #     single_vis(V,A,bad_iter)
        

    