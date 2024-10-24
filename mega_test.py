import os
import numpy as np
from utils import ply
from src.encoder import *
from graph.create import *
from graph.transforms import * 
from encode.encode import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    V = np.load('V_longdress.npy')
    A = np.load('C_longdress.npy')

    indexes = get_block_indexes(V,4)
    indexes_2 = block_indices_v2(V,4)

    # Ensure both index lists have the same length
    min_length = min(len(indexes), len(indexes_2))

    print(min_length)
    # Compare the indexes from both methods
    for i in range(min_length):
        if i+1 == min_length:
            break
        index = indexes[i]
        index_2 = indexes_2[i]
        
        if index[0] != index_2 or index[1] != indexes_2[i+1]-1:
            print(f"Old: {index}    New: {index_2,indexes_2[i+1]}")
        else:
            continue
            # print(f"Number of points of block {index[1]- index[0] + 1}")
    # Test block
    npoints = get_block_npoints(indexes,4)
    Vblock = V[indexes[4][0]:indexes[4][1]+1]
    Ablock = A[indexes[4][0]:indexes[4][1]+1]
    print(f"Vblock size {Vblock.shape} and number of points {npoints}")
    Wblock,edge = compute_graph_MSR(Vblock)
    Wblock_2, edge_2 = compute_graph_MSR_v2(Vblock)
    print(edge.shape, edge_2.shape)

    I = edge_2[:,0]
    J = edge_2[:,1]

    for k in range(len(I)):
        print(f"Edge: {I[k],J[k]} \n Weight {Wblock_2[I[k],J[k]],Wblock_2[J[k],I[k]]} \n Position {Vblock[I[k],:], Vblock[J[k],:]}")

    GFT, Gfreqs, Ablockhat = compute_GFT_noQ(Wblock_2, Ablock)
    print(Gfreqs)
    print(GFT.T)
    plt.matshow(GFT.T, cmap = plt.cm.Blues)
    plt.show()
    
    print(Ablockhat.shape)
    print(Ablockhat)
    Coeff = np.zeros(A.shape)
    Coeff[indexes[4][0]:indexes[4][1]+1] = Ablockhat
    plt.plot(Coeff[:100])
    plt.show()
    Coeff_sort = sort_gft_coeffs(Coeff,indexes,1)
    plt.plot(Coeff_sort[:60000])
    plt.show()