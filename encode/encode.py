import numpy as np
import matplotlib as plt
import os
import rlgr

def sort_gft_coeffs(Ahat,indexes,qstep, plot=False):
    N = Ahat[:,0].shape[0]
    mask_lo = np.zeros((N), dtype=bool)
    for i,start_end_tuple in enumerate(indexes):
        # This implies that the Ahat is sorted by coefficient
        if i < 6:
            print(start_end_tuple[0])
        mask_lo[start_end_tuple[0]] = True
    mask_hi = np.logical_not(mask_lo)

    Ahat_lo = Ahat[mask_lo, :]  # DC values
    Ahat_hi = Ahat[mask_hi, :]  # "high" pass values
    
    #print(f"Size checkers {mask_hi.shape, mask_lo.shape,Ahat.shape}")
    #print(f"Number of points {np.sum(mask_hi),np.sum(mask_lo),np.sum(mask_hi)+np.sum(mask_lo)}")
    # Concatenate
    Ahat_sort = np.concatenate((Ahat_lo, Ahat_hi))
    
    if plot:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(Ahat_lo[:, 0], Ahat_lo[:, 1], label='Ahat_lo', alpha=0.5, color='blue')
        plt.scatter(Ahat_hi[:, 0], Ahat_hi[:, 1], label='Ahat_hi', alpha=0.5, color='red')
        plt.title(f'Distribution of Ahat_lo and Ahat_hi for qstep = {qstep} ')
        plt.xlabel('First Coefficient')
        plt.ylabel('Second Coefficient')
        plt.legend()
        plt.grid()
        plt.show()
    
    return Ahat_sort

def encode_rlgr(data,filename="test.bin",is_signed=1):
    if os.path.isfile(filename):
        os.remove(filename)
    #np.uint is unsigned int, the data is in signed fashion. Also 8bits may be low for representation
    data = data.astype(np.int64)
    do_write = 1
    enc = rlgr.file(filename, do_write)

    # Write data
    enc.rlgrWrite(data, is_signed)
    enc.close()
    numbits = os.path.getsize(filename) * 8
    return numbits