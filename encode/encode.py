import numpy as np
import matplotlib.pyplot as plt
import os
import rlgr

class ADEncoder():
    def __init__(self, Coeffs: np.ndarray):
        self.Coeffs = Coeffs

    def _quantize(self, qstep: int) -> np.ndarray:
        return np.round(self.Coeffs/qstep)

    def _sort_coeffs(self, PointCloud: ADPointCloud) -> np.ndarray:
        ADPointCloud._sort_coeffs()
        pass









def sort_gft_coeffs(Ahat,indexes,qstep, plot=False, debug=False):
    N = Ahat[:,0].shape[0]
    mask_lo = np.zeros((N), dtype=bool)
    bad_blocks = []
    for i,start_end_tuple in enumerate(indexes):
        # This implies that the Ahat is sorted by coefficient
        if i < 6:
            print(start_end_tuple[0])
        mask_lo[start_end_tuple[0]] = True

        if debug:
            Asubhat_hi = Ahat[start_end_tuple[0],:]
            Asubhat_lo = Ahat[start_end_tuple[0]+1:start_end_tuple[1],:]
            min_len = Asubhat_lo.shape[0] > 0 
            if min_len:
                hi_val = np.max(Asubhat_lo[:,0],) >= Asubhat_hi[0]
                if min_len & hi_val:
                    print(f"This is the block number and tuple {i,start_end_tuple}")
                    bad_blocks.append(i)
                    
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
    
    if debug:
        print(f"Number of blocks with wrong behaviour {len(bad_blocks)}/{len(indexes)}")
        return Ahat_sort, bad_blocks
    
    return Ahat_sort

def code_YUV(Coeff_quant_sorted,bitstream_directory = '', plot=False):
    if plot:
        plt.figure(figsize=(10,6))
        plt.hist(np.abs(Coeff_quant_sorted[:,0]),bins=int(np.max(np.abs(Coeff_quant_sorted[:,0]))))
        plt.title(f'Density distribution of Absolute Cuofficients in the Y Channel')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.yscale('log')
        plt.grid(True,which="both", ls="-")
        plt.show()

    # Code Y, U, V separately 
    numbits_Y = encode_rlgr(Coeff_quant_sorted[:, 0], os.path.join(bitstream_directory, 'bitstream_Y.bin'))
    numbits_U = encode_rlgr(Coeff_quant_sorted[:, 1], os.path.join(bitstream_directory, 'bitstream_U.bin'))
    numbits_V = encode_rlgr(Coeff_quant_sorted[:, 2], os.path.join(bitstream_directory, 'bitstream_V.bin'))

    # Bit count
    bs_size = numbits_Y + numbits_U + numbits_V
    
    return bs_size

def decode_YUV(N,bitstream_directory = ''):
    Coeff = np.zeros((N,3))
    Coeff[:,0] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_Y.bin'), N=N)
    Coeff[:,1] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_U.bin'), N=N)
    Coeff[:,2] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_V.bin'), N=N)
    return Coeff

def encode_rlgr(data,filename="test.bin",is_signed=1):
    if os.path.isfile(filename):
        os.remove(filename)
    #np.uint is unsigned int, the data is in signed fashion. Also 8bits may be low for representation
    data = data.astype(np.int16)
    do_write = 1
    enc = rlgr.file(filename, do_write)

    # Write data
    enc.rlgrWrite(data, is_signed)
    enc.close()
    numbits = os.path.getsize(filename) * 8
    return numbits

def decode_rlgr(filename,N, is_signed=1):
    do_write = 0
    dec = rlgr.file(filename,do_write)
    Coeff = dec.rlgrRead(N, is_signed)
    dec.close()
    return Coeff