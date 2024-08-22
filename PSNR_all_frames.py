from src.encoder import *
import matplotlib.pyplot as plt
#import cupy as cp
from utils.color import YUVtoRGB
import os
import pandas as pd
import rlgr
def calculate_norm(Y,Coeff_quant,N,qstep):
        # Extract the first column of Coeff_quant
    Coeff_dequant = Coeff_quant*qstep
    Coeff_dequant_col1 = Coeff_dequant[:, 0]

    # Calculate the norm (Euclidean distance) between Y and Coeff_quant_col1
    norm_value = np.linalg.norm(Y - Coeff_dequant_col1)
    value = (norm_value ** 2) / (N * 255 ** 2)
    return value

def calculate_psnr(inner_value):
    # Calculate PSNR
    psnr_Y = -10 * np.log10(inner_value)
    
    return psnr_Y

def get_coefficients(V,C_rgb,block_size,self_loop_weight,number_of_points=2,point_fraction=None):
    directional_encoder = DirectionalEncoder(V,C_rgb)
    directional_encoder.block_indexes(block_size = block_size) 
    indexes = directional_encoder.indexes
    Coeff = np.zeros(C_rgb.shape)
    nCoeff = np.zeros(C_rgb.shape)
    dCoeff = np.zeros(C_rgb.shape)
    N = V.shape[0]
    count = 0
    dynamic_overhead_estimate = 0
    overhead_estimate = 0
    decision_bs = len(indexes)
    for iteration,start_end_tuple in enumerate(indexes):
        # NOTE: Original implementation avoid one points blocks
        
        sorted_nodes = directional_encoder.simple_direction_sort(iteration)
        if point_fraction is not None:
            choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*point_fraction)]
        else:
            choosed_positions = sorted_nodes[:number_of_points]
        W,edges = directional_encoder.structural_graph(iteration)
        choosed_weights = [self_loop_weight]*len(choosed_positions)
        
        idx_map = dict(zip(choosed_positions,choosed_weights))
        #Ablockhat,block_decision = directional_encoder.dynamic_transform(iteration,W,idx_map)
        #decision.append(block_decision)
        GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(iteration,W,idx_map)
        nGFT, nGfreq, nAblockhat = directional_encoder.gft_transform(iteration,W,None)      
        Ablockconstructed = np.zeros(Ablockhat.shape)
        # Get Y components
        Coeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
        nCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
        k_choosed_pos = len(choosed_positions)
        N_block = Ablockhat[:,0].shape[0]
        overhead_estimate += k_choosed_pos*np.log2(N_block)
        if abs(Ablockhat[0,0]) > abs(nAblockhat[0,0]):
            # We get coefficients for the Y channel
            Ablockconstructed[:,0] = Ablockhat[:,0]
            # Different coefficients for the U and V channels
            Ablockconstructed[:,1:2] = nAblockhat[:,1:2]
            dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockconstructed
            dynamic_overhead_estimate += k_choosed_pos*np.log2(N_block)
            count+=1

        else:
            dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
        

    print(f"{count} blocks used adaptative method in this iteration representing {count*100/len(indexes)} % of total")
    return Coeff,nCoeff,dCoeff,indexes,count,decision_bs

def sort_gft_coeffs(Ahat,indexes):
    N = Ahat[:,0].shape[0]
    mask_lo = np.zeros((N), dtype=bool)
    for start_end_tuple in indexes:
        mask_lo[start_end_tuple[0]] = True
    mask_hi = np.logical_not(mask_lo)

    Ahat_lo = Ahat[mask_lo, :]  # DC values
    Ahat_hi = Ahat[mask_hi, :]  # "high" pass values
    # Concatenate
    Ahat_sort = np.concatenate((Ahat_lo, Ahat_hi))
    return Ahat_sort

def encode_rlgr(data,filename="test.bin",is_signed=1):
    if os.path.isfile(filename):
        os.remove(filename)
    data = data.astype(np.uint8).tolist()
    do_write = 1
    enc = rlgr.file(filename, do_write)

    # Write data
    enc.rlgrWrite(data, is_signed)
    enc.close()
    numbits = os.path.getsize(filename) * 8
    return numbits

def code_YUV(Coeff_quant_sorted,bitstream_directory = ''):
    
    # Code Y, U, V separately 
    numbits_Y = encode_rlgr(Coeff_quant_sorted[:, 0], os.path.join(bitstream_directory, 'bitstream_Y.bin'))
    numbits_U = encode_rlgr(Coeff_quant_sorted[:, 1], os.path.join(bitstream_directory, 'bitstream_U.bin'))
    numbits_V = encode_rlgr(Coeff_quant_sorted[:, 2], os.path.join(bitstream_directory, 'bitstream_V.bin'))

    # Bit count
    bs_size = numbits_Y + numbits_U + numbits_V
    
    return bs_size

def quantize_PSNR_bs(Coeff,nCoeff,dCoeff,qstep,indexes):
    Y = Coeff[:,0]
    nY = nCoeff[:,0]
    dY = dCoeff[:,0]
    N = Y.shape[0]
    # Quantize
    Coeff_quant = np.round(Coeff/qstep)
    nCoeff_quant = np.round(nCoeff/qstep)
    dCoeff_quant = np.round(dCoeff/qstep)
    # PSNR
    Frame_val = calculate_norm(Y,Coeff_quant,N,qstep)
    nFrame_val = calculate_norm(nY,nCoeff_quant,N,qstep)
    dFrame_val = calculate_norm(dY,dCoeff_quant,N,qstep)
    # Sort hi to low
    Coeff_quant_sorted = sort_gft_coeffs(Coeff_quant,indexes)
    nCoeff_quant_sorted = sort_gft_coeffs(nCoeff_quant,indexes)
    dCoeff_quant_sorted = sort_gft_coeffs(dCoeff_quant,indexes)
    # Run-Length Golomb-Rice
    bs_Coeffs = code_YUV(Coeff_quant_sorted, bitstream_directory='res')
    bs_nCoeffs = code_YUV(nCoeff_quant_sorted, bitstream_directory='res')
    bs_dCoeffs = code_YUV(dCoeff_quant_sorted, bitstream_directory='res')
    return Frame_val,bs_Coeffs,nFrame_val,bs_nCoeffs, dFrame_val,bs_dCoeffs 

if __name__ == "__main__":
    base_path = "/home/simao/Documents/longdress/longdress/Ply"
    ply_list = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.ply')]

    print(f"Found {len(ply_list)} PLY files.")

    num_of_points = 16
    weight = 1.2
    steps = [8, 12, 16, 20, 26, 32, 40, 52, 64, 78, 90, 128, 256]
    block_sizes = [4, 8, 16]

    bits_all_frames = {4: [0]*len(steps), 8: [0]*len(steps), 16: [0]*len(steps)}
    nbits_all_frames = {4: [0]*len(steps), 8: [0]*len(steps), 16: [0]*len(steps)}
    dbits_all_frames = {4: [0]*len(steps), 8: [0]*len(steps), 16: [0]*len(steps)}
    innerpsnr_all_frames = {4: 0, 8: 0, 16: 0}
    ninnerpsnr_all_frames = {4: 0, 8: 0, 16: 0}
    dinnerpsnr_all_frames = {4: 0, 8: 0, 16: 0}
    N_tot = 0
    T = len(ply_list[:10])

    for idx, ply_file in enumerate(ply_list[:10]):
        print(f"Processing file {idx + 1}/{T}: {ply_file}")
        V, C_rgb, _ = ply.ply_read8i(ply_file)
        #V = cp.asarray(V)
        #C_rgb = cp.asarray(C_rgb)
        N = V.shape[0]
        N_tot += N
        
        print(f"  Number of points in current frame: {N}")
        
        for bsize in block_sizes:
            print(f"  Processing block size {bsize}...")
            Coeff, nCoeff, dCoeff, indexes, count, decision_estimate = get_coefficients(
                V=V, C_rgb=C_rgb, block_size=bsize, self_loop_weight=weight, number_of_points=num_of_points
            )
            
            print(f"    Decision Estimate: {decision_estimate}")

            for i, step in enumerate(steps):
                Frame_val, bs_Coeffs, nFrame_val, bs_nCoeffs, dFrame_val, bs_dCoeffs = quantize_PSNR_bs(
                    Coeff, nCoeff, dCoeff, step, indexes
                )
                bits_all_frames[bsize][i] += bs_Coeffs
                nbits_all_frames[bsize][i] += bs_nCoeffs
                dbits_all_frames[bsize][i] += (bs_dCoeffs + decision_estimate)
                innerpsnr_all_frames[bsize] += Frame_val
                ninnerpsnr_all_frames[bsize] += nFrame_val
                dinnerpsnr_all_frames[bsize] += dFrame_val

                if i % (len(steps) // 4) == 0:
                    print(f"      Step {step}:")
                    print(f"        Bits: {bs_Coeffs}, N Bits: {bs_nCoeffs}, D Bits: {bs_dCoeffs}")
                    print(f"        Inner Value: {Frame_val}, N Inner Value: {nFrame_val}, D Inner Value: {dFrame_val}")
    
    print("Calculating averages...")
    avg_bits_all_frames = {bsize: [x / N_tot for x in bits_all_frames[bsize]] for bsize in block_sizes}
    avg_nbits_all_frames = {bsize: [x / N_tot for x in nbits_all_frames[bsize]] for bsize in block_sizes}
    avg_dbits_all_frames = {bsize: [x / N_tot for x in dbits_all_frames[bsize]] for bsize in block_sizes}
    avg_innerpsnr_all_frames = {bsize: x / T for bsize, x in innerpsnr_all_frames.items()}
    avg_ninnerpsnr_all_frames = {bsize: x / T for bsize, x in ninnerpsnr_all_frames.items()}
    avg_dinnerpsnr_all_frames = {bsize: x / T for bsize, x in dinnerpsnr_all_frames.items()}
    
    print("Calculating PSNR values...")
    psnr_all = {}
    npsnr_all = {}
    dpsnr_all = {}
    for bsize in block_sizes:
        psnr_all[bsize] = calculate_psnr([avg_innerpsnr_all_frames[bsize]])
        npsnr_all[bsize] = calculate_psnr([avg_ninnerpsnr_all_frames[bsize]])
        dpsnr_all[bsize] = calculate_psnr([avg_dinnerpsnr_all_frames[bsize]])
    
    print("Processing complete.")
    
    # Convert results to DataFrame
    data = {
        'Block Size': [],
        'Step': [],
        'Average Bits': [],
        'Average N Bits': [],
        'Average D Bits': [],
        'Average Inner PSNR': [],
        'Average N Inner PSNR': [],
        'Average D Inner PSNR': [],
        'PSNR All': [],
        'N PSNR All': [],
        'D PSNR All': []
    }

    for bsize in block_sizes:
        for i, step in enumerate(steps):
            data['Block Size'].append(bsize)
            data['Step'].append(step)
            data['Average Bits'].append(avg_bits_all_frames[bsize][i])
            data['Average N Bits'].append(avg_nbits_all_frames[bsize][i])
            data['Average D Bits'].append(avg_dbits_all_frames[bsize][i])
            data['Average Inner PSNR'].append(avg_innerpsnr_all_frames[bsize])
            data['Average N Inner PSNR'].append(avg_ninnerpsnr_all_frames[bsize])
            data['Average D Inner PSNR'].append(avg_dinnerpsnr_all_frames[bsize])
            data['PSNR All'].append(psnr_all[bsize])
            data['N PSNR All'].append(npsnr_all[bsize])
            data['D PSNR All'].append(dpsnr_all[bsize])
    
    df = pd.DataFrame(data)
    df.to_csv('PSNR_all_frames.csv', index=False)
    print("Results saved to 'PSNR_all_frames.csv'.")