from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB
from utils.morton import morton3D_64_encode
from utils.encode_octree import octree_byte_count
import os
import pandas as pd
import rlgr 

def calculate_psnr(Y, Coeff_quant, N,qstep):
    # Extract the first column of Coeff_quant
    Coeff_dequant = Coeff_quant*qstep
    Coeff_dequant_Y = Coeff_dequant[:, 0]

    # Calculate the norm (Euclidean distance) between Y and Coeff_quant_Y
    norm_value = np.linalg.norm(Y - Coeff_dequant_Y)
    inner_value = (norm_value ** 2) / ((255 ** 2) * N )
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
    print(f"Number of points {N}")
    count = 0
    decision_bs = len(indexes)
    choosed_count = 0
    V_choosed = None
    for iteration,start_end_tuple in enumerate(indexes):
        # NOTE: Original implementation avoid one points blocks
        Vblock,_ = directional_encoder.get_block(iteration)    
        sorted_nodes = directional_encoder.simple_direction_sort(iteration)
        if point_fraction is not None:
            choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*point_fraction)]
        else:
            choosed_positions = sorted_nodes[:number_of_points]
        W,_ = directional_encoder.structural_graph(iteration)
        choosed_weights = [self_loop_weight]*len(choosed_positions)
        
        idx_map = dict(zip(choosed_positions,choosed_weights))
        #Ablockhat,block_decision = directional_encoder.dynamic_transform(iteration,W,idx_map)
        #decision.append(block_decision)
        _, _, Ablockhat = directional_encoder.gft_transform(iteration,W,idx_map)
        _, _, nAblockhat = directional_encoder.gft_transform(iteration,W,None)      
        Ablockconstructed = np.zeros(Ablockhat.shape)
        # Get Y coefficients 
        Coeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
        nCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
        if abs(Ablockhat[0,0]) > abs(nAblockhat[0,0]):
            choosed_count += len(choosed_positions)
            if V_choosed is None:
                V_choosed = Vblock[choosed_positions]
            else:
                V_choosed = np.concatenate([V_choosed,Vblock[choosed_positions]])
                
            # We get coefficients for the Y channel
            Ablockconstructed[:,0] = Ablockhat[:,0]
            # Different coefficients for the U and V channels
            Ablockconstructed[:,1:3] = nAblockhat[:,1:3]
            dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockconstructed
            count+=1

        else:
            dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
    V_choosed = V_choosed.astype(np.uint64)    
    octree_nbits,octree_bs = octree_byte_count(V_choosed,10)
    print(f"Octree coding total: {octree_nbits} \n Octree coding per position: {octree_nbits/choosed_count} \n Morton Code raw: {octree_bs}")
    print(f"{count} blocks used adaptative method in this iteration representing {count*100/len(indexes)} % of total")
    return Coeff,nCoeff,dCoeff,indexes,count,decision_bs, octree_nbits

def sort_gft_coeffs(Ahat,indexes,qstep, plot=False):
    N = Ahat[:,0].shape[0]
    mask_lo = np.zeros((N), dtype=bool)
    for start_end_tuple in indexes:
        # This implies that the Ahat is sorted by coefficient
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
        plt.scatter((Ahat_lo[:, 0]), (Ahat_lo[:, 1]), label='Ahat_lo', alpha=0.5, color='blue')
        plt.scatter((Ahat_hi[:, 0]), (Ahat_hi[:, 1]), label='Ahat_hi', alpha=0.5, color='red')
        
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
    PSNR_Y = calculate_psnr(Y,Coeff_quant,N,qstep)
    nPSNR_Y = calculate_psnr(nY,nCoeff_quant,N,qstep)
    dPSNR_Y = calculate_psnr(dY,dCoeff_quant,N,qstep)
    # Sort hi to low
    # BUG: First show bad distribution in plot
    Coeff_quant_sorted = sort_gft_coeffs(Coeff_quant,indexes,qstep)
    nCoeff_quant_sorted = sort_gft_coeffs(nCoeff_quant,indexes,qstep)
    dCoeff_quant_sorted = sort_gft_coeffs(dCoeff_quant,indexes,qstep)
    # Run-Length Golomb-Rice
    bs_Coeffs = code_YUV(Coeff_quant_sorted, bitstream_directory='res')
    bs_nCoeffs = code_YUV(nCoeff_quant_sorted, bitstream_directory='res')
    bs_dCoeffs = code_YUV(dCoeff_quant_sorted, bitstream_directory='res')
    return PSNR_Y,bs_Coeffs,nPSNR_Y,bs_nCoeffs, dPSNR_Y,bs_dCoeffs 

def extract_overhead(entropy_data, num_of_points, bsize):
    # Initialize the overhead sum to 0
    overhead_sum = 0

    # Filter the DataFrame for the given block size
    block_entropy = entropy_data[entropy_data['Block Size'] == bsize]
    
    # Loop through each number from 1 to num_of_points (inclusive)
    for num in range(1, num_of_points + 1):
        # Filter for the current self-loop priority
        self_loop_estimate = block_entropy[block_entropy['Self-Loop Priority'] == num]
        
        # Add the 'Overhead Stimation' value to the overhead_sum
        overhead_sum += self_loop_estimate['Overhead Stimation'].sum()  # sum() is used here in case there are multiple matches
    
    return overhead_sum


if __name__ == "__main__":
    ply_file = "res/longdress_vox10_1051.ply"
    V,C_rgb,_ = ply.ply_read8i(ply_file)
    N = V.shape[0]
    steps = [1, 2, 4, 8, 12, 16, 20, 24, 32, 64]
    block_sizes = [4]#,8,16]
    #point_fractions = [0.05]#,0.2,0.5]
    num_of_points=[1,2]#,4]
    weights = [1.2]#,1.6,2.0]
    data = []
    entropy_analysis = pd.read_csv('entropy_analysis.csv')
    for bsize in block_sizes:
        for num in num_of_points:
            for weight in weights:
                print(f"========================================================= \n Block size {bsize}, number of points: {num} and self-loop weight {weight} \n =========================================================")
                Coeff,nCoeff,dCoeff,indexes,count,decision_estimate, morton_bs = get_coefficients(V=V,C_rgb=C_rgb,block_size=bsize,self_loop_weight=weight,number_of_points=num)
                entropy_overhead_estimation = extract_overhead(entropy_analysis,num,bsize)
                print(f"Overhead stimate {entropy_overhead_estimation}")
                if count == 0:
                    print("Since no block was considered for adaptative method, then there is no use in iterate weights")
                    break
                for step in steps:
                    PSNR_Y,bs_Coeffs,nPSNR_Y,bs_nCoeffs, dPSNR_Y,bs_dCoeffs = quantize_PSNR_bs(Coeff,nCoeff,dCoeff,step,indexes)
                    bpv = (bs_Coeffs)/N
                    nbpv = bs_nCoeffs/N
                    #dbpv =(bs_dCoeffs+decision_estimate+entropy_overhead_estimation)/N
                    dbpv =(bs_dCoeffs+decision_estimate+morton_bs)/N
                    print(f"For block with size {bsize} and quantization step of {step} \n Adaptative method PSNR_Y,bitstream and bpv = {PSNR_Y,bs_Coeffs,bpv} \n Structural method PSNR_Y,bitstream and bpv = {nPSNR_Y,bs_nCoeffs,nbpv} \n Dynamic method PSNR_Y,bitstream and bpv = {dPSNR_Y,bs_dCoeffs+decision_estimate+entropy_overhead_estimation,dbpv} \n =========================================================")
                    data.append([bsize, num, weight, step, PSNR_Y,bs_Coeffs,bpv, nPSNR_Y,bs_nCoeffs,nbpv, dPSNR_Y,bs_dCoeffs+decision_estimate,dbpv])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Block Size", "Point Fraction", "Weight", "Step", "PSNR_Y","Adaptative bitstream", "Adaptative bpv","nPSNR_Y","Structural bitstream","Structural bpv", "dPSNR_Y","Dynamic bitstream","Dynamic bpv"])

    # Save the DataFrame to a CSV file (optional)
    df.to_csv("PSNR_experiment.csv", index=False)
