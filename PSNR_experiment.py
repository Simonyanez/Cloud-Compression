from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB

def calculate_psnr(Y, Coeff_quant, N):
    # Extract the first column of Coeff_quant
    Coeff_quant_col1 = Coeff_quant[:, 0]

    # Calculate the norm (Euclidean distance) between Y and Coeff_quant_col1
    norm_value = np.linalg.norm(Y - Coeff_quant_col1)

    # Calculate PSNR
    psnr_Y = -10 * np.log10((norm_value ** 2) / (N * 255 ** 2))
    
    return psnr_Y

def get_coefficients(block_size,ply_file):
    V,C_rgb,_ = ply.ply_read8i(ply_file)
    directional_encoder = DirectionalEncoder(V,C_rgb)
    directional_encoder.block_indexes(block_size = block_size) 
    indexes = directional_encoder.indexes
    Coeff = np.zeros(C_rgb.shape)
    nCoeff = np.zeros(C_rgb.shape)
    Y = Coeff[:,0]
    N = V.shape[0]
    
    for iteration,start_end_tuple in enumerate(indexes):
        # NOTE: Original implementation avoid one points blocks

        sorted_nodes = directional_encoder.simple_direction_sort(iteration)
        choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*0.05)]
        W,edges = directional_encoder.structural_graph(iteration)
        choosed_weights = [1.2]*len(choosed_positions)
        idx_map = dict(zip(choosed_positions,choosed_weights))
        #Ablockhat,block_decision = directional_encoder.dynamic_transform(iteration,W,idx_map)
        #decision.append(block_decision)
        GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(iteration,W,idx_map)
        nGFT, nGfreq, nAblockhat = directional_encoder.gft_transform(iteration,W,None)
        Coeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
        nCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
    return Y,Coeff,nCoeff

def quantize_and_PSNR(Y,Coeff,nCoeff,qstep):
    N = Y.shape[0]
    print(f"The total number of points is {N}")
    Coeff_quant = np.round(Coeff/qstep)*qstep
    nCoeff_quant = np.round(nCoeff/qstep)*qstep
    PSNR_Y = calculate_psnr(Y,Coeff_quant,N)
    nPSNR_Y = calculate_psnr(Y,nCoeff_quant,N)
    return PSNR_Y,nPSNR_Y

if __name__ == "__main__":
    ply_file = "res/longdress_vox10_1051.ply"
    steps = [1,2,4,8,16,32,64]
    block_sizes = [2,4,8,16]
    results = {}
    for bsize in block_sizes:
        Y, Coeff,nCoeff = get_coefficients(block_size=bsize, ply_file=ply_file)
        for step in steps:
            PSNR_Y,nPSNR_Y = quantize_and_PSNR(Y,Coeff,nCoeff,step)
            print(f"For block with size {bsize} and quantization step of {step} \n Adaptative method PSNR_Y = {PSNR_Y} \n Structural method PSNR_Y = {nPSNR_Y} \n =========================================================")

