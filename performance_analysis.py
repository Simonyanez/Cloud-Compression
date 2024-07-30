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

V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")
directional_encoder = DirectionalEncoder(V,C_rgb)
directional_encoder.block_indexes(block_size = 4) 
indexes = directional_encoder.indexes
print(C_rgb.shape)
Coeff = np.zeros(C_rgb.shape)
decision = []
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
    try:
        Coeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
    except:
        print(f"Size of the Ahat: {Ablockhat.shape} \n Start and end tuple: {start_end_tuple} \n Iteration:{iteration}")
    plt.close()
    if start_end_tuple[0] == 1:
        print(f"Border case {start_end_tuple}")
    if start_end_tuple[1] == C_rgb.shape[0]:
        print(f"Border case {start_end_tuple}")
step = 16
Y = Coeff[:,0]
Coeff_quant = np.round(Coeff/step)*step
np.save("Coeff_quant", Coeff_quant)
C_rec = np.zeros(C_rgb.shape)
for iteration,start_end_tuple in enumerate(indexes):
    # NOTE: Original implementation avoid one points blocks
    Ablockhat = Coeff_quant[start_end_tuple[0]:start_end_tuple[1]+1,:]
    sorted_nodes = directional_encoder.simple_direction_sort(iteration)
    choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*0.05)]
    W,edges = directional_encoder.structural_graph(iteration)
    choosed_weights = [1.2]*len(choosed_positions)
    idx_map = dict(zip(choosed_positions,choosed_weights))
    GFT_inv,Ablockrec = directional_encoder.igft_transform(iteration,W,idx_map,Ablockhat)
    C_rec[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockrec
    plt.close()

C_rgb_rec = YUVtoRGB(C_rec)
#C_rgb_rec = C_rgb_rec.astype(np.float64)

N = V.shape[0]
psnr_Y = calculate_psnr(Y,Coeff_quant,N)
V = V.astype(np.uint8)
print(f"This is the PSNR value {psnr_Y}")
ply.ply_write('PC_original.ply',V,C_rgb)
ply.ply_write('PC_coded.ply',V,C_rgb_rec)