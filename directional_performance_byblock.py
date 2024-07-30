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

def atribute_dependent_codec(directional_encoder,Coeff,iteration,start_end_tuple,step):
    # NOTE: This may be called dependent_encoder or something like that. Also the directional
    # encoder must be initialized
    sorted_nodes = directional_encoder.simple_direction_sort(iteration)
    choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*0.05)]
    W,edges = directional_encoder.structural_graph(iteration)
    choosed_weights = [1.2]*len(choosed_positions)
    idx_map = dict(zip(choosed_positions,choosed_weights))
    #Ablockhat,block_decision = directional_encoder.dynamic_transform(iteration,W,idx_map)
    #decision.append(block_decision)
    _,_,Ablockhat= directional_encoder.gft_transform(iteration,W,idx_map)
    _,_,nAblockhat = directional_encoder.gft_transform(iteration,W,None)
    try:
        Coeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
    except:
        print(f"Size of the Ahat: {Ablockhat.shape} \n Start and end tuple: {start_end_tuple} \n Iteration:{iteration}")
    V,A = directional_encoder.get_block(iteration)
    Block_Coeff_Quant = np.round(Ablockhat/step)*step
    nBlock_Coeff_Quant = np.round(nAblockhat/step)*step
    N_block = V.shape[0]
    block_psnr_Y = calculate_psnr(Ablockhat[:,0],Block_Coeff_Quant,N_block)
    nblock_psnr_Y = calculate_psnr(nAblockhat[:,0],nBlock_Coeff_Quant,N_block)
    print(f"Block number {iteration}: \n Atribute Dependent PSNR: {block_psnr_Y} \n Structural Only-Info PSNR: {nblock_psnr_Y} \n ===============================================")

def full_codec_performance_analysis(ply_rute,block_size,quant_step):
    V,C_rgb,_ = ply.ply_read8i(ply_rute)
    directional_encoder = DirectionalEncoder(V,C_rgb)
    directional_encoder.block_indexes(block_size = block_size) 
    indexes = directional_encoder.indexes
    Coeff = np.zeros(C_rgb.shape)
    step = quant_step
    for iteration,start_end_tuple in enumerate(indexes):
        # NOTE: Original implementation avoid one points blocks
        atribute_dependent_codec(directional_encoder,Coeff,iteration,start_end_tuple,step)

        #print(f"Block number {iteration} has a PSNR of {block_psnr_Y} \n for Dynamic execution with {decision_map[decision[iteration]]} decision")

        if start_end_tuple[0] == 1:
            print(f"Border case {start_end_tuple}")
        if start_end_tuple[1] == C_rgb.shape[0]:
            print(f"Border case {start_end_tuple}")


#Y = Coeff[:,0]
#Coeff_quant = np.round(Coeff/step)*step
#np.save("Coeff_quant", Coeff_quant)
#C_rec = np.zeros(C_rgb.shape)
#for iteration,start_end_tuple in enumerate(indexes):
#    # NOTE: Original implementation avoid one points blocks
#    Ablockhat = Coeff_quant[start_end_tuple[0]:start_end_tuple[1]+1,:]
#    #sorted_nodes = directional_encoder.simple_direction_sort(iteration)
#    #choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*0.05)]
#    W,edges = directional_encoder.structural_graph(iteration)
#    #choosed_weights = [1.2]*len(choosed_positions)
#    #idx_map = dict(zip(choosed_positions,choosed_weights))
#    GFT_inv,Ablockrec = directional_encoder.igft_transform(iteration,W,None,Ablockhat)
#    C_rec[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockrec
#    plt.close()

#C_rgb_rec = YUVtoRGB(C_rec)
#C_rgb_rec = C_rgb_rec.astype(np.float64)

#N = V.shape[0]
#psnr_Y = calculate_psnr(Y,Coeff_quant,N)

#print(f"This is the PSNR value {psnr_Y}")
#ply.ply_write('PC_original.ply',V,C_rgb)
#ply.ply_write('PC_coded.ply',V,C_rgb_rec)

if __name__ == "__main__":
    V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")
    directional_encoder = DirectionalEncoder(V,C_rgb)
    directional_encoder.block_indexes(block_size = 8) 
    indexes = directional_encoder.indexes
    Coeff = np.zeros(C_rgb.shape)
    step = 64
    for iteration, start_end_tuple in enumerate(indexes):
        atribute_dependent_codec(directional_encoder,Coeff,iteration,start_end_tuple,step)
    
    import pstats

    # Load the profiling results
    p = pstats.Stats('profile_output.prof')

    # Strip directory paths to make the output more readable
    p.strip_dirs()

    # Sort by cumulative time and print the top results
    p.sort_stats('cumulative').print_stats(10)

    # Print callers information for functions containing 'linalg'
    p.print_callers(0.1, 'linalg')
    #full_codec_performance_analysis(ply_rute="res/longdress_vox10_1051.ply",
    #                                block_size=16,
    #                                quant_step=64)