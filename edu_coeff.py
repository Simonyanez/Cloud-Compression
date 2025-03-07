import numpy as np
import graph.create as cr
import graph.transforms as tr
import utils.ply as ply
import scipy.io as sio
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from encode.encode import *

V = np.load('V_longdress.npy')

def scatter_coeff(Coeff_1, Coeff_2, bsize, qstep):
    Coeff_1_quant =  np.round(Coeff_1/qstep)
    Coeff_2_quant =  np.round(Coeff_2/qstep)
    fig = plt.figure()


    ax_1 = fig.add_subplot(1,2,1, projection='3d')
    ax_1.scatter(Coeff_1_quant[:,0],Coeff_1_quant[:,1],Coeff_1_quant[:,2], c='red', label='Simon Coeff')
    ax_1.legend()

    ax_2 = fig.add_subplot(1,2,2, projection='3d')
    ax_2.scatter(Coeff_2_quant[:,0],Coeff_2_quant[:,1],Coeff_2_quant[:,2], c='blue', label='Edu Coeff')
    ax_2.legend()
    
    plt.suptitle(f'Block Size: {bsize} Quantization Step: {qstep}')
    plt.tight_layout()
    plt.show()


    # Load the coefficient data for all blocks from .mat files
Coeff_b4_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_4simon.mat")
Coeff_b8_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_8simon.mat")
Coeff_b16_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_16simon.mat")

# Load the coefficient data for all blocks from .mat files
Coeff_quant_b4_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_4simon_quant.mat")
Coeff_quant_b8_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_8simon_quant.mat")
Coeff_quant_b16_read = sio.loadmat("res/Eduardo_Exp/_RA-GFT_exp_zhang_16simon_quant.mat")

# Extract coefficients and bytes from .mat files
Coeff_b4_edu = Coeff_b4_read['Coeff']
Coeff_b8_edu = Coeff_b8_read['Coeff']
Coeff_b16_edu = Coeff_b16_read['Coeff']


# Load coefficient data from .npy files
Coeff_b4_simon = np.load('res/struct_GFT_4_exp.npy')
Coeff_b8_simon = np.load('res/struct_GFT_8_exp.npy')
Coeff_b16_simon = np.load('res/struct_GFT_16_exp.npy')

# Save coefficients as .mat files
sio.savemat('Coeff_b4_simon.mat', {'Coeffs': Coeff_b4_simon})
sio.savemat('Coeff_b8_simon.mat', {'Coeffs': Coeff_b8_simon})
sio.savemat('Coeff_b16_simon.mat', {'Coeffs': Coeff_b16_simon})
# # Create subplots
# fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# # Plot for Coeff_b4
# axs[0, 0].plot(Coeff_b4_edu, label='Edu Coeff b4')
# axs[0, 0].set_title('Coefficients from .mat (b4)')
# # axs[0, 0].set_xticks([])
# axs[0, 0].legend()

# axs[0, 1].plot(Coeff_b4_simon, label='Simon Coeff b4')
# axs[0, 1].set_title('Coefficients from .npy (b4)')
# # axs[0, 1].set_xticks([])
# axs[0, 1].legend()

# # Plot for Coeff_b8
# axs[1, 0].plot(Coeff_b8_edu, label='Edu Coeff b8')
# axs[1, 0].set_title('Coefficients from .mat (b8)')
# # axs[1, 0].set_xticks([])
# axs[1, 0].legend()

# axs[1, 1].plot(Coeff_b8_simon, label='Simon Coeff b8')
# axs[1, 1].set_title('Coefficients from .npy (b8)')
# # axs[1, 1].set_xticks([])
# axs[1, 1].legend()

# # Plot for Coeff_b16
# axs[2, 0].plot(Coeff_b16_edu, label='Edu Coeff b16')
# axs[2, 0].set_title('Coefficients from .mat (b16)')
# # axs[2, 0].set_xticks([])
# axs[2, 0].legend()

# axs[2, 1].plot(Coeff_b16_simon, label='Simon Coeff b16')
# axs[2, 1].set_title('Coefficients from .npy (b16)')
# # axs[2, 1].set_xticks([])
# axs[2, 1].legend()

# # Adjust layout
# plt.tight_layout()
# plt.show()

# Get rare positions 
Ypositions_4 = np.argwhere(np.abs(Coeff_b4_simon[:,0])>400)
np.save('res/Ypositions_4.npy',Ypositions_4)
Ypositions_8 = np.argwhere(np.abs(Coeff_b8_simon[:,0])>400)
np.save('res/Ypositions_8.npy',Ypositions_8)
Ypositions_16 = np.argwhere(np.abs(Coeff_b16_simon[:,0])>400)
np.save('res/Ypositions_16.npy',Ypositions_16)
#====================================================================================

# # Scatter plot
# scatter_coeff(Coeff_b16_simon, Coeff_b16_edu, 4, 16)
# scatter_coeff(Coeff_b16_simon, Coeff_b16_edu, 8, 16)
# scatter_coeff(Coeff_b16_simon, Coeff_b16_edu, 16, 16)

# Squared diff
SQQ_b4 = (Coeff_b4_edu - Coeff_b4_simon)**2
SQQ_b8 = (Coeff_b8_edu - Coeff_b8_simon)**2
SQQ_b16 = (Coeff_b16_edu - Coeff_b16_simon)**2

# MSE
MSE_b4 = (SQQ_b4).mean(axis=1)
MSE_b8 = (SQQ_b8).mean(axis=1)
MSE_b16 = (SQQ_b16).mean(axis=1)

# Get bytes (bits?)
Bytes_b4_edu = Coeff_quant_b4_read['bytes']
Bytes_b8_edu = Coeff_quant_b8_read['bytes']
Bytes_b16_edu = Coeff_quant_b16_read['bytes']

# Get personal bits
Bits_b4_simon = np.load('res/struct_GFT_4_exp_bits.npy')
Bits_b8_simon = np.load('res/struct_GFT_8_exp_bits.npy')
Bits_b16_simon = np.load('res/struct_GFT_16_exp_bits.npy')

steps = [1, 2, 4, 8, 12, 16, 20, 24, 32, 64]

# Visualize squared diff

# Print MSE
SQQs = [SQQ_b4,SQQ_b8,SQQ_b16]
MSEs = [MSE_b4,MSE_b8,MSE_b16]
POSs = []
CHs = ['Y','U','V']
bsizes = [4,8,16]
# for i,MSE in enumerate(MSEs):
#     print(f"Block {bsizes[i]} MSE between coefficients {MSE}")
#     print("===============================================================")
#     for j in range(3):
#         plt.figure(figsize=(15,10))
#         plt.hist(SQQs[i][:,j], bins=50)
#         plt.title(f"Mean Squared Error for {CHs[j]} Channel and Block Size {bsizes[i]}")
#         plt.xlabel("MSE")
#         plt.ylabel("Count")
#         plt.show()

#         POSs.append(np.argwhere(SQQs[i][:,j]> 1e5))

# Check representation
print(f"Edu Coeff representation {type(np.round(Coeff_b4_edu/2)[0,0])}")
print(f"Simón Coeff representation {type(np.round(Coeff_b4_simon/2)[0,0])}")
# Print byte sizes
print("Byte sizes comparison:")
for i, step in enumerate(steps):
    bs_b4_edu  = code_YUV(np.round(Coeff_b4_edu/step),'b4')
    bs_b8_edu  = code_YUV(np.round(Coeff_b8_edu/step), 'b8')
    bs_b16_edu = code_YUV(np.round(Coeff_b16_edu/step), 'b16')
    print(f"Block 4 Step {step} .mat: {bs_b4_edu} bits, .npy: {Bits_b4_simon[i]} bits")
    print(f"Block 8 Step {step} .mat: {bs_b8_edu} bits, .npy: {Bits_b8_simon[i]} bits")
    print(f"Block 16 Step {step} .mat: {bs_b16_edu} bits, .npy: {Bits_b16_simon[i]} bits")
    print("================================================================================================= \n Decode MSE")
    N = Coeff_b4_edu.shape[0]
    decoded_Coeff_b4_edu = decode_YUV(N,'b4')
    decoded_Coeff_b8_edu = decode_YUV(N, 'b8')
    decoded_Coeff_b16_edu = decode_YUV(N, 'b16')
    print(f"b4 MSE: {np.mean((np.round(Coeff_b4_edu/step).astype(np.int16)-decoded_Coeff_b4_edu)**2)}")
    print(f"b8 MSE: {np.mean((np.round(Coeff_b8_edu/step).astype(np.int16)-decoded_Coeff_b8_edu)**2)}")
    print(f"b16 MSE: {np.mean((np.round(Coeff_b16_edu/step).astype(np.int16)-decoded_Coeff_b16_edu)**2)}")
# List of coefficients and block names for both sources
coeff_blocks = [(Coeff_b4_edu, Coeff_b4_simon), (Coeff_b8_edu, Coeff_b8_simon), (Coeff_b16_edu, Coeff_b16_simon)]
block_names = ['Block 4', 'Block 8', 'Block 16']

# Number of channels (assuming the second dimension represents channels)
num_channels = Coeff_b4_edu.shape[1]

# Create side-by-side plots
# for block_index, (coeff, coeff_simon) in enumerate(coeff_blocks):
#     plt.figure(figsize=(15, 10))
#     plt.suptitle(f'Histograms for {block_names[block_index]}', fontsize=16)
    
#     for channel in range(num_channels):
#         # .mat histogram
#         plt.subplot(2, num_channels, channel + 1)
#         plt.hist(coeff[:, channel], bins=50, alpha=0.7)
#         plt.title(f'.mat Channel {channel + 1}')
#         plt.xlabel('Coefficient Value')
#         plt.ylabel('Frequency')

#         # .npy histogram
#         plt.subplot(2, num_channels, channel + 1 + num_channels)
#         plt.hist(coeff_simon[:, channel], bins=50, alpha=0.7)
#         plt.title(f'.npy Channel {channel + 1}')
#         plt.xlabel('Coefficient Value')
#         plt.ylabel('Frequency')
    
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
#     plt.show()

def calculate_psnr(Coeff, N,qstep):
    # Extract the first column of Coeff_quant
    Coeff_quant = np.round(Coeff/qstep)
    Coeff_dequant = Coeff_quant*qstep
    Coeff_dequant_Y = Coeff_dequant[:, 0]
    Y = Coeff[:,0]
    # Calculate the norm (Euclidean distance) between Y and Coeff_quant_Y
    norm_value = np.linalg.norm(Y - Coeff_dequant_Y)
    inner_value = (norm_value ** 2) / ((255 ** 2) * N )
    # Calculate PSNR
    psnr_Y = -10 * np.log10(inner_value)
    
    return psnr_Y
from utils.bj_delta import *
def PSNR_vs_bpv_from_coeff(Coeffs, matlab_bitstream, N, V, compressed_images=False,references=["Simón PyRLGR", "Eduardo PyRLGR", "Eduardo Matlab RLGR"]):
    indexes_b4 = cr.get_block_indexes(V,4)
    indexes_b8 = cr.get_block_indexes(V,8)
    indexes_b16 = cr.get_block_indexes(V,16)
    # COEFFS for Eduardo and Simón
    Coeff_b4_edu = Coeffs['Eduardo b4']
    Coeff_b8_edu = Coeffs['Eduardo b8']
    Coeff_b16_edu = Coeffs['Eduardo b16']
    Coeff_b4_sim = Coeffs['Simon b4']
    Coeff_b8_sim = Coeffs['Simon b8']
    Coeff_b16_sim = Coeffs['Simon b16']
    
    steps = [8, 12, 16, 20, 24, 32, 64]
    
    # Lists for storing BPV and PSNR data
    bpv_b4_edu  = []
    bpv_b8_edu  = []
    bpv_b16_edu = []
    bpv_b4_sim  = []
    bpv_b8_sim  = []
    bpv_b16_sim = []
    matbpv_b4_edu = []
    matbpv_b8_edu = []
    matbpv_b16_edu = []

    psnr_b4_edu  = []
    psnr_b8_edu  = []
    psnr_b16_edu = []
    psnr_b4_sim  = []
    psnr_b8_sim  = []
    psnr_b16_sim = []

    for step in steps:
        # Calculate BPV and PSNR for Eduardo
        bpv_b4_edu.append(code_YUV(np.round(Coeff_b4_edu / step), '') / N)
        bpv_b8_edu.append(code_YUV(np.round(Coeff_b8_edu / step), '') / N)
        bpv_b16_edu.append(code_YUV(np.round(Coeff_b16_edu / step), '') / N)
        matbpv_b4_edu.append(matlab_bitstream[f'b4 {step}']*8 / N)
        matbpv_b8_edu.append(matlab_bitstream[f'b8 {step}']*8 / N)
        matbpv_b16_edu.append(matlab_bitstream[f'b16 {step}']*8 / N)
        
        psnr_b4_edu.append(calculate_psnr(Coeff_b4_edu, N, step))
        psnr_b8_edu.append(calculate_psnr(Coeff_b8_edu, N, step))
        psnr_b16_edu.append(calculate_psnr(Coeff_b16_edu, N, step))
        
        # Calculate BPV and PSNR for Simón
        bpv_b4_sim.append(code_YUV(np.round(Coeff_b4_sim / step), '') / N)
        if compressed_images:
            get_compressed_images(f"longdress_qstep{step}_b4.ply",'',indexes_b4,N,V)
        bpv_b8_sim.append(code_YUV(np.round(Coeff_b8_sim / step), '') / N)
        if compressed_images:
            get_compressed_images(f"longdress_qstep{step}_b8.ply",'',indexes_b8,N,V)
        bpv_b16_sim.append(code_YUV(np.round(Coeff_b16_sim / step), '') / N)
        if compressed_images:
            get_compressed_images(f"longdress_qstep{step}_b16.ply",'',indexes_b16,N,V)
        
        psnr_b4_sim.append(calculate_psnr(Coeff_b4_sim, N, step))
        psnr_b8_sim.append(calculate_psnr(Coeff_b8_sim, N, step))
        psnr_b16_sim.append(calculate_psnr(Coeff_b16_sim, N, step))

    # Plotting BPV vs PSNR
    plt.figure(figsize=(10, 6))

    # Plot Eduardo's data (Structural Mode)
    plt.plot(bpv_b4_edu, psnr_b4_edu, marker='o', color='blue', linestyle='-', label='Eduardo - Block Size 4 (PyRLGR)', alpha=0.7)
    plt.plot(bpv_b8_edu, psnr_b8_edu, marker='s', color='green', linestyle='-', label='Eduardo - Block Size 8 (PyRLGR)', alpha=0.7)
    plt.plot(bpv_b16_edu, psnr_b16_edu, marker='^', color='red', linestyle='-', label='Eduardo - Block Size 16 (PyRLGR)', alpha=0.7)

    # Plot Simón's data (Structural Mode)
    plt.plot(bpv_b4_sim, psnr_b4_sim, marker='o', color='blue', linestyle='--', label='Simón - Block Size 4 (PyRLGR)', alpha=0.7)
    plt.plot(bpv_b8_sim, psnr_b8_sim, marker='s', color='green', linestyle='--', label='Simón - Block Size 8 (PyRLGR)', alpha=0.7)
    plt.plot(bpv_b16_sim, psnr_b16_sim, marker='^', color='red', linestyle='--', label='Simón - Block Size 16 (PyRLGR)', alpha=0.7)

    # Plot MATLAB bitstreams (Dynamic Mode) using Eduardo's PSNR values
    plt.plot(matbpv_b4_edu, psnr_b4_edu, marker='o', color='blue', linestyle=':', label='Eduardo - Block Size 4 (Matlab RLGR)', alpha=0.5)
    plt.plot(matbpv_b8_edu, psnr_b8_edu, marker='s', color='green', linestyle=':', label='Eduardo - Block Size 8 (Matlab RLGR)', alpha=0.5)
    plt.plot(matbpv_b16_edu, psnr_b16_edu, marker='^', color='red', linestyle=':', label='Eduardo - Block Size 16 (Matlab RLGR)', alpha=0.5)

    # Customize the plot
    plt.title('BPV vs PSNR')
    plt.xlabel('Bitrate per Pixel (bpv)')
    plt.ylabel('PSNR_Y')
    plt.grid(True, which='both')
    plt.legend(title='Legend')
    plt.show()

Coeffs = {}
Coeffs['Eduardo b4'] = Coeff_b4_edu
Coeffs['Eduardo b8'] = Coeff_b8_edu
Coeffs['Eduardo b16'] = Coeff_b16_edu
Coeffs['Simon b4'] = Coeff_b4_simon
Coeffs['Simon b8'] = Coeff_b8_simon
Coeffs['Simon b16'] = Coeff_b16_simon

matlab_bitstreams = {}
print(Coeff_b4_read['colorStep'])
print(Coeff_b4_read['bytes'])
for i,b4_step in enumerate(Coeff_b4_read['colorStep'][0]):
    matlab_bitstreams[f'b4 {b4_step}'] = Coeff_b4_read['bytes'][0][i]
    matlab_bitstreams[f'b8 {b4_step}'] = Coeff_b8_read['bytes'][0][i]
    matlab_bitstreams[f'b16 {b4_step}'] = Coeff_b16_read['bytes'][0][i]

print(matlab_bitstreams)

def get_compressed_images(ply_file, bin_folder, indexes, N, V):
    Coeff = decode_YUV(N, bin_folder) 
    Arec = np.zeros(V.shape)
    for index in indexes:
        Vblock = V[index[0]: index[1]]
        W,_ = cr.compute_graph_MSR(Vblock)  # Structural data is given
        Coeff_block = Coeff[index[0]:index[1]]
        _, Ablockrec = tr.compute_iGFT_noQ(W,Coeff_block)
        Arec[index[0]:index[1]] = Ablockrec
    
    ply.ply_write(filename=ply_file,V=V,C=Arec)


PSNR_vs_bpv_from_coeff(Coeffs, matlab_bitstreams, N, V, compressed_images = False )