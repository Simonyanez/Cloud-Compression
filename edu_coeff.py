import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Plot for Coeff_b4
axs[0, 0].plot(Coeff_b4_edu, label='Edu Coeff b4')
axs[0, 0].set_title('Coefficients from .mat (b4)')
# axs[0, 0].set_xticks([])
axs[0, 0].legend()

axs[0, 1].plot(Coeff_b4_simon, label='Simon Coeff b4')
axs[0, 1].set_title('Coefficients from .npy (b4)')
# axs[0, 1].set_xticks([])
axs[0, 1].legend()

# Plot for Coeff_b8
axs[1, 0].plot(Coeff_b8_edu, label='Edu Coeff b8')
axs[1, 0].set_title('Coefficients from .mat (b8)')
# axs[1, 0].set_xticks([])
axs[1, 0].legend()

axs[1, 1].plot(Coeff_b8_simon, label='Simon Coeff b8')
axs[1, 1].set_title('Coefficients from .npy (b8)')
# axs[1, 1].set_xticks([])
axs[1, 1].legend()

# Plot for Coeff_b16
axs[2, 0].plot(Coeff_b16_edu, label='Edu Coeff b16')
axs[2, 0].set_title('Coefficients from .mat (b16)')
# axs[2, 0].set_xticks([])
axs[2, 0].legend()

axs[2, 1].plot(Coeff_b16_simon, label='Simon Coeff b16')
axs[2, 1].set_title('Coefficients from .npy (b16)')
# axs[2, 1].set_xticks([])
axs[2, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Get rare positions 
Ypositions_4 = np.argwhere(np.abs(Coeff_b4_simon[:,0])>400)
np.save('res/Ypositions_4.npy',Ypositions_4)
Ypositions_8 = np.argwhere(np.abs(Coeff_b8_simon[:,0])>400)
np.save('res/Ypositions_8.npy',Ypositions_8)
Ypositions_16 = np.argwhere(np.abs(Coeff_b16_simon[:,0])>400)
np.save('res/Ypositions_16.npy',Ypositions_16)
#====================================================================================


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
for i,MSE in enumerate(MSEs):
    print(f"Block {bsizes[i]} MSE between coefficients {MSE}")
    print("===============================================================")
    for j in range(3):
        plt.figure(figsize=(15,10))
        plt.hist(SQQs[i][:,j], bins=50)
        plt.title(f"Mean Squared Error for {CHs[j]} Channel and Block Size {bsizes[i]}")
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.show()

        POSs.append(np.argwhere(SQQs[i][:,j]> 1e5))
# Print byte sizes
print("Byte sizes comparison:")
for i, step in enumerate(steps):
    print(f"Block 4 Step {step} .mat: {Bytes_b4_edu[0][i]*8} bits, .npy: {Bits_b4_simon[i]} bits")
    print(f"Block 8 Step {step} .mat: {Bytes_b8_edu[0][i]*8} bits, .npy: {Bits_b8_simon[i]} bits")
    print(f"Block 16 Step {step} .mat: {Bytes_b16_edu[0][i]*8} bits, .npy: {Bits_b16_simon[i]} bits")
    print("=================================================================================================")
# List of coefficients and block names for both sources
coeff_blocks = [(Coeff_b4_edu, Coeff_b4_simon), (Coeff_b8_edu, Coeff_b8_simon), (Coeff_b16_edu, Coeff_b16_simon)]
block_names = ['Block 4', 'Block 8', 'Block 16']

# Number of channels (assuming the second dimension represents channels)
num_channels = Coeff_b4_edu.shape[1]

# Create side-by-side plots
for block_index, (coeff, coeff_simon) in enumerate(coeff_blocks):
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Histograms for {block_names[block_index]}', fontsize=16)
    
    for channel in range(num_channels):
        # .mat histogram
        plt.subplot(2, num_channels, channel + 1)
        plt.hist(coeff[:, channel], bins=50, alpha=0.7)
        plt.title(f'.mat Channel {channel + 1}')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Frequency')

        # .npy histogram
        plt.subplot(2, num_channels, channel + 1 + num_channels)
        plt.hist(coeff_simon[:, channel], bins=50, alpha=0.7)
        plt.title(f'.npy Channel {channel + 1}')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
    plt.show()