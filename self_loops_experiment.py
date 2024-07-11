from src.encoder import *
import matplotlib.pyplot as plt
V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  

structural_encoder = StructuralEncoder(V,C_rgb)
directional_encoder = DirectionalEncoder(V,C_rgb)
structural_encoder.block_indexes(block_size = 16)
directional_encoder.block_indexes(block_size = 16)

sorted_indexes = directional_encoder.std_sorted_indexes()

# Find borders by structural graph
trial = sorted_indexes[-100]
W, border_positions, border_fig, y_values = directional_encoder.find_borders(trial)

# Add weights to specific border
#choosed_positions = [261,264,263,256,254,253,243,238,235,234,201,198,184]#[116,118,115,111,112,110,97,96,95,92,91,43,45,42,39]#[116,118,115]#117,89,0
choosed_positions = [39,42,45,43,91,92,95,96,97,110,111,112,115,116,118,125,128]
choosed_weights = [1.2]*len(choosed_positions)
idx_map = dict(zip(choosed_positions,choosed_weights))
GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(trial,W,idx_map)
nGFT, _, nAblockhat = directional_encoder.gft_transform(trial,W,None)
energy_fig = directional_encoder.energy_block(Ablockhat,"self-looped")
energy_fig_2 = directional_encoder.energy_block(nAblockhat,"structural")
base_fig= directional_encoder.component_projection(trial,GFT[:,0],"self-looped",y_values)
nbase_fig = directional_encoder.component_projection(trial,nGFT[:,0],"structural",y_values)
print(f"{np.min(GFT[:,0])},{np.max(GFT[:,0])}")
print(f"{np.min(nGFT[:,0])},{np.max(nGFT[:,0])}")
plt.show()

# #directional_encoder.
# print(f"Sizes {np.shape(GFT),np.shape(Gfreq),np.shape(Ablockhat)}")
#fig_2 = directional_encoder.direction_visualization(3300)
#test_blocks = [-49,-70]

#test_blocks_2 = [1600]

# Good iteration: block = 1612, n[117], possible borders = [117,89] 
# 1164sl vs 1156struct
# best 64

# block = -100 
# good_points = 116, 312
# bad_poinst = 0
# Con 0 muere, combinación muere, solos funciona
# Interesting combination = n[121,124] c_val = 1/np.sqrt(2) block =-70
# With this  Ablock_normalized = (Ablock - np.mean(Ablock, axis=0)) / np.std(Ablock, axis=0)