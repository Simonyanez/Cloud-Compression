from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB

V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  

# Find positions
directional_encoder = DirectionalEncoder(V,C_rgb)
indexes = directional_encoder.block_indexes(block_size = 16)
sorted_indexes = directional_encoder.std_sorted_indexes()
trial = sorted_indexes[-100]
block_fig = directional_encoder.block_visualization(trial)
Y_fig = directional_encoder.Y_visualization(trial)
dir_fig, count_fig, sorted_nodes = directional_encoder.simple_direction_visualization(trial)
choosed_positions = sorted_nodes[0:15]
nodes_fig = directional_encoder.node_positions(trial,choosed_positions)

W,edges = directional_encoder.structural_graph(trial)
choosed_weights = [1.2]*len(choosed_positions)
idx_map = dict(zip(choosed_positions,choosed_weights))
GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(trial,W,idx_map)
energy_fig = directional_encoder.energy_block(Ablockhat,"self-looped")
base_fig= directional_encoder.component_projection(trial,GFT[:,0],"self-looped",None)
# Show the sub graph of sorted nodes
plt.show()