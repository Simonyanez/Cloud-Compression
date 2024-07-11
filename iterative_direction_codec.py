import pandas as pd
from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB

V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  

# Find positions
directional_encoder = DirectionalEncoder(V,C_rgb)
indexes = directional_encoder.block_indexes(block_size = 16)
sorted_indexes = directional_encoder.std_sorted_indexes()
DC_components_sl = []
DC_components_nr = []
for index in sorted_indexes:
    # try:
    sorted_nodes = directional_encoder.simple_direction_sort(index)
    choosed_positions = sorted_nodes[0:int(len(sorted_nodes)*0.05)]
    W,edges = directional_encoder.structural_graph(index)
    choosed_weights = [1.2]*len(choosed_positions)
    idx_map = dict(zip(choosed_positions,choosed_weights))
    GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(index,W,idx_map)
    nGFT, nGfreq, nAblockhat = directional_encoder.gft_transform(index,W,idx_map=None)
    DC_components_sl.append(Ablockhat[0,0])
    DC_components_nr.append(nAblockhat[0,0])
    # except:
    #     DC_components_sl.append(0)
    #     DC_components_nr.append(0)

fig = visual.plot_DC_difference_with_annotation(DC_components_sl,DC_components_nr)


# Show the sub graph of sorted nodes
plt.show()