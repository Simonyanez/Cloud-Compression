import pandas as pd
from pyvis.network import Network
from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB

V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  


structural_encoder = StructuralEncoder(V,C_rgb)
indexes = structural_encoder.block_indexes(block_size = 16)
sorted_indexes = structural_encoder.std_sorted_indexes()
trial = sorted_indexes[-100]
net = structural_encoder.graph_visualization(trial)
net.show('trial_graph.html',notebook=False)
structural_encoder.block_visualization(trial)
choosed_positions = [39,42,45,43,91,92,95,96,97,110,111,112,115,116,118,125,128]
choosed_weights = [1.2]*len(choosed_positions)
idx_map = dict(zip(choosed_positions,choosed_weights))
net2 = structural_encoder.graph_visualization(trial, idx_map)
net2.show('trial_graph_self_looped.html',notebook=False)
plt.show()