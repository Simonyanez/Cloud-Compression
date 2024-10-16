import os
import logging
import numpy as np
import utils.ply as ply

from src.encoder import *
logger = logging.getLogger(__name__)
# Pre-save numpy so that it doesn't read it every time
file_conditions = os.path.exists('V_longdress.npy') and os.path.exists('C_longdress.npy')

if file_conditions:
    V = np.load('V_longdress.npy')
    C_rgb = np.load('C_longdress.npy')
else:
    V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  
    np.save('V_longdress.npy',V)
    np.save('C_longdress.npy',C_rgb)

directional_encoder = DirectionalEncoder(V, C_rgb,True)
directional_encoder.block_indexes(block_size=16)
print(len(directional_encoder.indexes))
# Basic check
for i, (start, end) in enumerate(directional_encoder.indexes):
    # NOTE: There are some indexes that are equal
    assert end >= start, f'Last index greater than first {start,end}'
    assert (start, end) not in directional_encoder.indexes[i+1:], 'Se repiten los índices'

directional_encoder.block_visualization(100)
Vblock,Ablock = directional_encoder.get_block(100)
plt.show()
