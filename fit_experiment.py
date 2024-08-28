from src.encoder import *
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-interactive backend for scripts

if __name__ == "__main__":
    ply_file = "res/longdress_vox10_1051.ply"
    V,C_rgb,_ = ply.ply_read8i(ply_file)
    block_sizes = [4,8,16]
    directional_encoder = DirectionalEncoder(V,C_rgb,plots_flag=True)
    directional_encoder.block_indexes(block_size=16)
    #test_block_iter = directional_encoder.indexes[100]
    directional_encoder.Y_visualization(300)
    pc_matrix = directional_encoder.pearson_correlation_matrix(300)
    print(f"This is how the pc matrix look \n {pc_matrix}")
    plt.show()
    
