import sys
sys.path.append('/home/simao/Repositories/Cloud-Compression')
# for path in sys.path:
#     print(path)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import graph.create as cr 
import graph.transforms as tf 
import utils.color as clr
import utils.ply as ply
# Product Class
class StructuralEncoder:
    def __init__(self,V,A) -> None:
        A = clr.RGBtoYUV(A)  
        self.V = V
        self.A = A
    
    def block_indexes(self,block_size):    # Stores indexes for later access
        # Assumes point cloud is morton ordered
        base_block_size = np.log2(block_size) 
        assert np.all(np.floor(base_block_size) == base_block_size), "block size b should be a power of 2"
        V_coarse = np.floor(self.V / block_size) * block_size

        variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)

        variation = np.concatenate(([1], variation))

        start_indexes = np.nonzero(variation)[0]
        Nlevel = self.V.shape[0]
        end_indexes = np.concatenate((start_indexes[1:] - 1, np.array([Nlevel - 1])))
        
        indexes = list(zip(start_indexes,end_indexes))  # Paired start and end indexes
        return indexes

    def structural_graph(self,index_pair):
        first_point, last_point = index_pair
        Vblock = self.V[first_point:last_point + 1, :] 
        W, edge = cr.compute_graph_MSR(Vblock)
        return W,edge
    
    def gft_transform(self,index_pair):
        W,_= self.structural_graph(index_pair)
        first_point, last_point = index_pair
        Ablock = self.A[first_point:last_point + 1, :] 

        GFT, Gfreq, Ablockhat = tf.compute_GFT_noQ(W,Ablock)
        return GFT, Gfreq, Ablockhat
    
    def quantization(self,index_pair,step):
        _,_ , Ablockhat = self.gft_transform(index_pair)
        Ablockhat_quantized = round(Ablockhat/step)*step 
        return Ablockhat_quantized
    
    def entropy_coding(self,coeff):
        # RGLR encoder
        pass

    def encode_block(self, index_pair, step):
        Ablockhat_quantized = self.quantization(index_pair, step)
        #blockbitstream = entropy_coding(Ablockhat_quantized)
        #return blockbitstream
    
    def encode(self,block_size,step):
        indexes = self.block_indexes(block_size)
        indexes_filtered = [index for index, pair in enumerate(indexes) if pair[1]!=pair[0]]
        for index_pair in indexes_filtered:
            blockbitstream = self.encode_block(index_pair,step)
            #bitstream = entropy_coding(Ablockhat_quantized)

    def energy_block(self, index_pair):
        _, _, Ablockhat = self.gft_transform(index_pair)
        Y = abs(Ablockhat[:, 0])
        U = abs(Ablockhat[:, 1])
        V = abs(Ablockhat[:, 2])
        
        # Create scatter plot
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))  # Create 3 subplots vertically
        
        # Plot Y in the first subplot
        axs[0].scatter(range(len(Y)), Y, c='r', label='Y')  # Red color for Y
        axs[0].set_title('Y Channel')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Magnitude')
        
        # Plot U in the second subplot
        axs[1].scatter(range(len(U)), U, c='g', label='U')  # Green color for U
        axs[1].set_title('U Channel')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Magnitude')
        
        # Plot V in the third subplot
        axs[2].scatter(range(len(V)), V, c='b', label='V')  # Blue color for V
        axs[2].set_title('V Channel')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Magnitude')
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        return fig


    
if __name__ == "__main__":
    V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  
    encoder = StructuralEncoder(V,C_rgb)
    indexes = encoder.block_indexes(block_size = 16)
    fig = encoder.energy_block(indexes[14])
    plt.show()

    