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
import graph.properties as pt
import utils.color as clr
import utils.ply as ply
import utils.visualization as visual

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

class DirectionalEncoder:
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
    
    def block_visualization(self,index_pair):
        first_point, last_point = index_pair
        Vblock = self.V[first_point:last_point + 1, :] 
        Ablock = self.A[first_point:last_point + 1, :] 
        _,vis_fig = visual.visualization(Vblock,Ablock,1,1,"Normal")
        return vis_fig 
    def directional_graph(self,index_pair):
        first_point, last_point = index_pair
        Vblock = self.V[first_point:last_point + 1, :] 
        Ablock = self.A[first_point:last_point + 1, :] 
        _, _, distance_vectors, weights = pt.direction(Vblock, Ablock)
        print(f"This is distance vectors and weights sizes {np.shape(distance_vectors),np.shape(weights)}")
        W, edge, idx_closest = cr.compute_graph_sl(Vblock,distance_vectors,weights)
        return W,edge, idx_closest
    
    def gft_transform(self,index_pair):
        W,_,idx_closest= self.directional_graph(index_pair)
        first_point, last_point = index_pair
        Ablock = self.A[first_point:last_point + 1, :] 

        GFT, Gfreq, Ablockhat = tf.compute_GFT_noQ(W,Ablock,idx_closest=idx_closest)

        return GFT, Gfreq, Ablockhat
    
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

def energy_comparison(Coeffs_1, Coeffs_2):  
    # Comparing just the first 5 coefficients DC and 4 AC
    if np.shape(Coeffs_1)[0]>5:
        Y_diff = np.absolute(Coeffs_1[:5,0]) - np.absolute(Coeffs_2[:5,0])
        U_diff = np.absolute(Coeffs_1[:5,1]) - np.absolute(Coeffs_2[:5,1])
        V_diff = np.absolute(Coeffs_1[:5,2]) - np.absolute(Coeffs_2[:5,2])
    else:
        Y_diff = np.absolute(Coeffs_1[:,0]) - np.absolute(Coeffs_2[:,0])
        U_diff = np.absolute(Coeffs_1[:,1]) - np.absolute(Coeffs_2[:,1])
        V_diff = np.absolute(Coeffs_1[:,2]) - np.absolute(Coeffs_2[:,2])

    return np.sum(Y_diff),np.sum(U_diff),np.sum(V_diff)

if __name__ == "__main__":
    V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  
    structural_encoder = StructuralEncoder(V,C_rgb)
    directional_encoder = DirectionalEncoder(V,C_rgb)
    indexes = structural_encoder.block_indexes(block_size = 16)
    indexes_2 = directional_encoder.block_indexes(block_size = 16)
    assert indexes == indexes_2, "Los índices difieren"
    Y_diffs = np.zeros(len(indexes))
    U_diffs = np.zeros(len(indexes))
    V_diffs = np.zeros(len(indexes))
    for i,index in enumerate(indexes):
        _, _, Ablockhat_1 = structural_encoder.gft_transform(index)
        _, _, Ablockhat_2 = directional_encoder.gft_transform(index)
        (Y_diffs[i],
        U_diffs[i],
        V_diffs[i]
                    ) = energy_comparison(Ablockhat_1,Ablockhat_2) 
    
    indexes_range = np.array(list(range(1, len(indexes) + 1)))
    plt.hist(Y_diffs)
    plt.show()
    plt.hist(U_diffs)  
    plt.show()
    plt.hist(V_diffs)  
    plt.show()

    # Five most different coefficients
    improved_blocks_index = np.argsort(Y_diffs)
    best_five = improved_blocks_index[-5:]

    for best in best_five:
        index = indexes[best]
        fig = directional_encoder.block_visualization(index)
        plt.show()
        #fig = structural_encoder.energy_block(indexes[14])


    
    #fig = directional_encoder.energy_block(indexes[14])
    #plt.show()

    