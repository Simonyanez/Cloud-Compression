import sys
sys.path.append('/home/simao/Repositories/Cloud-Compression')
# for path in sys.path:
#     print(path)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import graph.create as cr 
import graph.transforms as tf 
import graph.properties as pt
import utils.color as clr
import utils.ply as ply
import utils.visualization as visual
from utils.encode_rlgr import *

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
        self.indexes = sorted(indexes, key=lambda x: x[1]-x[0], reverse=True)
        return self.indexes

    def std_sorted_indexes(self):
        indexes = self.indexes
        stds = np.zeros(len(indexes))
        for i in range(len(indexes)):
            _, Ablock = self.get_block(i)
            # Normalize Ablock
            mean_Ablock = np.mean(Ablock, axis=0)
            std_Ablock = np.std(Ablock, axis=0)
            # Avoid division by zero or close to zero
            epsilon = 1e-8  # Small epsilon value
            # Calculate normalized Ablock
            Ablock_normalized = (Ablock - mean_Ablock) / (std_Ablock + epsilon)
            channel_std = np.std(Ablock_normalized, axis=0)
            stds[i] = np.sqrt(np.sum(channel_std ** 2))
        order = np.argsort(stds)
        return order
    
    def get_block(self,iter):
        first_point, last_point = self.indexes[iter]
        Vblock = self.V[first_point:last_point + 1, :] 
        Ablock = self.A[first_point:last_point + 1, :] 
        return Vblock,Ablock
    
    def block_visualization(self,iter):
        Vblock,Ablock = self.get_block(iter)
        _,vis_fig = visual.visualization(Vblock,Ablock,1,1,"Normal")
        return vis_fig 
    
    def graph_visualization(self,iter,idx_map=None):
        _,Ablock = self.get_block(iter)
        W,edges = self.structural_graph(iter)
        net = visual.graph_visualization(edges,W,Ablock,idx_map=idx_map)
        return net
    
    def structural_graph(self,index_pair):
        first_point, last_point = index_pair
        Vblock = self.V[first_point:last_point + 1, :] 
        W, edges = cr.compute_graph_MSR(Vblock)
        return W,edges
    
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
    def __init__(self,V,A,plots_flag = False) -> None:
        A = clr.RGBtoYUV(A)  
        self.V = V
        self.A = A
        self.indexes = None
        self.plots = plots_flag
    
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
        indexes = sorted(indexes, key=lambda x: x[1]-x[0],reverse=True)
        self.indexes = indexes

    def get_block(self,iter):
        first_point, last_point = self.indexes[iter]
        Vblock = self.V[first_point:last_point + 1, :] 
        Ablock = self.A[first_point:last_point + 1, :] 
        return Vblock,Ablock
    
    def Y_visualization(self,iter):
        Vblock,Ablock = self.get_block(iter)
        if self.plots:
            _,vis_fig = visual.Yvisualization(Vblock,Ablock)
            return vis_fig 
        else:
            print("Plots are desactivated")

    def std_sorted_indexes(self):
        indexes = self.indexes
        stds = np.zeros(len(indexes))
        for i in range(len(indexes)):
            _, Ablock = self.get_block(i)
            # Normalize Ablock
            mean_Ablock = np.mean(Ablock, axis=0)
            std_Ablock = np.std(Ablock, axis=0)
            # Avoid division by zero or close to zero
            epsilon = 1e-8  # Small epsilon value
            # Calculate normalized Ablock
            Ablock_normalized = (Ablock - mean_Ablock) / (std_Ablock + epsilon)
            channel_std = np.std(Ablock_normalized, axis=0)
            stds[i] = np.sqrt(np.sum(channel_std ** 2))
        order = np.argsort(stds)
        return order

    
    
    def block_visualization(self,iter):
        Vblock,Ablock = self.get_block(iter)
        if self.plots:
            _,vis_fig = visual.visualization(Vblock,Ablock,1,1,"Normal")
            return vis_fig 
        else:
            print("Plots are desactivated")
    
    def simple_direction_visualization(self,iter):
        Vblock,Ablock = self.get_block(iter)
        W,edges = self.structural_graph(iter)
        direction_dict,normalization_dict = pt.simple_direction(Ablock,W,edges)
        count_dict = pt.find_most_pointed_to(direction_dict,normalization_dict)
        if self.plots:
            dir_fig = visual.plot_vector_field(Vblock,Ablock,W,edges)
            count_fig ,sorted_nodes = visual.plot_count_dict(count_dict,Ablock[:,0])
        sorted_nodes = pt.sort_most_pointed(count_dict,Ablock[:,0])
        return sorted_nodes
    
    def simple_direction_sort(self,iter):
        _,Ablock = self.get_block(iter)
        W,edges = self.structural_graph(iter)
        direction_dict,normalization_dict = pt.simple_direction(Ablock,W,edges)
        count_dict = pt.find_most_pointed_to(direction_dict,normalization_dict)
        sorted_nodes = pt.sort_most_pointed(count_dict,Ablock[:,0])
        return sorted_nodes
    
    def pearson_correlation_matrix(self,iter):
        sorted_nodes = self.simple_direction_sort(iter)
        Vblock,_ = self.get_block(iter)
        sorted_positions = []
        how_many = 5
        for node in sorted_nodes[:how_many]:
            node_position = Vblock[node]
            sorted_positions.append(node_position)

        # Convert to a DataFrame
        df = pd.DataFrame(sorted_positions, columns=['X', 'Y', 'Z'])

        # Calculate the Pearson correlation matrix
        correlation_matrix = df.corr(method='pearson')
        print(f"Actually using only {how_many} nodes")
        #if self.plots:
        #    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        #    plt.title('Pearson Correlation Matrix')
        return correlation_matrix

    def structural_graph(self,iter):
        first_point, last_point = self.indexes[iter]
        Vblock = self.V[first_point:last_point + 1, :] 
        W, edges = cr.compute_graph_MSR(Vblock)
        return W,edges
    
    def node_positions(self,iter,nodes):
        Vblock,Ablock = self.get_block(iter)
        if self.plots:
            _,nodes_fig = visual.border_visualization(Vblock, Ablock, nodes)
            return nodes_fig
        else:
            print("Plots are desactivated")
    
    def get_direction(self,iter):
        Vblock,Ablock = self.get_block(iter)
        _, _, distance_vectors, weights = pt.direction(Vblock, Ablock)
        return Vblock, distance_vectors, weights
    
    def find_borders(self,iter):
        """
        Find the place where the borders of the graph are based
        in the degree of graph
        """
        Vblock,Ablock = self.get_block(iter)
        W, _ = cr.compute_graph_MSR(Vblock)
        DegreeVector = np.sum(W, axis=1)
        sorted_indices = np.argsort(DegreeVector)
        first_threshold = int(0.2 * len(sorted_indices))
        borders_idx = sorted_indices[:first_threshold]
        if self.plots:
            y_values,border_fig = visual.border_visualization(Vblock, Ablock, borders_idx)
        return W, borders_idx,border_fig,y_values
    
    def direction_visualization(self,iter):
        Vblock,distance_vectors,_ = self.get_direction(iter)
        if self.plots:
            dir_fig = visual.direction_visualization(Vblock,distance_vectors)
            return dir_fig
        else:
            print("Plots are desactivated")
    
    def directional_graph(self,iter):
        Vblock, distance_vectors, weights = self.get_direction(iter)
        W, edge, idx_closest = cr.compute_graph_sl(Vblock,distance_vectors,weights)
        return W,edge, idx_closest
    
    
    def gft_transform(self,iter,W,idx_map):
        Vblock,Ablock = self.get_block(iter)
        if not idx_map is None:
            GFT, Gfreq, Ablockhat = tf.compute_GFT_noQ(W,Ablock,idx_closest=idx_map)
            idx = list(idx_map.keys())
            if self.plots:
                _,_ = visual.border_visualization(Vblock, Ablock, idx)
        else:
            GFT, Gfreq, Ablockhat = tf.compute_GFT_noQ(W,Ablock,idx_closest=idx_map)
        return GFT, Gfreq, Ablockhat
    
    def igft_transform(self,iter,W,idx_map,Ablockhat):
        Vblock,Ablock = self.get_block(iter)
        if not idx_map is None:
            GFT_inv, Ablockrec = tf.compute_iGFT_noQ(W,Ablockhat,idx_closest=idx_map)
            idx = list(idx_map.keys())
            if self.plots:
                _,_ = visual.border_visualization(Vblock, Ablock, idx)
        else:
            GFT_inv, Ablockrec = tf.compute_iGFT_noQ(W,Ablockhat,idx_closest=idx_map)
        return GFT_inv, Ablockrec
        
    def dynamic_transform(self,iter,W,idx_map):
        _,_,Ablockhat = self.gft_transform(iter,W,idx_map)
        _,_,nAblockhat = self.gft_transform(iter,W,None)
        if np.abs(Ablockhat[0:0]) > np.abs(nAblockhat[0:0]):
            return Ablockhat,0
        else:
            return nAblockhat,1
        
    def component_projection(self,iter,base,version,num_nodes,c_value):
        Vblock,Ablock = self.get_block(iter)
        if self.plots:
            base_fig = visual.component_visualization(Vblock, base, version,num_nodes,c_value)
            return base_fig
        else:
            print("Plots are desactivated")
    
    def quantize(self,qstep,Coeff):
        Coeff_quant = np.round(Coeff/qstep)
        return Coeff_quant

    def PSNR(self,N,Coeff,Coeff_quant,qstep):
        # Extract the first column of Coeff_quant
        Coeff_dequant = Coeff_quant*qstep
        Coeff_dequant_Y = Coeff_dequant[:, 0]

        # Calculate the norm (Euclidean distance) between Y and Coeff_quant_Y
        norm_value = np.linalg.norm(Coeff - Coeff_dequant_Y)

        # Calculate PSNR
        psnr_Y = -10 * np.log10((norm_value ** 2) / (N * 255 ** 2))
        return psnr_Y
    
    def RLGR(self,Coeff_quant,indexes):
        N = Coeff_quant[:,0].shape[0]
        mask_lo = np.zeros((N), dtype=bool)
        for start_end_tuple in indexes:
            mask_lo[start_end_tuple[0]] = True
        mask_hi = np.logical_not(mask_lo)

        Coeff_quant_lo = Coeff_quant[mask_lo, :]  # DC values
        Coeff_quant_hi = Coeff_quant[mask_hi, :]  # "high" pass values
        # Concatenate
        Coeff_quant_sorted = np.concatenate((Coeff_quant_lo, Coeff_quant_hi))
        
        # Code Y, U, V separately 
        numbits_Y = encode_rlgr(Coeff_quant_sorted[:, 0], os.path.join('res', 'bitstream_Y.bin'))
        numbits_U = encode_rlgr(Coeff_quant_sorted[:, 1], os.path.join('res', 'bitstream_U.bin'))
        numbits_V = encode_rlgr(Coeff_quant_sorted[:, 2], os.path.join('res', 'bitstream_V.bin'))

        # Bit count
        bs_size = numbits_Y + numbits_U + numbits_V
        
        return bs_size
    
    def get_coefficients(self, mode=2, sl_num = 2, sl_weight = 1.2):
        '''
        mode = 0 is the Structural Method
        mode = 1 is the Adaptative Method
        mode = 2 is the Dynamic Method
        '''
        indexes = self.indexes
        A_shape = self.A.shape
        aCoeff = np.zeros(A_shape)
        nCoeff = np.zeros(A_shape)
        dCoeff = np.zeros(A_shape)
        N = A_shape[0]
        for iteration,start_end_tuple in enumerate(indexes):

            # Best nodes for self-loops considering direction of change and Y channel
            sorted_nodes = self.simple_direction_sort(iteration)

            # We get the Adjancency matrix from structural implementation
            W,_ = self.structural_graph(iteration)
            _, _, nAblockhat = self.gft_transform(iteration,W,None)      
            nCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat

            if mode != 0:
                # Choosed nodes to apply self-loops
                choosed_positions = sorted_nodes[:sl_num]
                choosed_weights = [sl_weight]*len(choosed_positions)
                idx_map = dict(zip(choosed_positions,choosed_weights))
                _, _, Ablockhat = self.gft_transform(iteration,W,idx_map)
                aCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = Ablockhat
                DC_criteria = abs(Ablockhat[0,0]) > abs(nAblockhat[0,0]) 
            
            if mode == 1:
                return aCoeff
            
            if mode == 2:
                if DC_criteria:
                    # Reconstruction of coefficients by criteria
                    dAblockhat = np.zeros(Ablockhat.shape)

                    # We get coefficients for the Y channel
                    dAblockhat[:,0] = Ablockhat[:,0]

                    # Different coefficients for the U and V channels
                    dAblockhat[:,1:2] = nAblockhat[:,1:2]
                    
                    # We save the dynamic coefficients
                    dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = dAblockhat
                else:
                    # We save the structural coefficients
                    dCoeff[start_end_tuple[0]:start_end_tuple[1]+1,:] = nAblockhat
                return dCoeff
            
            return nCoeff


    def energy_block(self, Ablockhat,version, c = None):
        # Example data
        Y = abs(Ablockhat[:10, 0])

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Slightly increase the figure size

        # Add supertitle above subplot

        fig.suptitle(f'Energy {version} first 10 bases', fontsize=20, y=0.95)  

        # Plot the Y channel in the subplot
        ax.scatter(range(len(Y)), Y, c='r', label='Y')  # Scatter plot for the Y channel
        ax.set_title('Y Channel', fontsize=16, pad=20)  # Add more padding to the title
        ax.set_xlabel('Index', fontsize=14, labelpad=10)  # Add padding to the xlabel
        ax.set_ylabel('Magnitude', fontsize=14, labelpad=10)  # Add padding to the ylabel

        # Annotate each point with its value
        for i, mag in enumerate(Y):
            # Alternate the vertical offset to reduce overlap
            offset = 10 if i % 2 == 0 else -10
            ax.annotate(f'{mag:.2f}', (i, mag), textcoords="offset points", xytext=(0, offset), ha='center')

        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust layout to prevent overlap, leaving space for the suptitle
        return fig
    
    
def energy_comparison(Coeffs_1, Coeffs_2):  
    # Comparing just the first 5 coefficients DC and 4 AC
    if np.shape(Coeffs_1)[0]>5:
        Y_diff = np.absolute(Coeffs_1[:1,0]) - np.absolute(Coeffs_2[:1,0])
        U_diff = np.absolute(Coeffs_1[:1,1]) - np.absolute(Coeffs_2[:1,1])
        V_diff = np.absolute(Coeffs_1[:1,2]) - np.absolute(Coeffs_2[:1,2])
    else:
        Y_diff = np.absolute(Coeffs_1[:,0]) - np.absolute(Coeffs_2[:,0])
        U_diff = np.absolute(Coeffs_1[:,1]) - np.absolute(Coeffs_2[:,1])
        V_diff = np.absolute(Coeffs_1[:,2]) - np.absolute(Coeffs_2[:,2])

    return np.sum(Y_diff),np.sum(U_diff),np.sum(V_diff)

if __name__ == "__main__":
    V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  
    structural_encoder = StructuralEncoder(V,C_rgb)
    directional_encoder = DirectionalEncoder(V,C_rgb)
    indexes = structural_encoder.block_indexes(block_size = 4)
    directional_encoder.block_indexes(block_size = 4)
    W, edge, idx_closest = directional_encoder.directional_graph(14)
    GFT,Gfreq,Ablockhat =directional_encoder.gft_transform(14,W, None)
    #directional_encoder.energy_block(Ablockhat,"structural")

    print(Ablockhat)



    
    #fig = structural_encoder.energy_block(indexes[14])
    #plt.show()

    