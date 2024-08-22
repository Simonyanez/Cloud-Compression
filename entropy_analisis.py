from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB
from scipy.interpolate import UnivariateSpline
import pandas as pd

if __name__ == "__main__":
    ply_file = "res/longdress_vox10_1051.ply"
    steps = [8,12,16,20,26,32,40,52,64,78,90,128,256]
    block_sizes = [4,8,16]
    #point_fractions = [0.05]#,0.2,0.5]
    num_of_points=[1,2,4,8,16]
    weights = [1.2]#,1.6,2.0]
    V,C_rgb,_ = ply.ply_read8i(ply_file)
    N = V.shape[0]
    data = []
    directional_encoder = DirectionalEncoder(V,C_rgb)
    for bsize in block_sizes:
        print(f"========================================================= \n Block size {bsize} \n =========================================================")
        directional_encoder.block_indexes(block_size=bsize)
        indexes = directional_encoder.indexes
        sl_in_morton_positions = np.full((len(indexes),20),-1,dtype=np.float32)
        count = 0
        for iteration,_ in  enumerate(indexes):
            Vblock,_ = directional_encoder.get_block(iteration)
            N = Vblock.shape[0]
            # Normalize mordon order by total of points
            sorted_nodes = directional_encoder.simple_direction_sort(iteration)
            # We get the Adjancency matrix from structural implementation
            W,_ = directional_encoder.structural_graph(iteration)
            _, _, nAblockhat = directional_encoder.gft_transform(iteration,W,None)      

            # Choosed nodes to apply self-loops
            choosed_positions = sorted_nodes[:5]
            choosed_weights = [1.6]*len(choosed_positions)
            idx_map = dict(zip(choosed_positions,choosed_weights))
            _, _, Ablockhat = directional_encoder.gft_transform(iteration,W,idx_map)
            DC_criteria = abs(Ablockhat[0,0]) > abs(nAblockhat[0,0]) 
            if DC_criteria:
                count+=1
                first_five_or_less = [position_i for position_i in sorted_nodes[:5]]
                sl_in_morton_positions[iteration,:len(first_five_or_less)] = first_five_or_less 

        cmap = plt.get_cmap('tab10')
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i in range(sl_in_morton_positions.shape[1]):
            first_sl_by_morton_position = sl_in_morton_positions[:, i]
            first_sl_by_morton_position = first_sl_by_morton_position[first_sl_by_morton_position > 0.0]
            
            if first_sl_by_morton_position.size > 0:
                # Histogram
                counts, bins = np.histogram(first_sl_by_morton_position, bins='auto', density=True)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                
                # Calculate PDF from histogram
                pdf = counts / np.sum(counts)
                
                # Plot histogram on the upper subplot
                axes[0].stairs(counts, bins, color=cmap(i % 10), label=f'Position {i+1}')
                
                # Spline fitting
                spline = UnivariateSpline(bin_centers, counts, s=1)
                x_spline = np.linspace(0, np.max(first_sl_by_morton_position), 500)
                y_spline = spline(x_spline)
                
                # Normalize the spline curve
                y_spline /= np.trapz(y_spline, x_spline)
                
                axes[1].plot(x_spline, y_spline, color=cmap(i % 10), linestyle='--', label=f'Spline Fit Position {i+1}')
                
                # Calculate entropy
                # Use histogram counts to estimate entropy
                non_zero_pdf = pdf[pdf > 0]  # Avoid log(0)
                entropy = -np.sum(non_zero_pdf * np.log(non_zero_pdf))
                print(f'Entropy for position {i+1}: {entropy}')
                print(f'All Blocks Bits Estimation for position {i+1}: {entropy*count}')
                data.append([bsize,i+1,entropy,entropy*count])

        # Add labels and a legend
        axes[0].set_ylabel('Density (Histogram)')
        axes[1].set_ylabel('Density (Spline Fit)')
        axes[1].set_xlabel('Morton Position')
        
        axes[0].set_title(f'Histograms of First 5 Morton Positions in each Block \n Block Size {bsize}')
        axes[1].set_title('Spline Fit to Histogram')
        
        axes[0].legend()
        plt.show()

    df = pd.DataFrame(data=data,columns=["Block Size","Self-Loop Priority","Entropy","Overhead Stimation"])
    df.to_csv("entropy_analysis.csv")