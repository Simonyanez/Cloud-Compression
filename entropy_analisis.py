from src.encoder import *
import matplotlib.pyplot as plt
from utils.color import YUVtoRGB
from scipy.interpolate import UnivariateSpline

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
                first_twenty_or_less = [round(position_i/N,2) for position_i in sorted_nodes[:5]]
                sl_in_morton_positions[iteration,:len(first_twenty_or_less)] = first_twenty_or_less 

        cmap = plt.get_cmap('tab10')
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i in range(sl_in_morton_positions.shape[1]):
            first_sl_by_morton_position = sl_in_morton_positions[:, i]
            first_sl_by_morton_position = first_sl_by_morton_position[first_sl_by_morton_position > 0.0]
            if first_sl_by_morton_position.size > 0:
                # Histogram
                counts, bins = np.histogram(first_sl_by_morton_position, bins=100, range=(0, 1), density=True)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                
                # Plot histogram on the upper subplot
                axes[0].stairs(counts, bins, color=cmap(i % 10), label=f'Position {i+1}')
                
                # Spline fitting
                spline = UnivariateSpline(bin_centers, counts, s=1)
                x_spline = np.linspace(0, 1, 500)
                y_spline = spline(x_spline)
                
                # Normalize the spline curve
                y_spline /= np.trapz(y_spline, x_spline)  # Ensure the area under the curve sums to 1
                
                axes[1].plot(x_spline, y_spline, color=cmap(i % 10), linestyle='--', label=f'Spline Fit Position {i+1}')
        
        # Add labels and a legend
        axes[0].set_ylabel('Density (Histogram)')
        axes[1].set_ylabel('Density (Spline Fit)')
        axes[1].set_xlabel('Normalized Morton Position')
        
        axes[0].set_title(f'Histograms of First 5 Morton Positions in each Block \n Block Size {bsize}')
        axes[1].set_title('Spline Fit to Histogram')
        
        axes[0].legend()
        plt.show()