import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Arc
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'
from typing import Optional

import matplotlib.pyplot as plt
from pyvis.network import Network
from matplotlib import cm
from matplotlib.colors import Normalize
from objects import *
from graph import *
# from utils.color import YUVtoRGB

class Visualizer:
    def __init__(self):
        self.colourist = Colourist()


    def __call__(self, graph: Graph, block: Block):
        self.graph = graph
        self.block = block
        self._split_block()
        self.fig = None

    def _split_block(self):
        self.Vblock, self.Ablock = self.block.Vblock, self.block.Ablock
        self.Xblock, self.Yblock, self.Zblock = np.hsplit(self.Vblock, 3)
    

    def _min_max_norm(self, vector: np.ndarray) -> np.ndarray:
        vector_min = np.min(vector)
        vector_max = np.max(vector)
        normalized_base = (vector - vector_min) / (vector_max - vector_min)
        return normalized_base

    def _init_3d_figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scatter plot of block')
        ax.view_init(elev=60, azim=30)
        self.fig = fig 
        self.ax = ax
    
    def _init_2d_figure(self):
        """Initialize a single 2D matplotlib subplot."""
        fig, ax = plt.subplots(figsize=(8, 6))  # Single subplot
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Visualization')
        
        # Store references
        self.fig = fig 
        self.ax = ax  # Single axis object

    def _init_2d_3ch_figure(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))  # 1 row, 3 columns
        
        # Configure first subplot
        ax1.grid(True)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Y Channel')
        
        # Configure second subplot
        ax2.grid(True)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('U Channel')
        
        # Configure third subplot
        ax3.grid(True)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('V Channel')
        
        # Adjust layout to prevent overlap
        fig.tight_layout()
        
        self.fig = fig 
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3


    def visualize_graph(self):
        self._init_3d_figure()
        self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c='k', s=20)
        self.add_graph_edges()

    def visualize_block(self):
        Ablock = self.colourist._YUVtoRGB(self.block.Ablock).astype(float) / 256
        self._init_3d_figure()
        self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c=Ablock, s=20)
        
    def visualize_base(self, base: np.ndarray):
        normalized_base = self._min_max_norm(base)
        self._init_3d_figure()
        sc = self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c=normalized_base, 
                     cmap='inferno', vmin=0, vmax=1, s=50, alpha=0.8)
        self.fig.colorbar(sc)
        
    def visualize_motion_matrix(self):
        assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
        M = self.graph.M
        plt.imshow(self.graph.M)
        plt.colorbar()

    def visualize_sink(self):
        assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
        self._init_3d_figure()
        M = self.graph.M
        np.fill_diagonal(M, np.inf)
        dec_i = np.argmin(M, axis=1)
        N = self.block.Ablock.shape[0]
        for i in range(N):
            og_x, og_y, og_z = self.Xblock[i], self.Yblock[i], self.Zblock[i]
            j = dec_i[i]
            dir_x, dir_y, dir_z = self.Xblock[j], self.Yblock[j], self.Zblock[j]
            self.ax.quiver(og_x, og_y, og_z, dir_x - og_x, dir_y - og_y, dir_z - og_z, color='b', normalize=True)
            self.ax.scatter3D(og_x, og_y, og_z, c= 'gray', s=10)

    def visualize_block_coeffs(self, result: tuple[np.ndarray, np.ndarray],title: str, num_of_coeffs: int = 10, vis_gft: Optional[bool] = False):
        """Visualizes the top GFT coefficients for Y, U, V channels."""
        gft_mat, coeffs = result 
         
        if vis_gft:
            self.visualize_gft(gft_mat)

        self._init_2d_3ch_figure()
        self.fig.suptitle(title)
        
        # Ensure we don't request more coefficients than available
        num_available = min(num_of_coeffs, coeffs.shape[0])
        x = np.arange(1, num_available + 1)  # 1-based indexing
        
        # Channel configurations
        channels = [
            (self.ax1, 'black', 'Y (Luminance)', 0),
            (self.ax2, 'blue', 'U (Chrominance)', 1),
            (self.ax3, 'red', 'V (Chrominance)', 2)
        ]
        
        for ax, color, name, channel_idx in channels:
            # Get coefficients for this channel (N×1 array)
            channel_data = coeffs[:, channel_idx]
            
            # Sort by absolute magnitude (descending)
            sorted_idx = np.argsort(-np.abs(channel_data))[:num_available]
            sorted_coeffs = channel_data[sorted_idx]
            
            # Plot
            ax.scatter(x, sorted_coeffs, color=color, label=name)
            ax.set_title(f'Top {name} Coefficients')
            ax.set_xlabel('Coefficient Index')
            ax.set_ylabel('Magnitude')
            ax.legend()
            ax.grid(True)
            
            # Annotate values
            for i, val in enumerate(sorted_coeffs):
                ax.text(x[i], val, f"{val:.2f}", 
                    ha='center', 
                    va='bottom' if val >= 0 else 'top',
                    fontsize=8, color=color)

        self.fig.tight_layout()

    def visualize_coeffs(self, Coeffs: np.ndarray, title: Optional[str]="Coeffs for all"):
        self._init_2d_3ch_figure()
        self.fig.suptitle(title)

        # Ensure we don't request more coefficients than available
        x = np.arange(1, Coeffs.shape[0]+1)  # 1-based indexing
        
        # Channel configurations
        channels = [
            (self.ax1, 'black', 'Y (Luminance)', 0),
            (self.ax2, 'blue', 'U (Chrominance)', 1),
            (self.ax3, 'red', 'V (Chrominance)', 2)
        ]
        
        for ax, color, name, channel_idx in channels:
            # Get coefficients for this channel (N×1 array)
            channel_data = Coeffs[:, channel_idx]
            
            # Plot
            ax.scatter(x, channel_data, color=color, label=name)
            ax.set_title(f'Top {name} Coefficients')
            ax.set_xlabel('Coefficient Index')
            ax.set_ylabel('Magnitude')
            ax.legend()
            ax.grid(True)
            
        self.fig.tight_layout()

    def visualize_rd(self):
        self.ax.set_title("Rate-Distortion Curve")
        self.ax.set_xlabel("Bits per Voxel (bpv)")
        self.ax.set_ylabel("PSNR (dB)")
        self.ax.grid(True)
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.show()

    def add_rd_data(self, qsteps, bpvs, psnrs, color, label):
        self.ax.plot(bpvs, psnrs, marker='o', linestyle='-', color=color, label=label)
        for q, x, y in zip(qsteps, bpvs, psnrs):
            self.ax.text(x, y, f"q={q}", fontsize=8, ha="right", va="bottom")

    def visualize_gft(self, gft_mat: np.ndarray, title: Optional[str] = "GFT matrix"):
        """Visualize a matrix with colormap and value range display.
        
        Args:
            gft_mat: Matrix to visualize (if None, computes GFT)
            title: Title for the plot
        """
        
        self._init_2d_figure()
        
        # Create the heatmap with colorbar
        im = self.ax.imshow(gft_mat, cmap='viridis', aspect='auto')
        
        # Add colorbar with value range
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.set_label('Magnitude')
        
        # Display min/max values
        min_val = np.min(gft_mat)
        max_val = np.max(gft_mat)
        mean_val = np.mean(gft_mat)
        
        # Add text box with stats
        stats_text = (f"Min: {min_val:.2f}\n"
                    f"Max: {max_val:.2f}\n"
                    f"Mean: {mean_val:.2f}")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        self.ax.text(0.02, 0.98, stats_text, 
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=props)
        
        # Formatting
        self.ax.set_title(title)
        self.ax.set_xlabel('Frequency Components')
        self.ax.set_ylabel('Spatial Components')
        
        # Adjust layout to prevent text overlap
        self.fig.tight_layout()

    def add_graph_edges(self):
        for edge in self.graph.edges:
            x_start, y_start, z_start = self.Vblock[edge[0]]
            x_end, y_end, z_end = self.Vblock[edge[1]]
            weight = self.graph.weights[edge[0], edge[1]]/4
            if edge[0] == edge[1]:
                loop_radius = 0.25  # Adjust radius for self-loop
                theta = np.linspace(0, 2 * np.pi, 100)  # Parametric angle for a full loop
                loop_x = x_start -loop_radius/np.sqrt(2) + loop_radius * np.cos(theta)  # x-values for the loop
                loop_y = y_start -loop_radius/np.sqrt(2) + loop_radius * np.sin(theta)  # y-values for the loop
                loop_z = z_start   # Keep the z-coordinate constant for a horizontal loop

                # Create a loop (self-loop) with multiple small line segments
                self.ax.plot(loop_x, loop_y, loop_z,color = 'r',  lw=weight)  # Self-loop in red dashed lineedge_arc = Arc()
            else:
                
                edge_line = Line3D([x_start, x_end], [y_start, y_end], [z_start, z_end], color='k', lw=weight)
                self.ax.add_line(edge_line)


    def add_selected_nodes(self):
        assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
        selected_nodes = self.graph.selected_nodes
        self.ax.scatter3D(self.Xblock[selected_nodes], self.Yblock[selected_nodes], self.Zblock[selected_nodes]
                          ,c='cyan', s =120, alpha=0.3)
            
    def display(self):
        plt.show()

    def close(self):
        plt.close()
         

if __name__ == "__main__":
    pass