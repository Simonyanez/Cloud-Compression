import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Arc
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'


import matplotlib.pyplot as plt
from pyvis.network import Network
from matplotlib import cm
from matplotlib.colors import Normalize
from objects import *
from graph import *
from transforms import *
# from utils.color import YUVtoRGB

class Visualizer:
    def __init__(self):
        self.gft_computer = GFT()
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

    def visualize_coeffs(self,title: str,  num_of_coeffs: int = 10):
        """Visualizes the top GFT coefficients for Y, U, V channels with values annotated."""
        self._init_2d_figure()  # Initialize the 2D figure with 3 subplots
        self.fig.suptitle(title)    
        # Compute GFT coefficients (shape: [num_coeffs, 3] where columns are Y, U, V)
        _, coeffs = self.gft_computer(self.graph, self.block)
        
        # Sort coefficients in descending order (magnitude) per channel
        sorted_coeffs = np.sort(np.abs(coeffs), axis=0)[::-1]  # [num_coeffs, 3]
        
        # X-axis (1 to num_of_coeffs)
        x = np.arange(1, num_of_coeffs + 1)
        
        # --- Plot 1: Y Channel (Luminance) ---
        sc1 = self.ax1.scatter(x, sorted_coeffs[:num_of_coeffs, 0], color='black', label='Y (Luminance)')
        self.ax1.set_title('Top Y Channel Coefficients')
        self.ax1.set_xlabel('Coefficient Index')
        self.ax1.set_ylabel('Magnitude')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Add text labels for Y values
        for i, (xi, yi) in enumerate(zip(x, sorted_coeffs[:num_of_coeffs, 0])):
            self.ax1.text(xi, yi, f"{yi:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        
        # --- Plot 2: U Channel (Chrominance) ---
        sc2 = self.ax2.scatter(x, sorted_coeffs[:num_of_coeffs, 1], color='blue', label='U (Chrominance)')
        self.ax2.set_title('Top U Channel Coefficients')
        self.ax2.set_xlabel('Coefficient Index')
        self.ax2.set_ylabel('Magnitude')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Add text labels for U values
        for i, (xi, yi) in enumerate(zip(x, sorted_coeffs[:num_of_coeffs, 1])):
            self.ax2.text(xi, yi, f"{yi:.2f}", ha='center', va='bottom', fontsize=8, color='blue')
        
        # --- Plot 3: V Channel (Chrominance) ---
        sc3 = self.ax3.scatter(x, sorted_coeffs[:num_of_coeffs, 2], color='red', label='V (Chrominance)')
        self.ax3.set_title('Top V Channel Coefficients')
        self.ax3.set_xlabel('Coefficient Index')
        self.ax3.set_ylabel('Magnitude')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Add text labels for V values
        for i, (xi, yi) in enumerate(zip(x, sorted_coeffs[:num_of_coeffs, 2])):
            self.ax3.text(xi, yi, f"{yi:.2f}", ha='center', va='bottom', fontsize=8, color='red')
        
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
         

if __name__ == "__main__":
    pass