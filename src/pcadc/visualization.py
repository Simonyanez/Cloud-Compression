import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Arc
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from .color import *
from .graph import *


class VisualizationType(Enum):
    BLOCK_OVERVIEW = "block_overview"
    COEFFICIENT_ANALYSIS = "coefficient_analysis"
    RDO_COMPARISON = "rdo_comparison"
    FIT_ANALYSIS = "fit_analysis"
    GRAPH_ANALYSIS = "graph_analysis"

class Visualizer:
    def __init__(self):
        self.colourist = Colourist()
        self._reset_state()

    def _reset_state(self):
        """Clear internal state for new visualization"""
        self.fig = None
        self.ax = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.vertices = None
        self.attributes = None
        self.graph = None
        self.x_coords = None
        self.y_coords = None
        self.z_coords = None

        
    def _set_figure(self, fig: Figure) -> 'Visualizer':
        """Internal helper to set figure"""
        self.fig = fig
        return self

    def with_data(self, vertices: np.ndarray, attributes: np.ndarray) -> 'Visualizer':
        """Set vertex and attribute data"""
        self.vertices = vertices
        self.attributes = attributes
        self._split_vertices()
        return self

    def with_graph(self, graph) -> 'Visualizer':
        """Set graph structure"""
        self.graph = graph
        return self

    def _split_vertices(self):
        """Split vertices into X, Y, Z components"""
        if self.vertices is not None:
            self.x_coords, self.y_coords, self.z_coords = np.hsplit(self.vertices, 3)

    # Setup methods (modified to work with builder pattern)
    def setup_3d_plot(self, title: str = "3D Visualization", figsize: tuple = (8, 6)) -> 'Visualizer':
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.grid(True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(title)
        self.ax.view_init(elev=60, azim=30)
        return self

    def setup_2d_plot(self, title: str = "2D Visualization", figsize: tuple = (8, 6)) -> 'Visualizer':
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.grid(True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(title)
        return self

    def setup_multichannel_plot(self, title: str = "Multi-channel", figsize: tuple = (15, 5)) -> 'Visualizer':
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=figsize)
        
        channels = [('Y (Luminance)', self.ax1), ('U (Chrominance)', self.ax2), ('V (Chrominance)', self.ax3)]
        for name, ax in channels:
            ax.grid(True)
            ax.set_title(name)
        
        self.fig.suptitle(title)
        return self

    def setup_custom_grid(self, rows: int, cols: int, figsize: tuple = (15, 10), title: str = "") -> 'Visualizer':
        """Setup custom subplot grid"""
        self.fig = plt.figure(figsize=figsize)
        if title:
            self.fig.suptitle(title, fontsize=16)
        return self

    # Plotting methods (return self for chaining)
    def plot_point_cloud(self, color_channel: Optional[int] = None, size: int = 20, alpha: float = 0.8) -> 'Visualizer':

        """Plot 3D point cloud, optionally colored by attribute channel"""
        if self.vertices is None:
            raise ValueError("No vertex data set. Call with_data() first.")
        if self.ax is None:
            raise ValueError("No 3D plot setup. Call setup_3d_plot() first.")

        if color_channel is not None and self.attributes is not None:
            colors = self.attributes[:, color_channel]
            scatter = self.ax.scatter3D(
                self.x_coords.flatten(), 
                self.y_coords.flatten(), 
                self.z_coords.flatten(), 
                c=colors, s=size, alpha=alpha, cmap='gray'
            )
            self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5)
        else:
            self.ax.scatter3D(
                self.x_coords.flatten(), 
                self.y_coords.flatten(), 
                self.z_coords.flatten(), 
                c='black', s=size, alpha=alpha
            )
        return self

    def plot_colored_block(self, size: int = 40) -> 'Visualizer':
        """Plot block with RGB colors"""
        if self.attributes is None:
            raise ValueError("No attribute data set. Call with_data() first.")
        
        rgb_colors = self.colourist._YUVtoRGB(self.attributes).astype(float) / 255.0
        
        self.ax.scatter3D(
            self.x_coords.flatten(),
            self.y_coords.flatten(), 
            self.z_coords.flatten(),
            c=rgb_colors, s=size
        )
        return self

    def plot_graph_edges(self) -> 'Visualizer':
        """Add graph edges to current 3D plot"""
        if self.graph is None:
            raise ValueError("No graph set. Call with_graph() first.")
        if self.ax is None:
            raise ValueError("No 3D plot setup. Call setup_3d_plot() first.")

        # FIXME: Edges could be bad implemented
        for edge in self.graph.edges:
            start_pos = self.vertices[edge[0]]
            end_pos = self.vertices[edge[1]]
            weight = self.graph.weights[edge[0], edge[1]] / 4
            
            if edge[0] == edge[1]:  # Self-loop
                self._plot_self_loop(start_pos, weight)
            else:
                self._plot_edge(start_pos, end_pos, weight)
        return self

    def plot_basis_function(self, basis_values: np.ndarray, colormap: str = 'inferno') -> 'Visualizer':
        """Plot basis function values on vertices"""
        if self.vertices is None:
            raise ValueError("No vertex data set.")
        
        normalized_values = self._min_max_norm(basis_values)
        scatter = self.ax.scatter3D(
            self.x_coords.flatten(),
            self.y_coords.flatten(), 
            self.z_coords.flatten(),
            c=normalized_values, 
            cmap=colormap, 
            vmin=0, vmax=1, 
            s=50, alpha=0.8
        )
        self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5)
        return self

    def plot_coefficients(self, coeffs: np.ndarray, num_coeffs: int = 10, plot_type: str = 'scatter') -> 'Visualizer':
        """Plot coefficient magnitudes for Y, U, V channels"""
        if not all([self.ax1, self.ax2, self.ax3]):
            raise ValueError("Multi-channel plot not set up. Call setup_multichannel_plot() first.")

        num_available = min(num_coeffs, coeffs.shape[0])
        x = np.arange(1, num_available + 1)
        
        channels = [
            (self.ax1, 'black', 'Y (Luminance)', 0),
            (self.ax2, 'blue', 'U (Chrominance)', 1),
            (self.ax3, 'red', 'V (Chrominance)', 2)
        ]
        
        for ax, color, name, channel_idx in channels:
            if channel_idx < coeffs.shape[1]:
                channel_data = coeffs[:, channel_idx]
                
                if plot_type == 'energy_compaction':
                    # Sort by absolute magnitude (descending)
                    sorted_idx = np.argsort(-np.abs(channel_data))[:num_available]
                    sorted_coeffs = channel_data[sorted_idx]
                    values = sorted_coeffs
                else:
                    values = channel_data[:num_available]
                
                if plot_type == 'scatter':
                    ax.scatter(x, values, color=color, label=name)
                else:
                    ax.bar(x, values, color=color, alpha=0.7, label=name)
                
                ax.set_xlabel('Coefficient Index')
                ax.set_ylabel('Magnitude')
                ax.legend()
                ax.grid(True)
        return self

    def plot_matrix_heatmap(self, matrix: np.ndarray, title: str = "Matrix", colormap: str = 'viridis') -> 'Visualizer':
        """Plot matrix as heatmap with statistics"""
        if self.ax is None:
            raise ValueError("No 2D plot setup. Call setup_2d_plot() first.")
            
        im = self.ax.imshow(matrix, cmap=colormap, aspect='auto')
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.set_label('Magnitude')
        
        # Add statistics
        stats_text = (f"Min: {matrix.min():.2f}\n"
                     f"Max: {matrix.max():.2f}\n"
                     f"Mean: {matrix.mean():.2f}")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        self.ax.text(0.02, 0.98, stats_text, 
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=props)
        
        self.ax.set_title(title)
        self.ax.set_xlabel('Frequency Components')
        self.ax.set_ylabel('Spatial Components')
        return self

    def plot_rate_distortion(self, rates: List[float], distortions: List[float], 
                           selected_idx: int = None, labels: List[str] = None) -> 'Visualizer':
        """Plot rate-distortion scatter"""
        if self.ax is None:
            raise ValueError("No 2D plot setup. Call setup_2d_plot() first.")
        
        colors = ['red' if i == selected_idx else 'blue' for i in range(len(rates))]
        sizes = [100 if i == selected_idx else 50 for i in range(len(rates))]
        
        self.ax.scatter(rates, distortions, c=colors, s=sizes, alpha=0.7)
        
        # Add labels if provided
        if labels:
            for i, (r, d, label) in enumerate(zip(rates, distortions, labels)):
                self.ax.annotate(label, (r, d), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        self.ax.set_xlabel('Rate')
        self.ax.set_ylabel('Distortion')
        self.ax.grid(True, alpha=0.3)
        return self

    # Utility methods
    def build(self) -> Figure:
        """Finalize and return the current figure"""
        if self.fig is None:
            raise ValueError("No figure created. Set up a plot first.")
        
        plt.tight_layout()
        figure = self.fig
        self._reset_state()  # Clean slate for next visualization
        return figure

    def _min_max_norm(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to [0, 1] range"""
        vector_min = np.min(vector)
        vector_max = np.max(vector)
        if vector_max == vector_min:
            return np.zeros_like(vector)
        return (vector - vector_min) / (vector_max - vector_min)

    def _plot_self_loop(self, position: np.ndarray, weight: float):
        """Plot self-loop at given position"""
        x, y, z = position
        loop_radius = 0.25
        theta = np.linspace(0, 2 * np.pi, 100)
        loop_x = x - loop_radius/np.sqrt(2) + loop_radius * np.cos(theta)
        loop_y = y - loop_radius/np.sqrt(2) + loop_radius * np.sin(theta)
        loop_z = np.full_like(loop_x, z)
        
        self.ax.plot(loop_x, loop_y, loop_z, color='r', lw=weight)

    def _plot_edge(self, start: np.ndarray, end: np.ndarray, weight: float):
        """Plot edge between two positions with given weight"""
        x_start, y_start, z_start = start
        x_end, y_end, z_end = end
        edge_line = Line3D([x_start, x_end], [y_start, y_end], [z_start, z_end], 
                          color='k', lw=weight)
        self.ax.add_line(edge_line)






# Example usage:
if __name__ == "__main__":
    # Traditional builder pattern usage
    viz = Visualizer()
    fig1 = (viz
            .with_data(vertices, attributes)
            .with_graph(graph)
            .setup_3d_plot("Block with Graph")
            .plot_point_cloud(color_channel=0)
            .plot_graph_edges()
            .build())
    
    # Pipeline manager usage
    manager = VisualizationManager()
    fig2 = manager.create_visualization(
        VisualizationType.COEFFICIENT_ANALYSIS,
        vertices=vertices,
        attributes=attributes,
        coeffs=coeffs,
        gft_matrix=gft_matrix,
        graph=graph,
        name="Block Analysis"
    )
    
    # Show both figures
    fig1.show()
    fig2.show()
# import numpy as np
# import matplotlib
# from mpl_toolkits.mplot3d.art3d import Line3D
# from matplotlib.patches import Arc
# matplotlib.use('Qt5Agg')  # or 'Qt5Agg'
# from typing import Optional
#
# import matplotlib.pyplot as plt
# from pyvis.network import Network
# from matplotlib import cm
# from matplotlib.colors import Normalize
# from .color import *
# from .graph import *
# # from utils.color import YUVtoRGB
#
# # TODO: FIX VISUALIZER
# class Visualizer:
#     def __init__(self):
#         self.colourist = Colourist()
#
#
#     def __call__(self, graph: Graph, block: Block):
#         self.graph = graph
#         self.block = block
#         self.Vblock, self.Ablock = self.block.Vblock, self.block.Ablock
#         self._split_block()
#         self.fig = None
#
#     def _split_block(self):
#         self.Xblock, self.Yblock, self.Zblock = np.hsplit(self.Vblock, 3)
#
#
#     def _min_max_norm(self, vector: np.ndarray) -> np.ndarray:
#         vector_min = np.min(vector)
#         vector_max = np.max(vector)
#         normalized_base = (vector - vector_min) / (vector_max - vector_min)
#         return normalized_base
#
#     def _init_3d_figure(self, title: str):
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection = '3d')
#         ax.grid(True)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title(title)
#         ax.view_init(elev=60, azim=30)
#         self.fig = fig 
#         self.ax = ax
#
#     def _init_2d_figure(self):
#         """Initialize a single 2D matplotlib subplot."""
#         fig, ax = plt.subplots(figsize=(8, 6))  # Single subplot
#         ax.grid(True)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_title('2D Visualization')
#
#         # Store references
#         self.fig = fig 
#         self.ax = ax  # Single axis object
#
#     def _init_2d_3ch_figure(self):
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))  # 1 row, 3 columns
#
#         # Configure first subplot
#         ax1.grid(True)
#         ax1.set_xlabel('X')
#         ax1.set_ylabel('Y')
#         ax1.set_title('Y Channel')
#
#         # Configure second subplot
#         ax2.grid(True)
#         ax2.set_xlabel('X')
#         ax2.set_ylabel('Z')
#         ax2.set_title('U Channel')
#
#         # Configure third subplot
#         ax3.grid(True)
#         ax3.set_xlabel('Y')
#         ax3.set_ylabel('Z')
#         ax3.set_title('V Channel')
#
#         # Adjust layout to prevent overlap
#         fig.tight_layout()
#
#         self.fig = fig 
#         self.ax1 = ax1
#         self.ax2 = ax2
#         self.ax3 = ax3
#
#     # NOTE: All transformations of Vblock are applied in visualizer class only so it doesn't modify original data
#     def set_Vblock(self, Vblock_new: np.ndarray):
#         self.Vblock = Vblock_new
#         self._split_block()
#
#     def set_Ablock(self, Ablock_new: np.ndarray):
#         self.Ablock = Ablock_new
#         self._split_block()
#
#     def visualize_graph(self, title: str = "Graph Visualization"):
#         self._init_3d_figure(title)
#         self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c='k', s=20)
#         self.add_graph_edges()
#
#     def visualize_block(self, title: str = "Block Visualization"):
#         Ablock = self.colourist._YUVtoRGB(self.block.Ablock).astype(float) / 256
#         self._init_3d_figure(title)
#         self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c=Ablock, s=40)
#
#     def visualize_base(self, base: np.ndarray, title: str = "Base Visualization"):
#         normalized_base = self._min_max_norm(base)
#         self._init_3d_figure(title)
#         sc = self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c=normalized_base, 
#                      cmap='inferno', vmin=0, vmax=1, s=50, alpha=0.8)
#         self.fig.colorbar(sc)
#
#     def visualize_motion_matrix(self):
#         assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
#         M = self.graph.M
#         plt.imshow(self.graph.M)
#         plt.colorbar()
#
#     def visualize_sink(self):
#         assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
#         self._init_3d_figure()
#         M = self.graph.M
#         np.fill_diagonal(M, np.inf)
#         dec_i = np.argmin(M, axis=1)
#         N = self.block.Ablock.shape[0]
#         for i in range(N):
#             og_x, og_y, og_z = self.Xblock[i], self.Yblock[i], self.Zblock[i]
#             j = dec_i[i]
#             dir_x, dir_y, dir_z = self.Xblock[j], self.Yblock[j], self.Zblock[j]
#             self.ax.quiver(og_x, og_y, og_z, dir_x - og_x, dir_y - og_y, dir_z - og_z, color='b', normalize=True)
#             self.ax.scatter3D(og_x, og_y, og_z, c= 'gray', s=10)
#
#     def visualize_block_coeffs(self, result: tuple[np.ndarray, np.ndarray],title: str = "Energy Compaction", num_of_coeffs: int = 10, vis_gft: Optional[bool] = False):
#         """Visualizes the top GFT coefficients for Y, U, V channels."""
#         gft_mat, coeffs = result 
#
#         if vis_gft:
#             self.visualize_gft(gft_mat)
#
#         self._init_2d_3ch_figure()
#         self.fig.suptitle(title)
#
#         # Ensure we don't request more coefficients than available
#         num_available = min(num_of_coeffs, coeffs.shape[0])
#         x = np.arange(1, num_available + 1)  # 1-based indexing
#
#         # Channel configurations
#         channels = [
#             (self.ax1, 'black', 'Y (Luminance)', 0),
#             (self.ax2, 'blue', 'U (Chrominance)', 1),
#             (self.ax3, 'red', 'V (Chrominance)', 2)
#         ]
#
#         for ax, color, name, channel_idx in channels:
#             # Get coefficients for this channel (N×1 array)
#             channel_data = coeffs[:, channel_idx]
#
#             # Sort by absolute magnitude (descending)
#             sorted_idx = np.argsort(-np.abs(channel_data))[:num_available]
#             sorted_coeffs = channel_data[sorted_idx]
#
#             # Plot
#             ax.scatter(x, sorted_coeffs, color=color, label=name)
#             ax.set_title(f'Top {name} Coefficients')
#             ax.set_xlabel('Coefficient Index')
#             ax.set_ylabel('Magnitude')
#             ax.legend()
#             ax.grid(True)
#
#             # Annotate values
#             for i, val in enumerate(sorted_coeffs):
#                 ax.text(x[i], val, f"{val:.2f}", 
#                     ha='center', 
#                     va='bottom' if val >= 0 else 'top',
#                     fontsize=8, color=color)
#
#         self.fig.tight_layout()
#
#     def visualize_coeffs(self, Coeffs: np.ndarray, title: Optional[str]="Coeffs for all"):
#         self._init_2d_3ch_figure()
#         self.fig.suptitle(title)
#
#         # Ensure we don't request more coefficients than available
#         x = np.arange(1, Coeffs.shape[0]+1)  # 1-based indexing
#
#         # Channel configurations
#         channels = [
#             (self.ax1, 'black', 'Y (Luminance)', 0),
#             (self.ax2, 'blue', 'U (Chrominance)', 1),
#             (self.ax3, 'red', 'V (Chrominance)', 2)
#         ]
#
#         for ax, color, name, channel_idx in channels:
#             # Get coefficients for this channel (N×1 array)
#             channel_data = Coeffs[:, channel_idx]
#
#             # Plot
#             ax.scatter(x, channel_data, color=color, label=name)
#             ax.set_title(f'Top {name} Coefficients')
#             ax.set_xlabel('Coefficient Index')
#             ax.set_ylabel('Magnitude')
#             ax.legend()
#             ax.grid(True)
#
#         self.fig.tight_layout()
#
#     def visualize_rd(self):
#         self.ax.set_title("Rate-Distortion Curve")
#         self.ax.set_xlabel("Bits per Voxel (bpv)")
#         self.ax.set_ylabel("PSNR (dB)")
#         self.ax.grid(True)
#         self.ax.legend()
#         self.fig.tight_layout()
#         self.fig.show()
#
#     def add_rd_data(self, qsteps, bpvs, psnrs, color, label, linestyle):
#         self.ax.plot(bpvs, psnrs, marker='o', linestyle=linestyle, color=color, label=label)
#         for q, x, y in zip(qsteps, bpvs, psnrs):
#             self.ax.text(x, y, f"q={q}", fontsize=8, ha="right", va="bottom")
#
#     def visualize_gft(self, gft_mat: np.ndarray, title: Optional[str] = "GFT matrix"):
#         """Visualize a matrix with colormap and value range display.
#
#         Args:
#             gft_mat: Matrix to visualize (if None, computes GFT)
#             title: Title for the plot
#         """
#
#         self._init_2d_figure()
#
#         # Create the heatmap with colorbar
#         im = self.ax.imshow(gft_mat, cmap='viridis', aspect='auto')
#
#         # Add colorbar with value range
#         cbar = self.fig.colorbar(im, ax=self.ax)
#         cbar.set_label('Magnitude')
#
#         # Display min/max values
#         min_val = np.min(gft_mat)
#         max_val = np.max(gft_mat)
#         mean_val = np.mean(gft_mat)
#
#         # Add text box with stats
#         stats_text = (f"Min: {min_val:.2f}\n"
#                     f"Max: {max_val:.2f}\n"
#                     f"Mean: {mean_val:.2f}")
#         props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#         self.ax.text(0.02, 0.98, stats_text, 
#                     transform=self.ax.transAxes,
#                     verticalalignment='top',
#                     bbox=props)
#
#         # Formatting
#         self.ax.set_title(title)
#         self.ax.set_xlabel('Frequency Components')
#         self.ax.set_ylabel('Spatial Components')
#
#         # Adjust layout to prevent text overlap
#         self.fig.tight_layout()
#
#     def add_graph_edges(self):
#         for edge in self.graph.edges:
#             x_start, y_start, z_start = self.Vblock[edge[0]]
#             x_end, y_end, z_end = self.Vblock[edge[1]]
#             weight = self.graph.weights[edge[0], edge[1]]/4
#             if edge[0] == edge[1]:
#                 loop_radius = 0.25  # Adjust radius for self-loop
#                 theta = np.linspace(0, 2 * np.pi, 100)  # Parametric angle for a full loop
#                 loop_x = x_start -loop_radius/np.sqrt(2) + loop_radius * np.cos(theta)  # x-values for the loop
#                 loop_y = y_start -loop_radius/np.sqrt(2) + loop_radius * np.sin(theta)  # y-values for the loop
#                 loop_z = z_start   # Keep the z-coordinate constant for a horizontal loop
#
#                 # Create a loop (self-loop) with multiple small line segments
#                 self.ax.plot(loop_x, loop_y, loop_z,color = 'r',  lw=weight)  # Self-loop in red dashed lineedge_arc = Arc()
#             else:
#
#                 edge_line = Line3D([x_start, x_end], [y_start, y_end], [z_start, z_end], color='k', lw=weight)
#                 self.ax.add_line(edge_line)
#
#
#     def add_selected_nodes(self):
#         assert isinstance(self.graph, AttributeGraph), "This visualizations is for AttributeGraph only"
#         selected_nodes = self.graph.selected_nodes
#         self.ax.scatter3D(
#             self.Xblock[selected_nodes],
#             self.Yblock[selected_nodes],
#             self.Zblock[selected_nodes],
#             facecolors='none',       # Hollow inside
#             edgecolors='cyan',       # Cyan outline
#             s=120,
#             alpha=0.8                # You can tweak this
# )
#
#     def display(self):
#         plt.show()
#
#     def close(self):
#         plt.close()
#
#
# if __name__ == "__main__":
#     pass
