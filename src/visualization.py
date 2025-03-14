import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.patches import Arc
matplotlib.use('Qt5Agg')  # or 'Qt5Agg'


import matplotlib.pyplot as plt
from pyvis.network import Network
from matplotlib import cm
from matplotlib.colors import Normalize
from graph.properties import block_indices
from src.objects import *
from src.graph import *
from utils.color import YUVtoRGB
from graph.properties import direction,gradient,simple_direction
from graph.create import compute_graph_sl

class Visualizer:
    def __init__(self):
        self.__init__transformations()

    def __init__transformations(self):
        self.Q_RGBtoYUV = np.array(
            [
                [0.29899999, -0.1687, 0.5],
                [0.587, -0.3313, -0.4187],
                [0.114, 0.5, -0.0813],
                [0, 0.50196078, 0.50196078],
            ]
        )

        self.M_YUVtoRGB = np.array(
            [
                [1, 1, 1],
                [0, -0.34414, 1.772],
                [1.402, -0.71414, 0],
                [-0.703749019, 0.53121505, -0.88947451],
            ]
        )

    def __call__(self, graph: Graph, block: Block):
        self.graph = graph
        self.block = block
        self._split_block()
        self.fig = None

    def _split_block(self):
        self.Vblock, self.Ablock = self.block.Vblock, self.block.Ablock
        self.Xblock, self.Yblock, self.Zblock = np.hsplit(self.Vblock, 3)
    
    def _YUVtoRGB(self, rounding: bool = True):
        A_yuv = self.block.Ablock
        A_yuv_1 = np.concatenate((A_yuv / 255, np.ones((A_yuv.shape[0], 1))), axis=1)
        A_rgb = np.dot(A_yuv_1, self.M_YUVtoRGB)
        A_rgb = 255 * np.clip(A_rgb, 0, 1)
        if rounding:
            A_rgb = A_rgb.round().astype(np.uint8)

        return A_rgb
    
    def _RGBtoYUV(self, rounding=False) -> np.ndarray:
        A_rgb = self.block.Ablock
        A_rgb_1 = np.concatenate((A_rgb / 255, np.ones((A_rgb.shape[0], 1))), axis=1)
        A_yuv = np.dot(A_rgb_1, self.Q_RGBtoYUV)
        A_yuv = 255 * np.clip(A_yuv, 0, 1)
        if rounding:
            A_yuv = A_yuv.round().astype(np.uint8)
        return A_yuv

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
    
    def visualize_graph(self):
        self._init_3d_figure()
        self.ax.scatter3D(self.Xblock, self.Yblock, self.Zblock, c='k', s=20)
        self.add_graph_edges()

    def visualize_block(self):
        Ablock = self._YUVtoRGB().astype(float) / 256
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