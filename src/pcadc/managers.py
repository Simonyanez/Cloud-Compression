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
from .visualization import *

@dataclass
class VisualizationPipeline:
    """Defines a standard visualization pipeline"""
    name: str
    steps: List[Callable[['VisualizationManager', 'Visualizer', dict], 'Visualizer']]
    figure_size: tuple = (15, 10)
    title_template: str = "{name}"

class VisualizationManager:
    def __init__(self):
        self.visualizer = Visualizer()
        self.pipelines: Dict[VisualizationType, VisualizationPipeline] = {}
        self._register_standard_pipelines()

    def _register_standard_pipelines(self):
        """Register common visualization patterns"""
        
        # Block overview pipeline
        self.pipelines[VisualizationType.BLOCK_OVERVIEW] = VisualizationPipeline(
            name="Block Overview",
            steps=[
                self._setup_block_overview,
                self._add_original_block,
                self._add_y_channel_view,
                self._add_attribute_stats
            ],
            figure_size=(18, 6)
        )
        
        # Coefficient analysis pipeline
        self.pipelines[VisualizationType.COEFFICIENT_ANALYSIS] = VisualizationPipeline(
            name="Coefficient Analysis", 
            steps=[
                self._setup_coeff_analysis,
                self._add_energy_compaction,
                self._add_gft_matrix,
                self._add_graph_structure
            ],
            figure_size=(20, 12)
        )
        
        # Fit analysis pipeline
        self.pipelines[VisualizationType.FIT_ANALYSIS] = VisualizationPipeline(
            name="Fit Analysis",
            steps=[
                self._setup_fit_analysis,
                self._add_original_vs_fit,
                self._add_residual_analysis
            ],
            figure_size=(18, 6)
        )
        
        # RDO comparison pipeline
        self.pipelines[VisualizationType.RDO_COMPARISON] = VisualizationPipeline(
            name="RDO Analysis",
            steps=[
                self._setup_rdo_analysis,
                self._add_rd_scatter,
                self._add_cost_comparison,
                self._add_decision_table
            ],
            figure_size=(15, 10)
        )

    def create_visualization(self, viz_type: VisualizationType, **data) -> Figure:
        """Execute a standard pipeline and return figure"""
        pipeline = self.pipelines[viz_type]
        
        # Start fresh
        viz = self.visualizer
        
        # Execute pipeline steps
        for step in pipeline.steps:
            viz = step(self, viz, data)
        
        # Set overall title and build
        if hasattr(viz, 'fig') and viz.fig:
            viz.fig.suptitle(pipeline.title_template.format(**data), fontsize=16)
        
        return viz.build()

    # Pipeline step implementations
    def _setup_block_overview(self, viz: Visualizer, data: dict) -> Visualizer:
        """Setup for block overview"""
        return viz.setup_custom_grid(1, 3, figsize=(18, 6))
    
    def _add_original_block(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add original block visualization"""
        vertices = data['vertices']
        attributes = data['attributes']
        
        ax = viz.fig.add_subplot(1, 3, 1, projection='3d')
        viz.ax = ax  # Set current axis
        
        # TODO: Implement proper 3D scatter with RGB colors
        viz.with_data(vertices, attributes).plot_colored_block()
        ax.set_title("Original Block")
        return viz
    
    def _add_y_channel_view(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add Y channel visualization"""
        vertices = data['vertices']
        attributes = data['attributes']
        
        ax = viz.fig.add_subplot(1, 3, 2, projection='3d')
        viz.ax = ax
        
        viz.with_data(vertices, attributes).plot_point_cloud(color_channel=0)
        ax.set_title("Y Channel")
        return viz
    
    def _add_attribute_stats(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add attribute statistics"""
        attributes = data['attributes']
        
        ax = viz.fig.add_subplot(1, 3, 3)
        
        # TODO: Implement attribute histogram/statistics
        for i, channel in enumerate(['Y', 'U', 'V']):
            if i < attributes.shape[1]:
                ax.hist(attributes[:, i], bins=30, alpha=0.7, label=channel)
        
        ax.set_title("Attribute Distributions")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        return viz

    def _setup_coeff_analysis(self, viz: Visualizer, data: dict) -> Visualizer:
        """Setup for coefficient analysis"""
        return viz.setup_custom_grid(2, 3, figsize=(20, 12))
    
    def _add_energy_compaction(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add energy compaction plots"""
        coeffs = data['coeffs']
        
        # TODO: Implement energy compaction for each channel
        for i, channel in enumerate(['Y', 'U', 'V']):
            ax = viz.fig.add_subplot(2, 3, i+1)
            if i < coeffs.shape[1]:
                channel_coeffs = coeffs[:, i]
                sorted_energy = np.sort(channel_coeffs**2)[::-1]
                cumulative_energy = np.cumsum(sorted_energy) / np.sum(sorted_energy)
                
                ax.plot(range(1, min(21, len(cumulative_energy)+1)), 
                       cumulative_energy[:20], 'o-', linewidth=2)
                ax.axhline(0.9, color='red', linestyle='--', label='90% Energy')
                ax.set_title(f'{channel} Channel Energy Compaction')
                ax.set_xlabel('Top-k Coefficients')
                ax.set_ylabel('Cumulative Energy Fraction')
                ax.legend()
                ax.grid(True, alpha=0.3)
        return viz
    
    def _add_gft_matrix(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add GFT matrix visualization"""
        gft_matrix = data['gft_matrix']
        
        ax = viz.fig.add_subplot(2, 3, (4, 5))  # Span two columns
        viz.ax = ax
        
        im = ax.imshow(gft_matrix, cmap='viridis', aspect='auto')
        viz.fig.colorbar(im, ax=ax)
        ax.set_title('Graph Fourier Transform Matrix')
        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Vertex Index')
        
        return viz
    
    def _add_graph_structure(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add graph structure visualization"""
        vertices = data['vertices']
        graph = data.get('graph')
        
        ax = viz.fig.add_subplot(2, 3, 6, projection='3d')
        viz.ax = ax
        
        viz.with_data(vertices, data['attributes']).with_graph(graph)
        viz.plot_point_cloud(size=20)
        if graph:
            viz.plot_graph_edges()
        ax.set_title("Graph Structure")
        
        return viz

    def _setup_fit_analysis(self, viz: Visualizer, data: dict) -> Visualizer:
        """Setup for fit analysis"""
        return viz.setup_custom_grid(1, 3, figsize=(18, 6))
    
    def _add_original_vs_fit(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add original vs fit comparison"""
        vertices = data['vertices']
        attributes = data['attributes']
        fit_result = data['fit_result']
        
        # Original
        ax1 = viz.fig.add_subplot(1, 3, 1, projection='3d')
        viz.ax = ax1
        viz.with_data(vertices, attributes).plot_point_cloud(color_channel=0)
        ax1.set_title("Original Y Channel")
        
        # Fit approximation
        ax2 = viz.fig.add_subplot(1, 3, 2, projection='3d')
        viz.ax = ax2
        
        # TODO: Implement fit approximation calculation
        # fit_approx = compute_fit_approximation(vertices, fit_result)
        # viz.plot_point_cloud_with_values(fit_approx)
        ax2.set_title(f"Fit Approximation (RMSE: {fit_result.rmse:.4f})")
        
        return viz
    
    def _add_residual_analysis(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add residual analysis"""
        attributes = data['attributes']
        fit_result = data['fit_result']
        
        ax = viz.fig.add_subplot(1, 3, 3)
        
        # TODO: Implement residual calculation
        # residual = attributes[:, 0] - fit_approximation
        # ax.hist(residual, bins=30, alpha=0.7)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")
        
        return viz

    def _setup_rdo_analysis(self, viz: Visualizer, data: dict) -> Visualizer:
        """Setup for RDO analysis"""
        return viz.setup_custom_grid(2, 2, figsize=(15, 10))
    
    def _add_rd_scatter(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add rate-distortion scatter plot"""
        rates = data['rates']
        distortions = data['distortions']
        selected_idx = data.get('selected_idx', 0)
        
        ax = viz.fig.add_subplot(2, 2, 1)
        viz.ax = ax
        
        viz.plot_rate_distortion(rates, distortions, selected_idx, 
                                labels=[f'G{i}' for i in range(len(rates))])
        ax.set_title('Rate-Distortion Trade-off')
        
        return viz
    
    def _add_cost_comparison(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add cost comparison bar plot"""
        rates = data['rates']
        distortions = data['distortions']
        q_step = data.get('q_step', 1)
        selected_idx = data.get('selected_idx', 0)
        
        ax = viz.fig.add_subplot(2, 2, 2)
        
        # TODO: Implement proper RD cost calculation
        costs = [r + q_step * d for r, d in zip(rates, distortions)]
        colors = ['red' if i == selected_idx else 'blue' for i in range(len(costs))]
        
        bars = ax.bar(range(len(costs)), costs, color=colors)
        ax.set_title(f'RD Cost (λ={q_step})')
        ax.set_xlabel('Graph Index')
        ax.set_ylabel('Cost')
        ax.set_xticks(range(len(costs)))
        ax.set_xticklabels([f'G{i}' for i in range(len(costs))])
        
        return viz
    
    def _add_decision_table(self, viz: Visualizer, data: dict) -> Visualizer:
        """Add decision summary table"""
        rates = data['rates']
        distortions = data['distortions']
        selected_idx = data.get('selected_idx', 0)
        
        ax = viz.fig.add_subplot(2, 2, (3, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # TODO: Create decision summary table
        table_data = []
        for i in range(len(rates)):
            selected_mark = "✓" if i == selected_idx else ""
            table_data.append([f'G{i}', f'{rates[i]:.3f}', f'{distortions[i]:.3f}', selected_mark])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Graph', 'Rate', 'Distortion', 'Selected'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title('RDO Decision Summary')
        
        return viz

# TODO: Add these helper functions
def compute_fit_approximation(vertices: np.ndarray, fit_result) -> np.ndarray:
    """
    TODO: Implement fit approximation calculation
    - Apply spatial normalization to vertices
    - Compute Y approximation using fit coefficients
    """
    rotated_vertices = Approximator()._spatial_norm(vertices)
    fit_result
    pass

def compute_residual(original: np.ndarray, approximation: np.ndarray) -> np.ndarray:
    """
    TODO: Implement residual calculation
    """
    pass

def compute_energy_compaction(coeffs: np.ndarray, k: int = 10) -> float:
    """
    TODO: Implement energy compaction metric
    - Calculate energy in top-k coefficients vs total energy
    """
    pass
