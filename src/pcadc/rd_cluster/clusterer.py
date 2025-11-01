from typing import List, Optional
from src.pcadc.blocks import Block
from src.pcadc.transforms import GFTStrategyWraper
from src.pcadc.graph import StructuralGraph, AttributeGraph
from src.pcadc.color import Approximator, FitCollection
from convergence import ConvergenceChecker
from optimizer import SlopeOptimizer
from training_set import TrainingSetSelector
from gft_cache import InMemoryCacheStrategy
from states import *
from tqdm import tqdm
import numpy as np



class RDClusterer:
    
    def __init__(self,
                 num_clusters: int,
                 lambda_schedule: List[float],
                 gft_cache: InMemoryCacheStrategy,
                 gft_computer: GFTStrategyWraper,
                 slope_optimizer: SlopeOptimizer,
                 convergence_checker: ConvergenceChecker,
                 decider: Decider,
                 sequential_parameters: SequentialParameters,
                 training_selector: Optional[TrainingSetSelector] = None,
                 use_two_stage: bool = True):
        
        self.num_clusters = num_clusters
        self.lambda_schedule = lambda_schedule
        self.lambda_step = 0
        self.gft_cache = gft_cache
        self.gft_computer = gft_computer
        self.slope_optimizer = slope_optimizer
        self.convergence_checker = convergence_checker
        self.sequential_parameters = sequential_parameters
        self.training_selector = training_selector
        self.decider = decider
        self.use_two_stage = use_two_stage
    
    def fit(self, blocks: List, vertices: np.ndarray, attributes: np.ndarray):
        if self.use_two_stage:
            return self._fit_two_stage(blocks, vertices, attributes)
        else:
            return self._fit_full(blocks, vertices, attributes)
    
    def _fit_two_stage(self, blocks, vertices, attributes):
        # TODO: Select training blocks
        # TODO: Train on subset (_fit_full on subset)
        # TODO: Precompute GFTs for all blocks
        # TODO: Assign all blocks to trained clusters
        # TODO: Optional: refine slopes with all blocks
        pass
    
    def _fit_full(self, blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray):
        # TODO: Precompute structural GFTs
        self._precompute_structural_gfts(blocks, vertices,attributes)

        # TODO: Initialize state
        state = self._initialize_state(blocks, vertices, attributes)
        history = ClusteringHistory([state])

        # TODO: Main loop:
        #   - Assignment step
        #   - Update step
        #   - Check convergence
        #   - Lambda schedule
        for iteration in range(self.convergence_checker.max_iterations):
            new_labels, total_cost = self._assignment_step(blocks, state, vertices, attributes)
            new_slopes = self.slope_optimizer.recalculate_slopes(blocks, new_labels, vertices, attributes, self.num_clusters)
            if self.convergence_checker.should_stop(history):
                break
            if self.convergence_checker.should_increase_lambda(history):
                self.lambda_step += 1
            state = RDClusterState(labels=new_labels,
                                   slopes=new_slopes,
                                   lambda_value=self.lambda_schedule[self.lambda_step],
                                   lambda_step=self.lambda_step,
                                   iteration=iteration,
                                   total_cost = total_cost)
        return state
    
    def _precompute_structural_gfts(self, blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray):
        for block in tqdm(blocks, "Pre-computing Structural Coefficients"):
            block.init_data(vertices, attributes)
            structural_graph = StructuralGraph(block.metadata)
            structural_graph.set_data(vertices)
            _, coeffs = self.gft_computer(block, structural_graph)
            self.gft_cache.store_coeffs(structural_graph.metadata, coeffs)
            block.clear_data()
    
    def _initialize_state(self, blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray):
        # Initialize labels (random or k-means)
        M = len(blocks) # NOTE: This is M because it could be a subset of the N blocks
        labels = np.random.randint(self.num_clusters, size=M)

        # Initialize slopes
        fit_collection = FitCollection()
        approximator = Approximator()
        for block in tqdm(blocks, "Fitting luminansce by linear approximation"):
            block.init_data(vertices, attributes)
            fit_result = approximator(block)
            fit_collection.add(fit_result)
            block.clear_data()
        slopes = fit_collection.get_slopes()

        # Return RDClusterState with initialization values
        return RDClusterState(labels=labels,
                       slopes=slopes,
                       lambda_value=self.lambda_schedule[self.lambda_step],lambda_step=self.lambda_step,
                       iteration=0)

    def _assignment_step(self, blocks: List[Block], state: RDClusterState, vertices: np.ndarray, attributes: np.ndarray):
        new_labels = np.zeros(len(blocks), dtype=int)
        slopes = state.slopes
        labels = state.labels
        total_cost = 0
        
        for i, block in enumerate(blocks):
            block.init_data(vertices, attributes)
            min_cost = np.inf
            for k, slope in enumerate(slopes):
                coeffs = self._compute_adaptive_gft(block, slope, k)
                # TODO: Decider needs to be previously initialized 
                rd_cost, sparcity, qerror = self.decider._RDcost(coeffs)
                if rd_cost < min_cost:
                    min_cost = rd_cost
                    new_labels[i] = k
            total_cost+=min_cost
            block.clear_data()
        return new_labels, total_cost
    
    def _compute_adaptive_gft(self, block: Block, slope: np.ndarray, label: int):
        # TODO: Get structural graph
        structural_graph = StructuralGraph(block.metadata)
        structural_graph.set_data(block.Vblock)

        # TODO: Add self-loops based on slope
        attribute_graph = AttributeGraph(structural_graph, slope, self.sequential_parameters.self_loop_threshold, label,self.sequential_parameters.self_loop_weight)
        #NOTE: Almost sure using the spatially normed vertices is the right way
        Vblock_rotated = Approximator()._spatial_norm(block.Vblock)
        Ablock_app = block.Ablock.copy()
        Ablock_app[:, 0] = Vblock_rotated @ slope
        attribute_graph.set_data(block.Vblock, Ablock_app)

        # TODO: Compute new Laplacian
        # TODO: Compute GFT (eigenvectors)
        # TODO: Return GFT matrix
        _, coeffs = self.gft_computer(block, attribute_graph)
        attribute_graph.clear_data()
        return coeffs
