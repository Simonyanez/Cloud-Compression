from typing import List, Optional, Tuple
from src.pcadc.blocks import Block
from src.pcadc.transforms import GFTStrategyWraper
from src.pcadc.graph import StructuralGraph, AttributeGraph
from src.pcadc.color import Approximator, FitCollection
from src.pcadc.clusterer import Clusterer
from src.pcadc.parameters import SequentialParameters, ClusteringParameters
from src.pcadc.rd_cluster.convergence import ConvergenceChecker
from src.pcadc.rd_cluster.optimizer import SlopeOptimizer
from src.pcadc.rd_cluster.training_set import TrainingSetSelector
from src.pcadc.rd_cluster.gft_cache import InMemoryCacheStrategy
from src.pcadc.rd_cluster.states import *
from src.pcadc.rd_cluster.annealing import SimulatedAnnealing
from tqdm import tqdm
import numpy as np



class RDClusterer:
    
    def __init__(self,
                 sequential_parameters:SequentialParameters,
                 clusterer_parameters:ClusteringParameters,
                 decider: Decider,
                 gft_cache: InMemoryCacheStrategy,
                 gft_computer: GFTStrategyWraper,
                 slope_optimizer: SlopeOptimizer,
                 convergence_checker: ConvergenceChecker,
                 training_selector: Optional[TrainingSetSelector] = None,
                 use_two_stage: bool = True):
        
        self.num_clusters = clusterer_parameters.number_of_clusters
        self.qstep_schedule = sequential_parameters.quantization_steps
        self.lambda_step = 0
        self.gft_cache = gft_cache
        self.gft_computer = gft_computer
        self.slope_optimizer = slope_optimizer
        self.convergence_checker = convergence_checker
        self.sequential_parameters = sequential_parameters
        self.training_selector = training_selector
        self.decider = decider
        self.use_two_stage = use_two_stage

        # Simulated Annealing
        self.annealing_scheduler = SimulatedAnnealing(
            initial_temperature=clusterer_parameters.initial_temperature,
            final_temperature=clusterer_parameters.final_temperature,
            cooling_rate=clusterer_parameters.cooling_rate
        )
    
    def fit(self, blocks: List, vertices: np.ndarray, attributes: np.ndarray) -> Tuple[RDClusterState, ClusteringHistory]:
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
    
    def _fit_full(self, blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray) -> Tuple[RDClusterState, ClusteringHistory]:
        self._precompute_structural_gfts(blocks, vertices,attributes)

        state = self._initialize_state(blocks, vertices, attributes)
        print(f"Initial state {state}")
        history = ClusteringHistory([state])

        for iteration in tqdm(range(self.convergence_checker.max_iterations), "Running RD Clustering"):
            new_labels, total_cost = self._assignment_step(blocks, state, vertices, attributes)
            new_slopes = self.slope_optimizer.recalculate_slopes(blocks, new_labels, vertices, attributes, self.num_clusters)
            state = RDClusterState(labels=new_labels,
                                   slopes=new_slopes,
                                   qstep_value=self.qstep_schedule[self.lambda_step],
                                   lambda_step=self.lambda_step,
                                   iteration=iteration,
                                   total_cost = total_cost)

            print(f"Current state: \n {state}")
            history.add_state(state)

            # Cool down the temperature
            self.annealing_scheduler.cool_down()
            print(f"Temperature updated to {self.annealing_scheduler.temperature:.4f}")

            if self.convergence_checker.should_stop(history):
                return state, history

            if self.convergence_checker.should_increase_lambda(history):
                self.lambda_step += 1

        return state, history
    
    def _precompute_structural_gfts(self, blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray):
        for block in tqdm(blocks, "Pre-computing Structural Coefficients"):
            block.init_data(vertices, attributes)
            structural_graph = StructuralGraph(block.metadata)
            structural_graph.set_data(block.Vblock)
            _, coeffs = self.gft_computer(block, structural_graph)
            self.gft_cache.store_coeffs(block.block_id, coeffs)
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
        clusterer = Clusterer(
            self.num_clusters, self.sequential_parameters.normalize_slopes)
        codebook = clusterer(fit_collection)
        codebook.assign(blocks, vertices, attributes)


        # Return RDClusterState with initialization values
        return RDClusterState(labels=codebook.labels,
                       slopes=codebook.centroids,
                       qstep_value=self.qstep_schedule[self.lambda_step],
                        lambda_step=self.lambda_step,
                       iteration=0)

    def _assignment_step(self, blocks: List[Block], state: RDClusterState, vertices: np.ndarray, attributes: np.ndarray):
        new_labels = np.zeros(len(blocks), dtype=int)
        slopes = state.slopes
        total_cost = 0
        self.decider._set_vars(state.qstep_value)

        for i, block in tqdm(enumerate(blocks), "Assigning blocks"):
            block.init_data(vertices, attributes)
            costs = []
            for k, slope in enumerate(slopes):
                coeffs = self.gft_cache.get_coeffs(block.block_id)
                if k != 0:
                    coeffs = self._compute_adaptive_gft(block, slope, k)
                
                rd_cost, _, _ = self.decider._RDcost(coeffs)
                costs.append(rd_cost)
            
            chosen_cluster = self.annealing_scheduler.choose(np.array(costs), len(slopes))
            new_labels[i] = chosen_cluster
            total_cost += costs[chosen_cluster]
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
        Ablock_app[:, 0] = Vblock_rotated @ slope.T
        attribute_graph.set_data(block.Vblock, Ablock_app)

        # TODO: Compute new Laplacian
        # TODO: Compute GFT (eigenvectors)
        # TODO: Return GFT matrix
        _, coeffs = self.gft_computer(block, attribute_graph)
        attribute_graph.clear_data()
        return coeffs
