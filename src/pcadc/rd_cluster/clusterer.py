from typing import List, Optional, Tuple
from pcadc.blocks import Block
from pcadc.transforms import GFTStrategyWraper
from pcadc.graph import StructuralGraph, AttributeGraph
from pcadc.color import Approximator, FitCollection
from pcadc.clusterer import YFitClusterer
from pcadc.parameters import SequentialParameters, ClusteringParameters
from pcadc.rd_cluster.convergence import ConvergenceChecker
from pcadc.rd_cluster.optimizer import SlopeOptimizer
from pcadc.rd_cluster.training_set import TrainingSetSelector
from pcadc.rd_cluster.gft_cache import InMemoryCacheStrategy
from pcadc.rd_cluster.states import *
from pcadc.rd_cluster.annealing import SimulatedAnnealing
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import math # For log2 in entropy calculation


class RDClusterer:
    
    def __init__(self,
                 sequential_parameters:SequentialParameters,
                 clusterer_parameters:ClusteringParameters,
                 decider: Decider,
                 gft_cache: InMemoryCacheStrategy,
                 gft_computer: GFTStrategyWraper,
                 slope_optimizer: SlopeOptimizer,
                 convergence_checker: ConvergenceChecker,
                 temp_folder: Path, # Add temp_folder
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
        self.temp_folder = temp_folder # Store temp_folder

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
        # Create temp directory
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        
        self._precompute_structural_gfts(blocks, vertices,attributes)

        state = self._initialize_state(blocks, vertices, attributes)
        print(f"Initial state {state}")
        history = ClusteringHistory([state])
        self._save_intermediate_state(state) # Save initial state

        for iteration in tqdm(range(self.convergence_checker.max_iterations), "Running RD Clustering"):
            # _assignment_step will now return more data
            new_labels, total_cost, all_rates, all_distortions, all_gains = self._assignment_step(blocks, state, vertices, attributes, iteration)
            
            # Inside your main clustering loop
            new_slopes, new_slw, new_slp = self.slope_optimizer.recalculate_slopes(
                blocks=blocks,
                labels=new_labels,
                vertices=vertices,
                attributes=attributes,
                num_clusters=self.num_clusters,
                old_slopes=state.slopes,
                old_slw=state.self_loop_weights,
                old_slp=state.self_loop_percentages,
                rd_cost_fn=self._evaluate_rd_cost_for_optimizer  # Pass the callback
            )

            # Calculate new metrics
            cluster_entropy = self._calculate_cluster_entropy(new_labels)
            avg_rate = np.mean(all_rates)
            avg_distortion = np.mean(all_distortions)
            cluster_gains = self._calculate_cluster_gains(new_labels, all_gains)

            state = RDClusterState(labels=new_labels,
                                   slopes=new_slopes,
                                   self_loop_weights=new_slw,
                                   self_loop_percentages=new_slp,
                                   qstep_value=self.qstep_schedule[self.lambda_step],
                                   lambda_step=self.lambda_step,
                                   iteration=iteration,
                                   total_cost = total_cost,
                                   cluster_entropy=cluster_entropy,
                                   avg_rate=avg_rate,
                                   avg_distortion=avg_distortion,
                                   cluster_gains=cluster_gains,
                                   all_rates=all_rates,
                                   all_distortions=all_distortions,
                                   all_gains=all_gains)

            print(f"Current state: \n {state}")
            history.add_state(state)
            self._save_intermediate_state(state) # Save state

            # Cool down the temperature
            self.annealing_scheduler.cool_down()
            print(f"Temperature updated to {self.annealing_scheduler.temperature:.4f}")

            if self.convergence_checker.should_stop(history):
                return state, history

            if self.convergence_checker.should_increase_lambda(history):
                self.lambda_step += 1

        return state, history

    def _evaluate_rd_cost_for_optimizer(self, block: Block, params: np.ndarray, cluster_idx: int) -> float:
        """
        Callback function passed to SlopeOptimizer to evaluate candidate slopes and self-loop params.
        params: [slope_x, slope_y, slope_z, slw, slp]
        """
        slope = params[:3]
        slw = params[3]
        slp = params[4]
        # Calculate coefficients using the candidate parameters
        coeffs = self._compute_adaptive_gft(block, slope, cluster_idx, slp, slw)
        
        # Calculate RD cost
        rd_cost, _, _ = self.decider._RDcost(coeffs)
        
        return rd_cost

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
        clusterer = YFitClusterer(
            self.num_clusters, self.sequential_parameters.normalize_slopes)
        codebook = clusterer(fit_collection)
        codebook.assign(blocks, vertices, attributes)


        # Return RDClusterState with initialization values
        return RDClusterState(labels=codebook.labels,
                       slopes=codebook.centroids,
                       self_loop_weights=np.full(self.num_clusters, self.sequential_parameters.self_loop_weight),
                       self_loop_percentages=np.full(self.num_clusters, self.sequential_parameters.self_loop_percentage),
                       qstep_value=self.qstep_schedule[self.lambda_step],
                        lambda_step=self.lambda_step,
                       iteration=0)


    def _assignment_step(self, blocks: List[Block], state: RDClusterState, 
                         vertices: np.ndarray, attributes: np.ndarray, iteration: int):
        new_labels = np.zeros(len(blocks), dtype=int)
        slopes = state.slopes
        total_cost = 0
        self.decider._set_vars(state.qstep_value)

        all_rates = np.zeros(len(blocks))
        all_distortions = np.zeros(len(blocks))
        all_gains = np.zeros(len(blocks))

        max_exploration_iters = 10
        if iteration < max_exploration_iters:
            # Linearly decays from 1.4 down to 1.0
            margin = 1.4 - (0.4 * (iteration / max_exploration_iters))
        else:
            margin = 1.0

        for i, block in tqdm(enumerate(blocks), "Assigning blocks"):
            block.init_data(vertices, attributes)
            costs = []
            rates = []
            distortions = []

            for k, slope in enumerate(slopes):
                if k == 0:
                    coeffs = self.gft_cache.get_coeffs(block.block_id)
                else:
                    slw = state.self_loop_weights[k]
                    slp = state.self_loop_percentages[k]
                    coeffs = self._compute_adaptive_gft(block, slope, k, slp, slw)
                
                rd_cost, sparsity, qerror = self.decider._RDcost(coeffs)
                costs.append(rd_cost)
                rates.append(sparsity)
                distortions.append(qerror)

            # --- NEW DETERMINISTIC SELECTION LOGIC ---
            cost_structural = costs[0]
            dynamic_costs = np.array(costs[1:])
            best_dynamic_idx = np.argmin(dynamic_costs) + 1 if len(dynamic_costs) > 0 else 0 # +1 to offset slicing
            cost_best_dynamic = costs[best_dynamic_idx]

            # If the best dynamic cluster is within our allowed margin of safety 
            # compared to the flat transform, force the block into it.
            if cost_best_dynamic < (cost_structural * margin):
                chosen_cluster = best_dynamic_idx
            else:
                chosen_cluster = 0 # Fallback to structural

            new_labels[i] = chosen_cluster
            total_cost += costs[chosen_cluster]
            all_rates[i] = rates[chosen_cluster]
            all_distortions[i] = distortions[chosen_cluster]
            
            if chosen_cluster != 0:
                all_gains[i] = cost_structural - costs[chosen_cluster]
            else:
                all_gains[i] = 0

            block.clear_data()
        return new_labels, total_cost, all_rates, all_distortions, all_gains    

    def _calculate_cluster_entropy(self, labels: np.ndarray) -> float:
        """Calculates the entropy of the cluster distribution."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        # Add a small epsilon to avoid log(0) if a probability is exactly zero
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def _calculate_cluster_gains(self, labels: np.ndarray, all_gains: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculates detailed gain statistics for each dynamic cluster."""
        cluster_gains_stats = {}
        for k in range(1, self.num_clusters):  # Iterate through dynamic clusters
            cluster_mask = (labels == k)
            if np.any(cluster_mask):
                cluster_specific_gains = all_gains[cluster_mask]
                
                # Basic stats
                avg_gain = np.mean(cluster_specific_gains)
                min_gain = np.min(cluster_specific_gains)
                max_gain = np.max(cluster_specific_gains)
                
                # Count blocks by gain type
                positive_gain_blocks = np.sum(cluster_specific_gains > 0)
                zero_gain_blocks = np.sum(cluster_specific_gains == 0)
                negative_gain_blocks = np.sum(cluster_specific_gains < 0)
                
                cluster_gains_stats[k] = {
                    "avg_gain": avg_gain,
                    "min_gain": min_gain,
                    "max_gain": max_gain,
                    "positive_gain_blocks": positive_gain_blocks,
                    "zero_gain_blocks": zero_gain_blocks,
                    "negative_gain_blocks": negative_gain_blocks,
                }
            else:
                # Cluster is empty
                cluster_gains_stats[k] = {
                    "avg_gain": 0.0,
                    "min_gain": 0.0,
                    "max_gain": 0.0,
                    "positive_gain_blocks": 0,
                    "zero_gain_blocks": 0,
                    "negative_gain_blocks": 0,
                }
        return cluster_gains_stats

    def _save_intermediate_state(self, state: RDClusterState):
        """Saves the current RDClusterState to a JSON file."""
        filepath = self.temp_folder / f"iteration_{state.iteration:03d}.json"
        with open(filepath, 'w') as f:
            json.dump(state.to_dict(), f, indent=4)

    def _compute_adaptive_gft(self, block: Block, slope: np.ndarray, label: int, slp: Optional[float] = None, slw: Optional[float] = None):
        if slp is None:
            slp = self.sequential_parameters.self_loop_percentage
        if slw is None:
            slw = self.sequential_parameters.self_loop_weight

        structural_graph = StructuralGraph(block.metadata)
        structural_graph.set_data(block.Vblock)

        attribute_graph = AttributeGraph(structural_graph, slope, label, slp, slw)

        Vblock_rotated = Approximator()._spatial_norm(block.Vblock)
        Ablock_app = block.Ablock.copy()
        Ablock_app[:, 0] = Vblock_rotated @ slope.T
        attribute_graph.set_data(block.Vblock, Ablock_app)

        _, coeffs = self.gft_computer(block, attribute_graph)
        attribute_graph.clear_data()
        return coeffs
