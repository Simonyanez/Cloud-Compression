from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from pcadc.parameters import SequentialParameters
from pcadc.decider import Decider


@dataclass
class RDClusterState:
    """Encapsulates the state of RD clustering at any iteration"""
    labels: np.ndarray          # (N_blocks,) cluster assignment per block
    slopes: np.ndarray          # (N_clusters, slope_dim) cluster centroids
    lambda_value: float         # Current lambda for RD tradeoff
    lambda_step: int            # Current position in lambda schedule
    iteration: int              # Iteration counter
    total_cost: Optional[float] = None  # Total RD cost (if tracked)

    def is_last(self, sequential_parameters: SequentialParameters, decider: Decider):
        last_lambda = decider.get_lagrange_mult(sequential_parameters.quantization_steps[-1])
        return self.lambda_value == last_lambda

@dataclass
class RDBlockCost:
    """Rate-distortion cost for a single block-cluster assignment"""
    cluster_id: int
    rate: float
    distortion: float
    cost: float  # rate + lambda * distortion


@dataclass
class ClusteringHistory:
    """Tracks clustering evolution over iterations"""
    states: List[RDClusterState] = field(default_factory=list)
    # convergence_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_state(self, state: RDClusterState):
        """Add a new state to history"""
        self.states.append(state)
    
    def get_cost_stable(self, num_of_iters: int, threshold: float = 1e-4) -> bool:
        """
        Check if cost has been stable for the last num_of_iters iterations
        
        Args:
            num_of_iters: Number of iterations to check
            threshold: Maximum relative change allowed
        
        Returns:
            True if cost changes are below threshold for all checked iterations
        """
        if len(self.states) < num_of_iters + 1:
            return False
        
        recent_states = self.states[-(num_of_iters + 1):]
        costs = [state.total_cost for state in recent_states]
        
        if any(c is None for c in costs):
            return False

        costs = np.array(costs)
        diffs = np.diff(costs)  # [cost[1]-cost[0], cost[2]-cost[1], ...]
        relative_changes = np.abs(diffs) / (np.abs(costs[:-1]) + 1e-10)
        return np.all(relative_changes < threshold)            



