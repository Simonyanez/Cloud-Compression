from typing import Optional
from src.pcadc.rd_cluster.states import RDClusterState, ClusteringHistory
from src.pcadc.parameters import SequentialParameters, ClusteringParameters
from src.pcadc.decider import Decider

# max_iterations: int = 100,
#                  rd_cost_threshold: float = 1e-4,
#                  iteration_window: int = 10,
#                  min_iterations: int = 5
class ConvergenceChecker:
    
    def __init__(self, sequential_parameters: SequentialParameters, clustering_parameters: ClusteringParameters, decider: Decider):
        self.sequential_parameters = sequential_parameters
        self.decider = decider
        self.max_iterations = clustering_parameters.max_iterations
        self.rd_cost_threshold = clustering_parameters.rd_cost_threshold
        self.iteration_window = clustering_parameters.iteration_window
        self.min_iterations = clustering_parameters.min_iterations
    
    def should_stop(self, clustering_history: ClusteringHistory) -> bool:
        current_state = clustering_history.states[-1]
        if current_state.iteration >= self.max_iterations:
            return True

        if current_state.iteration < self.min_iterations:
            return False

        if current_state.is_last(self.sequential_parameters) and self.should_increase_lambda(clustering_history):
            return True
        return False
    
    def should_increase_lambda(self, clustering_history: ClusteringHistory) -> bool:
        return clustering_history.get_cost_stable(self.iteration_window, self.rd_cost_threshold)

