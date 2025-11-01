from typing import Optional
from states import RDClusterState, ClusteringHistory
from pcadc.parameters import SequentialParameters
from pcadc.decider import Decider

class ConvergenceChecker:
    
    def __init__(self, sequential_parameters: SequentialParameters, decider: Decider, max_iterations: int = 100,
                 rd_cost_threshold: float = 1e-4,
                 iteration_window: int = 10,
                 min_iterations: int = 5):
        self.sequential_parameters = sequential_parameters
        self.decider = decider
        self.max_iterations = max_iterations
        self.rd_cost_threshold = rd_cost_threshold
        self.iteration_window = iteration_window
        self.min_iterations = min_iterations
    
    def should_stop(self, clustering_history: ClusteringHistory) -> bool:
        current_state = clustering_history.states[-1]
        if current_state.iteration >= self.max_iterations:
            return True

        if current_state.iteration < self.min_iterations:
            return False

        if current_state.is_last(self.sequential_parameters, self.decider) and self.should_increase_lambda(clustering_history):
            return True
        return False
    
    def should_increase_lambda(self, clustering_history: ClusteringHistory) -> bool:
        return clustering_history.get_cost_stable(self.iteration_window, self.rd_cost_threshold)

