import numpy as np
from typing import Optional
from pcadc.rd_cluster.states import RDClusterState, ClusteringHistory
from pcadc.parameters import SequentialParameters, ClusteringParameters
from pcadc.decider import Decider

class ConvergenceChecker:
    
    def __init__(self, sequential_parameters: SequentialParameters, clustering_parameters: ClusteringParameters, decider: Decider):
        self.sequential_parameters = sequential_parameters
        self.decider = decider
        self.max_iterations = clustering_parameters.max_iterations
        self.rd_cost_threshold = clustering_parameters.rd_cost_threshold
        self.iteration_window = 3 # Overridden to 3 as per requirement
        self.min_iterations = clustering_parameters.min_iterations
        self.assignment_threshold = 0.005 # 0.5%
    
    def should_stop(self, clustering_history: ClusteringHistory) -> bool:
        states = clustering_history.states
        if not states:
            return False
            
        current_state = states[-1]
        
        # 1. Mandatory Max Iterations stop
        if current_state.iteration >= self.max_iterations:
            return True

        # 2. Min iterations safeguard
        if current_state.iteration < self.min_iterations:
            return False

        # 3. Sliding Window Convergence (Last 3 iterations)
        if len(states) >= self.iteration_window:
            window = states[-self.iteration_window:]
            costs = [s.total_cost for s in window if s.total_cost is not None]
            
            if len(costs) == self.iteration_window:
                # Cost Convergence: Max relative difference < 1e-4
                max_cost = max(costs)
                min_cost = min(costs)
                rel_diff = (max_cost - min_cost) / (abs(min_cost) + 1e-10)
                
                if rel_diff < 1e-4:
                    print(f"[*] Stopping: Cost converged (rel_diff={rel_diff:.2e} < 1e-4)")
                    return True
            
            # Assignment Convergence: Avg label changes < 0.5% of total blocks
            # We need the hamming distances. We can calculate them from states.
            # ClusteringHistory doesn't store them directly but they are in the states if we were to add them, 
            # or we calculate them here.
            num_blocks = len(current_state.labels)
            changes = []
            for i in range(len(window) - 1):
                diff = np.sum(window[i].labels != window[i+1].labels)
                changes.append(diff)
            
            if changes:
                avg_changes = sum(changes) / len(changes)
                change_ratio = avg_changes / num_blocks
                if change_ratio < self.assignment_threshold:
                    print(f"[*] Stopping: Assignment converged (change_ratio={change_ratio:.2%} < 0.5%)")
                    return True

        return False
    
    def should_increase_lambda(self, clustering_history: ClusteringHistory) -> bool:
        # Keeping this for potential two-stage lambda schedule, though not strictly requested for the toggle
        return clustering_history.get_cost_stable(self.iteration_window, self.rd_cost_threshold)

