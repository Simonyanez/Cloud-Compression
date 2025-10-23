from typing import Optional


class ConvergenceChecker:
    
    def __init__(self, max_iterations: int = 100,
                 label_change_threshold: float = 0.01,
                 min_iterations: int = 5):
        self.max_iterations = max_iterations
        self.label_change_threshold = label_change_threshold
        self.min_iterations = min_iterations
    
    def should_stop(self, current_state, prev_state: Optional) -> bool:
        # TODO: Check max iterations
        # TODO: Check minimum iterations
        # TODO: Check label stability
        pass
    
    def should_increase_lambda(self, current_state, 
                              iterations_since_lambda_change: int) -> bool:
        # TODO: Implement lambda scheduling
        pass
