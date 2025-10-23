import numpy as np
from typing import List


class SlopeOptimizer:
    
    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
    
    def recalculate_slopes(self, blocks: List, labels: np.ndarray,
                          vertices: np.ndarray, attributes: np.ndarray,
                          num_clusters: int) -> np.ndarray:
        """Returns: (K, 3) array of slopes"""
        new_slopes = np.zeros((num_clusters, 3))
        
        for k in range(num_clusters):
            # TODO: Get blocks assigned to cluster k
            # TODO: Stack their vertices V and luminance Y
            # TODO: Solve least squares: slope = (V^T V)^-1 V^T Y
            # TODO: Store in new_slopes[k]
            pass
        
        return new_slopes
