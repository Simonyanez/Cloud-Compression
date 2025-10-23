import numpy as np
from typing import List


class TrainingSetSelector:
    
    def __init__(self, selection_ratio: float = 0.1):
        self.selection_ratio = selection_ratio
    
    def select(self, blocks: List, attributes: np.ndarray) -> List[int]:
        """Returns: List of selected block indices"""
        M = len(blocks)
        n_train = int(M * self.selection_ratio)
        
        # TODO: Extract features from blocks (mean Y, std Y, etc.)
        # TODO: Stratify using k-means on features
        # TODO: Sample uniformly from each stratum
        
        # Placeholder: random
        return np.random.choice(M, size=n_train, replace=False).tolist()

