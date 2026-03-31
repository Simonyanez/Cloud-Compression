import numpy as np
from typing import List
from pcadc.blocks import Block
from pcadc.color import Approximator


class SlopeOptimizer:
    
    def __init__(self, learning_rate: float, add_intercept: bool = True):
        # Not used. This is for the bias value
        self.add_intercept = add_intercept
        self.learning_rate = learning_rate
    
    def recalculate_slopes(self, blocks: List[Block], labels: np.ndarray,
                          vertices: np.ndarray, attributes: np.ndarray,
                          num_clusters: int, old_slopes: np.ndarray) -> np.ndarray:
        """Returns: (K, 3) array of slopes"""
        new_slopes = old_slopes.copy()
        
        for k in range(num_clusters):
            # Static DC cluster
            if k == 0:
                continue

            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                # If cluster is empty, keep the old slope, do not randomize
                continue
            
            V_list = []
            Y_list = []
            weight_list = []
            
            for idx in cluster_indices:
                block = blocks[idx]
                block.init_data(vertices, attributes)
                
                V_block = Approximator()._spatial_norm(block.Vblock)
                Y_block = block.Ablock[:, 0]
                n_i = len(V_block)
                
                V_list.append(V_block)
                Y_list.append(Y_block)
                weight_list.append(np.ones(n_i) / n_i)  # Peso por vértice
                
                block.clear_data()
            
            V_stacked = np.vstack(V_list)
            Y_stacked = np.concatenate(Y_list)
            weights = np.concatenate(weight_list)  # Vector (N_total,)

            # Weighted least squares
            W_sqrt = np.sqrt(weights)
            V_weighted = V_stacked * W_sqrt[:, np.newaxis]
            Y_weighted = Y_stacked * W_sqrt
            
            VtV = V_weighted.T @ V_weighted
            VtY = V_weighted.T @ Y_weighted
            
            # Calculated average slope
            beta = np.linalg.solve(VtV, VtY)
            
            # Apply learning rate
            new_slopes[k] = (1 - self.learning_rate) * old_slopes[k] + self.learning_rate * beta
        
        return new_slopes
