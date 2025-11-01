import numpy as np
from typing import List
from src.pcadc.blocks import Block


class SlopeOptimizer:
    
    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
    
    def recalculate_slopes(self, blocks: List[Block], labels: np.ndarray,
                          vertices: np.ndarray, attributes: np.ndarray,
                          num_clusters: int) -> np.ndarray:
        """Returns: (K, 3) array of slopes"""
        new_slopes = np.zeros((num_clusters, 3))
        
        for k in range(num_clusters):
            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            V_list = []
            Y_list = []
            weight_list = []
            
            for idx in cluster_indices:
                block = blocks[idx]
                block.init_data(vertices, attributes)
                
                V_block = block.Vblock
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
            
            beta = np.linalg.solve(VtV, VtY)
            new_slopes[k] = beta
        
        return new_slopes
