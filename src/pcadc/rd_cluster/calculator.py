from typing import Protocol
from states import RDBlockCost
from gft_cache import GFTCacheStrategy
from pcadc.decider import Decider
from pcadc.blocks import Block
import numpy as np


class RDCostCalculator(Protocol):
    """Protocol for RD cost calculators (assumes implementation exists)"""
    
    def compute_cost(self, block_id: int, cluster_slope: np.ndarray, 
                     lambda_value: float, **kwargs) -> RDBlockCost:
        """
        Compute RD cost for assigning block to cluster with given slope
        
        Assumes this method is already implemented elsewhere in your codebase.
        This is just the interface for the clustering algorithm.
        """
        ...


class RDCostAdapter:
    """Adapter to use your existing RD cost implementation"""
    
    #NOTE: Remember to change decider init parameters (lambda value: qstep)
    def __init__(self, gft_cache: GFTCacheStrategy, decider: Decider):
        """
        TODO: Initialize with your existing RD calculation components
        
        Args:
            gft_cache: Cache to retrieve structural GFT data
            existing_rd_calculator: Your existing RD cost implementation
        """
        self.gft_cache = gft_cache
        self.rd_calculator = decider._RDcost
    
    def compute_cost(self, block: Block, cluster_slope: np.ndarray,
                     lambda_value: float, block_vertices: np.ndarray,
                     block_attributes: np.ndarray) -> RDBlockCost:
        """
        TODO: Adapt your existing RD cost calculation to this interface
        
        Steps:
        1. Retrieve structural coefficients from cache
        2. Call your existing RD cost method with:
           - coefficients
           - cluster slope (for attribute prediction)
           - lambda value
           - any other needed parameters
        3. Package result as RDCost object
        """
        coeffs = self.gft_cache.get_coeffs(block.block_id)
        cost, rate, distortion = self.rd_calculator(coeffs)
        return RDBlockCost(cluster_id=-1, rate=rate, distortion=distortion, cost=cost)
