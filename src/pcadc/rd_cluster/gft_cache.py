from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class GFTCacheStrategy(ABC):
    
    @abstractmethod
    def store_coeffs(self, block_id: int, coeffs: np.ndarray):
        pass
    
    @abstractmethod
    def get_coeffs(self, block_id: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def exists(self, block_id: int) -> bool:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass


class InMemoryCacheStrategy(GFTCacheStrategy):
    
    def __init__(self):
        self.coeffs_cache: Dict[int, np.ndarray] = {}
    
    def store_coeffs(self, block_id: int, coeffs: np.ndarray):
        # TODO: Store in dictionary
        pass
    
    def get_coeffs(self, block_id: int) -> np.ndarray:
        # TODO: Retrieve from dictionary
        pass
    
    def exists(self, block_id: int) -> bool:
        # TODO: Check if exists
        pass
    
    def cleanup(self):
        # TODO: Clear cache
        pass
