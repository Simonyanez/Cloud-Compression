from abc import ABC, abstractmethod
from typing import Dict
from pcadc.graph import GraphMetadata
import numpy as np
import h5py
import os
import logging

# Setup logging
logging.basicConfig(
    filename="logs/gft_cache.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===============================
# Abstract Base Class
# ===============================
class GFTCacheStrategy(ABC):
    @abstractmethod
    def store_coeffs(self, graph_metadata: GraphMetadata, coeffs: np.ndarray):
        pass

    @abstractmethod
    def get_coeffs(self, graph_metadata: GraphMetadata) -> np.ndarray | None:
        pass

    @abstractmethod
    def exists(self, graph_metadata: GraphMetadata) -> bool:
        pass

    @abstractmethod
    def cleanup(self):
        pass


# ===============================
# In-Memory Implementation
# ===============================
class InMemoryCacheStrategy(GFTCacheStrategy):
    def __init__(self):
        self.coeffs_cache: Dict[str, np.ndarray] = {}

    def store_coeffs(self, block_id: str, coeffs: np.ndarray):
        self.coeffs_cache[block_id] = coeffs
        logger.debug(f"Stored coeffs for {block_id} in memory cache")

    def get_coeffs(self, block_id: str) -> np.ndarray | None:
        coeffs = self.coeffs_cache.get(block_id, None)
        if coeffs is not None:
            return coeffs
        logger.info(f"No cached coefficients for {block_id}")
        return None

    def exists(self, graph_metadata: GraphMetadata) -> bool:
        graph_id = graph_metadata.graph_cluster_descriptor
        return graph_id in self.coeffs_cache

    def clean_related_keys(self, centroid: np.ndarray):
        self.coeffs_cache = {
            k: v for k, v in self.coeffs_cache.items() if str(centroid) not in k
        }

    def cleanup(self):
        self.coeffs_cache.clear()
        logger.debug("Memory cache cleared")


# ===============================
# HDF5 Implementation
# ===============================
# class HDF5CacheStrategy(GFTCacheStrategy):
#     def __init__(self, hdf5_path: str = "cache/gft_coeffs.h5"):
#         os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
#         self.hdf5_path = hdf5_path
#
#     def _get_key(self, graph_metadata: GraphMetadata) -> str:
#         """Generate a unique dataset name based on graph metadata."""
#         return str(graph_metadata.graph_cluster_descriptor)
#
#     def store_coeffs(self, graph_metadata: GraphMetadata, coeffs: np.ndarray):
#         key = self._get_key(graph_metadata)
#         with h5py.File(self.hdf5_path, "a") as f:
#             if key in f:
#                 del f[key]  # Overwrite if already exists
#             f.create_dataset(key, data=coeffs, compression="gzip")
#         logger.debug(f"Stored coeffs for {key} in HDF5 cache")
#
#     def get_coeffs(self, graph_metadata: GraphMetadata) -> np.ndarray | None:
#         key = self._get_key(graph_metadata)
#         with h5py.File(self.hdf5_path, "r") as f:
#             if key in f:
#                 logger.debug(f"Loaded coeffs for {key} from HDF5 cache")
#                 return f[key][:]
#         logger.info(f"No cached coefficients for {key} in HDF5")
#         return None
#
#     def exists(self, graph_metadata: GraphMetadata) -> bool:
#         key = self._get_key(graph_metadata)
#         with h5py.File(self.hdf5_path, "r") as f:
#             return key in f
#
#     def cleanup(self):
#         if os.path.exists(self.hdf5_path):
#             os.remove(self.hdf5_path)
#             logger.debug("HDF5 cache file removed")

if __name__ == "__main__":
    pass




