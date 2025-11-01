from abc import ABC, abstractmethod
from typing import Dict
from src.pcadc.graph import GraphMetadata
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

    def store_coeffs(self, graph_metadata: GraphMetadata, coeffs: np.ndarray):
        graph_id = graph_metadata.graph_cluster_descriptor
        self.coeffs_cache[graph_id] = coeffs
        logger.debug(f"Stored coeffs for {graph_id} in memory cache")

    def get_coeffs(self, graph_metadata: GraphMetadata) -> np.ndarray | None:
        graph_id = graph_metadata.graph_cluster_descriptor
        coeffs = self.coeffs_cache.get(graph_id, None)
        if coeffs is not None:
            return coeffs
        logger.info(f"No cached coefficients for {graph_id}")
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
class HDF5CacheStrategy(GFTCacheStrategy):
    def __init__(self, hdf5_path: str = "cache/gft_coeffs.h5"):
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        self.hdf5_path = hdf5_path

    def _get_key(self, graph_metadata: GraphMetadata) -> str:
        """Generate a unique dataset name based on graph metadata."""
        return str(graph_metadata.graph_cluster_descriptor)

    def store_coeffs(self, graph_metadata: GraphMetadata, coeffs: np.ndarray):
        key = self._get_key(graph_metadata)
        with h5py.File(self.hdf5_path, "a") as f:
            if key in f:
                del f[key]  # Overwrite if already exists
            f.create_dataset(key, data=coeffs, compression="gzip")
        logger.debug(f"Stored coeffs for {key} in HDF5 cache")

    def get_coeffs(self, graph_metadata: GraphMetadata) -> np.ndarray | None:
        key = self._get_key(graph_metadata)
        with h5py.File(self.hdf5_path, "r") as f:
            if key in f:
                logger.debug(f"Loaded coeffs for {key} from HDF5 cache")
                return f[key][:]
        logger.info(f"No cached coefficients for {key} in HDF5")
        return None

    def exists(self, graph_metadata: GraphMetadata) -> bool:
        key = self._get_key(graph_metadata)
        with h5py.File(self.hdf5_path, "r") as f:
            return key in f

    def cleanup(self):
        if os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)
            logger.debug("HDF5 cache file removed")

if __name__ == "__main__":
    import numpy as np
    import tempfile
    from pathlib import Path

    # --- Create realistic GraphMetadata ---
    metadata = GraphMetadata(
        block_id="block_01",
        graph_type="knn",
        distance_threshold=0.15,
        luminance_centroid=np.array([0.3, 0.5, 0.7]),
        centroid_label=2,
        self_loop_threshold=0.1,
        self_loop_weight=0.5,
    )

    coeffs = np.random.randn(4, 4)

    print("=== Testing InMemoryCacheStrategy ===")
    mem_cache = InMemoryCacheStrategy()
    mem_cache.store_coeffs(metadata, coeffs)

    assert mem_cache.exists(metadata)
    loaded = mem_cache.get_coeffs(metadata)
    np.testing.assert_allclose(coeffs, loaded)
    mem_cache.cleanup()
    assert not mem_cache.exists(metadata)
    print("✅ InMemoryCacheStrategy works")

    print("\n=== Testing HDF5CacheStrategy ===")
    tmp_h5 = Path(tempfile.mkdtemp()) / "cache_test.h5"
    h5_cache = HDF5CacheStrategy(hdf5_path=str(tmp_h5))
    h5_cache.store_coeffs(metadata, coeffs)

    assert h5_cache.exists(metadata)
    loaded_h5 = h5_cache.get_coeffs(metadata)
    np.testing.assert_allclose(coeffs, loaded_h5)
    h5_cache.cleanup()
    assert not tmp_h5.exists()
    print("✅ HDF5CacheStrategy works")

    print("\nAll cache strategy tests passed successfully.")




