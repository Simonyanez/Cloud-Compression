from dataclasses import dataclass
from .io import *
from pathlib import Path
from .graph import *
from .blocks import *
from typing import Optional, List, Dict, Protocol
from .color import *
import numpy as np
import logging
import os

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "pointcloud.log")

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)


@dataclass
class PointCloudMetadata:
    dataset: str
    sequence: str
    depth: int
    frame: int

    @property
    def source_id(self) -> str:
        return f"{self.dataset}_{self.sequence}_v{self.depth}"

    @property
    def descriptor(self) -> str:
        return f"{self.dataset}_{self.sequence}_v{self.depth}_{self.frame}"

    def get_frame(self) -> int:
        return self.frame

    def __str__(self) -> str:
        return (
            f"Point Cloud Metadata\n"
            f"==============================\n"
            f"Dataset: {self.dataset}\n"
            f"Sequence: {self.sequence}\n"
            f"Depth: {self.depth}\n"
            f"Frame: {self.frame}\n"
        )


class PointCloud:
    def __init__(self, metadata: PointCloudMetadata):
        self.metadata = metadata
        logger.info(f"Point Cloud initialized {metadata}")
        self.V: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None  # Generic attributes


    @property
    def vertices(self) -> np.ndarray:
        assert self.V is not None, "Vertices haven't been initialized"
        return self.V

    @property
    def attributes(self) -> np.ndarray:
        assert self.A is not None, "Attributes haven't been initialized"
        return self.A

    @property
    def source_id(self) -> str:
        return self.metadata.source_id

    @property
    def frame(self) -> int:
        return self.metadata.frame

    @property
    def descriptor(self) -> str:
        return self.metadata.descriptor

    @classmethod
    def from_file(cls, path: Path, fmt: str, metadata: PointCloudMetadata):
        V, attrs = PointCloudIO.load(path, fmt)
        obj = cls(metadata)
        obj.V = V
        obj.A = attrs
        return obj

    def transform_attributes(self, fn):
        """Apply a function to attributes (e.g., RGB → YUV, intensity normalization, etc.)"""
        logger.info(f"Point Cloud attributes transformed by function")
        if self.A is not None:
            self.A = fn(self.A)


class BlockPartitionStrategy(Protocol):
    def partition(self, pc: PointCloud, **kwargs) -> List[Block]:
        ...


class MortonBlockPartition(BlockPartitionStrategy):
    def partition(self, pc: PointCloud, bsize: int) -> Tuple[List[Tuple[int,int]],List[Block]]:
        assert pc.V is not None, "Vertices not initialized"
        base_bsize = np.log2(bsize)
        assert np.floor(
            base_bsize) == base_bsize, "Block size must be a power of 2"
        logger.info(f"Partitioning pointcloud {pc.descriptor} by blocks of size {bsize}")

        V_coarse = np.floor(pc.V / bsize) * bsize
        variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)
        variation = np.concatenate(([1], variation))

        start_indexes = np.nonzero(variation)[0]
        Nlevel = pc.V.shape[0]
        end_indexes = np.concatenate((start_indexes[1:] - 1, [Nlevel - 1]))
        indexes = list(zip(start_indexes, end_indexes))
        logger.info(f"Pointcloud partitioned into {len(indexes)} blocks")

        return indexes, [Block(BlockMetadata(
                start=idx[0],
                end=idx[1],
                source=pc.source_id,
                frame=pc.frame,
                block_size=bsize,
                block_idx=i
            ))
            for i, idx in enumerate(indexes)
        ]
    

class Sampler:
    # NOTE: For now this is just doing stratified sampling
    def __init__(self, ratio: float, n_strata: int):
        self.ratio = ratio
        self.n_strata = n_strata

    def __call__(self,V: np.ndarray, A:np.ndarray, blocks: List[Block]) -> List[Block]:
        Yvariance = np.array([self._get_luminansce_variance(V, A, block) for block in blocks])
        bins = self._build_bins(Yvariance)

        pass
        
    def _get_luminansce_variance(self,V: np.ndarray, A:np.ndarray, block: Block):
        block.init_data(V, A)
        Yblock = block.Ablock[:,0]
        Yvar = np.var(Yblock) if Yblock.shape[0] > 1 else 0.0
        return Yvar

    def _build_bins(self, variances: np.ndarray):
        bins = np.percentile(variances, np.linspace(0, 100, self.n_strata + 1))
        bins = np.unique(bins)
        return bins

        
   #  1 def _stratified_subsample_by_luminance(self, blocks: List[Block], ratio: float, n_strata: int =
   #    5):
   #  2     import numpy as np
   #  3     import random
   #  4
   #  5     # 1. Calculate Luminance Variance for every block
   #  6     # self.A has shape (N, 3) -> [Y, U, V]
   #  7     variances = []
   #  8     for block in blocks:
   #  9         start, end = block.metadata.start, block.metadata.end
   # 10         # Extract the Y channel (index 0) for this block's points
   # 11         # end+1 because the end index is inclusive in your return_index()
   # 12         y_channel = self.A[start : end + 1, 0]
   # 13
   # 14         # Calculate variance (0 if block has only 1 point)
   # 15         var = np.var(y_channel) if len(y_channel) > 1 else 0.0
   # 16         variances.append(var)
   # 17
   # 18     variances = np.array(variances)
   # 19
   # 20     # 2. Create Strata based on Variance Percentiles
   # 21     # Using percentiles ensures that bins are representative even if 
   # 22     # most blocks are 'flat' (low variance).
   # 23     bins = np.percentile(variances, np.linspace(0, 100, n_strata + 1))
   # 24     bins = np.unique(bins) # Remove duplicates if many blocks have same variance
   # 25
   # 26     if len(bins) < 2:
   # 27         # Fallback if there's no variation across the whole cloud
   # 28         return random.sample(blocks, int(len(blocks) * ratio))
   # 29
   # 30     # 3. Assign blocks to bins
   # 31     # digitize returns 1-based index into bins
   # 32     bin_indices = np.digitize(variances, bins) - 1
   # 33
   # 34     strata = [[] for _ in range(len(bins) - 1)]
   # 35     for i, b_bin in enumerate(bin_indices):
   # 36         # Clip index to valid range of strata list
   # 37         idx = min(max(b_bin, 0), len(strata) - 1)
   # 38         strata[idx].append(blocks[i])
   # 39
   # 40     # 4. Sample proportionately from each stratum
   # 41     subsampled_blocks = []
   # 42     for group in strata:
   # 43         if not group: continue
   # 44
   # 45         n_to_pick = max(1, int(len(group) * ratio))
   # 46         n_to_pick = min(n_to_pick, len(group)) # safety check
   # 47
   # 48         subsampled_blocks.extend(random.sample(group, n_to_pick))
   # 49
   # 50     # 5. Maintain Morton Order
   # 51     # Keeping them sorted by original index prevents potential issues with 
   # 52     # spatial processing later.
   # 53     subsampled_blocks.sort(key=lambda b: b.block_idx)
   # 54
   # 55     logger.info(f"Subsampled {len(blocks)} blocks down to {len(subsampled_blocks)} using
   #    {n_strata} luminance strata.")
   # 56     return subsampled_blocks
   #
   #
   #

# --- Cache wrapper ---
# TODO: This is unused.
class PartitionCache:
    def __init__(self, pc: PointCloud):
        self.pc = pc
        self._cache: Dict[tuple[str, tuple], List[Block]] = {}

    def get(self, strategy: BlockPartitionStrategy, **kwargs) -> List[Block]:
        key = (strategy.__class__.__name__, tuple(kwargs.items()))
        if key not in self._cache:
            self._cache[key] = strategy.partition(self.pc, **kwargs)
        return self._cache[key]
