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
