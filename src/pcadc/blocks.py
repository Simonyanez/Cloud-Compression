# from utils.color import *
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from .graph import *
from typing import List, Dict, Union
from .color import *
import numpy as np
import logging
logging.basicConfig(filename="logs/blocks.log",
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)

# NOTE: This is a good example of the Template Pattern


@dataclass
class CoeffsContainer:
    block: Block
    graphs: List[StructuralGraph | AttributeGraph]
    coeffs: List[np.ndarray]


@dataclass
class BaseMetadata:
    start: int
    end: int

    def return_index(self):
        return np.arange(start=self.start, stop=self.end+1)


@dataclass
class BlockMetadata(BaseMetadata):
    source: str
    frame: int
    block_size: int
    block_idx: int

    def get_block_id(self):
        return f"{self.source}_{self.frame}_b{self.block_size}_{self.block_idx}"


@dataclass
class AuxiliaryBlockMetadata(BaseMetadata):
    parent_id: str
    task: str

    def get_block_id(self):
        return f"{self.parent_id}_aux_{self.task}"


class BlockBase(ABC):
    def __init__(self, metadata) -> None:
        self.metadata = metadata
        self.Vblock: np.ndarray | None = None
        self.Ablock: np.ndarray | None = None

    def __str__(self):
        return self.metadata.get_block_id()

    def init_data(self, V: np.ndarray, A: np.ndarray):
        idxs = self.metadata.return_index()
        self.Vblock = V[idxs, :]
        self.Ablock = A[idxs, :]

    def set_data(self, Vblock: np.ndarray, Ablock: np.ndarray, subidxs: np.ndarray | None = None):
        self.Vblock = Vblock
        self.Ablock = Ablock

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.Vblock is not None, "Vblock not initialized"
        assert self.Ablock is not None, "Ablock not initialized"
        return self.Vblock, self.Ablock

    def clear_data(self):
        self.Vblock = None
        self.Ablock = None


# TODO: Make blocks a more abstract class. Just represent a set of blocks, the way its initialized can vary yet it should be the definition
class Block(BlockBase):
    def __init__(self, metadata: BlockMetadata) -> None:
        super().__init__(metadata)


class AuxiliaryBlock(BlockBase):
    def __init__(self, metadata: AuxiliaryBlockMetadata) -> None:
        super().__init__(metadata)
