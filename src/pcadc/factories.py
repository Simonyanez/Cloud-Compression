# from utils.color import *
from . import ply
import h5py
import uuid
from pathlib import Path
from .graph import *
from .blocks import *
from .clusterer import *
from .parameters import *
from typing import List, Dict, Union, Tuple
from .color import *
from abc import ABC, abstractmethod
import numpy as np
import logging
logging.basicConfig(filename="logs/objects.log",
                    filemode="w",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)

# FIXME: EDGES ARE NOT CORRECT. YET THEY ARE NEVER USED SO I DONT CARE
# NOTE: This is a really simple Factory Pattern
# TODO: Improve to a real Factory Pattern


class Creator(ABC):
    @abstractmethod
    def factory_method(self) -> Tuple[Block, GraphBase | StructuralGraph | AttributeGraph]:
        pass


class GraphBlockCreator(Creator):
    def __init__(self,
                 V: np.ndarray,
                 A: np.ndarray,
                 block: Block,
                 luminance_centroid: np.ndarray,
                 parameters: SequentialParameters):
        self.V = V
        self.A = A
        self.block = block
        self.luminance_centroid = luminance_centroid
        self.self_loop_threshold = parameters.self_loop_threshold
        self.self_loop_weight = parameters.self_loop_weight

    def factory_method(self) -> Tuple[Block, StructuralGraph | AttributeGraph]:
        self.block.init_data(self.V, self.A)
        Vblock, Ablock = self.block.get_data()

        structural_graph = StructuralGraph(self.block.metadata)
        structural_graph.set_data(V=Vblock)

        if not self.is_structural:
            attribute_graph = AttributeGraph(structural_graph, self.luminance_centroid,
                                             self.self_loop_threshold, self.self_loop_weight)
            Ablock_app = Ablock.copy()
            Ablock_app[:, 0] = Vblock @ self.luminance_centroid.T
            # decorate graph with attributes
            attribute_graph.set_data(V=Vblock, A=Ablock_app)
            return self.block, attribute_graph
        self.block.clear_data()
        return self.block, structural_graph

    def get_all_products(self):
        _, graph = self.factory_method()
        logger.debug(
            f"Luminance centroid for this block is: {self.luminance_centroid}")
        if not self.is_structural:
            logger.debug(
                f"Block detected as not structural only")
            structural_graph = self._force_structural_graph()
            return [structural_graph, graph]
        logger.debug(f"Block was structural only")
        return [graph]

    def _force_structural_graph(self):
        _centroid_cache = self.luminance_centroid.copy()
        self.luminance_centroid = np.array([0, 0, 0])
        _, structural_graph = self.factory_method()
        self.luminance_centroid = _centroid_cache
        return structural_graph

    @property
    def is_structural(self):
        return np.equal(self.luminance_centroid, np.array([0, 0, 0])).any()


class SubGraphCreator(Creator):
    def __init__(self,
                 parent_block: Block,
                 parent_graph: StructuralGraph | AttributeGraph,
                 sub_idxs: np.ndarray,
                 task: str = "Disconnected Component"):
        self.parent_block = parent_block
        self.parent_graph = parent_graph
        self.sub_idxs = sub_idxs
        self.task = task
        pass

    def factory_method(self):
        Vblock, Ablock = self.parent_block.get_data()
        Asubblock = Ablock[self.sub_idxs, :]
        Vsubblock = Vblock[self.sub_idxs, :]
        weights, edges = self.parent_graph.get_data()
        weights_sub = weights[self.sub_idxs, :][:, self.sub_idxs]
        # FIXME: This is so Wrong. Edges are not used though
        edges_sub = np.zeros((0, 2), dtype=np.int64)
        # NOTE: SUB IDX -1 is quick fix for one point blocks
        sub_block_metadata = AuxiliaryBlockMetadata(start=self.parent_block.get_absolute_idx(self.sub_idxs[0]),
                                                    end=self.parent_block.get_absolute_idx(
                                                        self.sub_idxs[-1]),
                                                    parent_id=self.parent_block.block_id,
                                                    task=self.task)
        sub_block = AuxiliaryBlock(sub_block_metadata)
        sub_block.set_data(Vsubblock, Asubblock)

        # TODO: Not sure is this is a good idea
        sub_graph_metadata = GraphMetadata(block_id=sub_block_metadata.block_id,
                                           graph_type="Sub-graph",
                                           distance_threshold=np.sqrt(3),
                                           luminance_centroid=np.array(
                                               [0, 0, 0]),
                                           self_loop_threshold=None,
                                           self_loop_weight=None)
        sub_graph = GraphBase(sub_graph_metadata)
        sub_graph.set_data(weights=weights_sub,
                           edges=edges_sub)
        return sub_block, sub_graph


class MeanGraphFactory(Creator):
    def __init__(self,
                 parent_block: Block,
                 components_Vmean: np.ndarray,
                 num_components: int,
                 task: str = "Mean of Components"):
        self.parent_block = parent_block
        self.components_Vmean = components_Vmean
        self.num_components = num_components
        self.task = task

    def factory_method(self):
        mean_block_metadata = AuxiliaryBlockMetadata(start=0,
                                                     end=self.num_components-1,
                                                     parent_id=self.parent_block.block_id,
                                                     task=self.task
                                                     )
        mean_block = AuxiliaryBlock(mean_block_metadata)
        mean_block.set_data(Vblock=self.components_Vmean,
                            Ablock=self.components_Vmean, subidxs=None)
        mean_graph_metadata = GraphMetadata(block_id=mean_block_metadata.block_id,
                                            graph_type="Mean-Graph",
                                            distance_threshold=np.inf,
                                            luminance_centroid=np.array(
                                                [0, 0, 0]),
                                            self_loop_threshold=None,
                                            self_loop_weight=None)
        mean_graph = StructuralGraph(mean_block_metadata)
        mean_graph.set_metadata(mean_graph_metadata)
        mean_graph.set_data(self.components_Vmean)
        return mean_block, mean_graph


# class BlockManager:
#     def __init__(self, bsize: int, export_folder: Path, experiment_code: str, point_cloud_path: Path, rewrite=False):
#         point_cloud_name = point_cloud_path.stem
#         self.hdf5_path = export_folder / Path(f"{experiment_code}/{point_cloud_name}/block_size{bsize}_data.h5")
#         if rewrite and self.hdf5_path.exists():
#             self.hdf5_path.unlink()
#
#         self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
#         self.file = h5py.File(self.hdf5_path, "a")
#         if "blocks" not in self.file:
#             self.file.create_group("blocks")
#
#     def add_block(self, block: "Block") -> None:
#         """Register a new block with metadata."""
#         block_grp = self.file.create_group(f"blocks/{block.id}")
#         block_grp.create_dataset("idxs", data=block.idxs)
#         block_grp.create_group("graphs")  # Stores graph configurations
#
#     def add_result(
#         self,
#         graph: StructuralGraph | AttributeGraph,
#         result: tuple[np.ndarray, np.ndarray]):
#         """Add a graph configuration + GFT results to a block."""
#
#         graph_id = graph.id
#         block_id = graph.block_id
#         graph_grp = self.file.create_group(f"blocks/{block_id}/graphs/{graph_id}")
#         graph_grp.create_dataset("edges", data=graph.edges, compression="gzip")
#         graph_grp.create_dataset("coeffs", data=result[1], compression="gzip")
#
#     def add_decision(self, block: "Block", q_step: int, sel_graph_id: str, sel_coeff: np.ndarray):
#         sl_weight, sl_percentage = map(float, sel_graph_id.split("_"))
#         decision_grp = self.file.create_group(f"blocks/{block.id}/decision/{q_step}")
#         decision_grp.create_dataset("sl_weight", data=sl_weight)
#         decision_grp.create_dataset("sl_percentage", data=sl_percentage)
#         decision_grp.create_dataset("coeffs", data=sel_coeff, compression="gzip")
#
#     def add_overall(self,q_step: int, psnr: float, bpv: float, bitcount: int):
#         overall_grp = self.file.create_group(f"results/{q_step}")
#         overall_grp.create_dataset("psnr", data=psnr)
#         overall_grp.create_dataset("bpv", data=bpv)
#         overall_grp.create_dataset("bitcount", data=bitcount)
#
#
#
#     def matched_metadata(self, graph: StructuralGraph | AttributeGraph, rewrite=False):
#         if rewrite:
#             return False
#         graph_id = graph.id
#         block_id = graph.block_id
#         return f"blocks/{block_id}/graphs/{graph_id}" in self.file
#
#     def get_config_data(
#         self, block_id: int, graph_id: str
#     ) -> Dict[str, np.ndarray]:
#         """Load all data for a specific configuration."""
#         graph_grp = self.file[f"blocks/{block_id}/graphs/{graph_id}"]
#         return {
#             "edges": graph_grp["edges"][:],
#             "coeffs": graph_grp["coeffs"][:],
#
#         }
#
#     def get_data(self, block_id: int, graph_id: str, h5_key: str) -> np.ndarray:
#         graph_grp = self.file[f"blocks/{block_id}/graphs/{graph_id}"]
#         return graph_grp[h5_key][:]
#
#     def get_graph_metadata(self, block_id: str, graph_id: str) -> np.ndarray:
#         adjacency = self.get_data(block_id, graph_id, h5_key="adjacency")[:]
#         diag = np.diag(adjacency)
#         sl_pos = diag > 0
#         sl_count = np.sum(sl_pos)
#         sl_percentage = 100*sl_count/adjacency.shape[0]
#         sl_weight = 0
#         if sl_count > 0:
#             sl_weight = diag[sl_pos][0]
#         metadata = (sl_weight, sl_percentage, sl_count)
#         return metadata
#
#     def list_blocks(self) -> List[int]:
#         return list(self.file["blocks"].keys())
#
#     def list_graphs(self, block_id: UUID) -> List[UUID]:
#         """List all graphs IDs for a block."""
#         return list(self.file[f"blocks/{block_id}/graphs"].keys())
#
#     def get_coefficients(self, block_id:UUID) -> List[np.ndarray]:
#         return {graph_id:self.get_data(block_id, graph_id, h5_key='coeffs') for graph_id in self.list_graphs(block_id)}
#
#     def close(self):
#         self.file.close()
#
#     # Context manager support
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()
#
# # TODO: Make blocks a more abstract class. Just represent a set of blocks, the way its initialized can vary yet it should be the definition
# class Block():
#     def __init__(self ,idxs: tuple[int, int], block_num: str):
#         self.id: str = block_num
#         self.idxs: tuple[int, int] = idxs
#
#     def __str__(self):
#         return f""" Block at [{self.idxs[0], self.idxs[1]}] with UUID: {self.id}"""
#
#     def _init_data(self, V: np.ndarray, A: np.ndarray):
#         self.Vblock = V[self.as_index(),:]
#         self.Ablock = A[self.as_index(),:]
#
#     def _init_auxiliary(self, Vblock: np.ndarray, Ablock:np.ndarray, subidxs: np.ndarray):
#         self.Vblock = Vblock
#         self.Ablock = Ablock
#         self.subidxs = subidxs
#
#     def _del_data(self):
#         self.Vblock = None
#         self.Ablock = None
#
#     def get_data(self) -> tuple[np.ndarray, np.ndarray]:
#         assert self.Vblock is not None, "Vblock hasn't been initialized"
#         assert self.Ablock is not None, "Ablock hasn't been initialized"
#         return (self.Vblock, self.Ablock)
#
#     def set_data(self, Vblock: np.ndarray, Ablock: np.ndarray):
#         # assert self.Vblock is not None, "Vblock hasn't been initialized"
#         # assert self.Ablock is not None, "Ablock hasn't been initialized"
#         self.Vblock = Vblock
#         self.Ablock = Ablock
#
#     def as_index(self):
#         return np.arange(start=self.idxs[0], stop=self.idxs[1]+1) # Include end index
#
# class PointCloud():
#     def __init__(self) -> None:
#         # FIXME: Is ADCOlor really necessary for one operation
#         self.colourist = Colourist()
#         self.V: Optional[np.ndarray] = None
#         self.A: Optional[np.ndarray] = None
#
#     def __call__(self, point_cloud_path: Path):
#         self._read_point_cloud(point_cloud_path)
#
#     def _read_point_cloud(self, point_cloud_path: Path):
#         # Set the point cloud name (stem of the file path)
#         self.point_cloud_name = point_cloud_path.stem
#
#         # Define the save directory and file paths
#         save_dir = Path("res/npy")
#         save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
#         v_file = save_dir / f"{self.point_cloud_name}_V.npy"
#         c_file = save_dir / f"{self.point_cloud_name}_C.npy"
#
#         # Check if the .npy files already exist
#         if v_file.exists() and c_file.exists():
#             # Load the existing .npy files
#             self.V = np.load(v_file)
#             C_rgb = np.load(c_file)
#         else:
#             # Read the point cloud from the original file
#             self.V, C_rgb, _ = ply.ply_read8i(point_cloud_path)
#
#             # Save the point cloud data as .npy files
#             np.save(v_file, self.V)
#             np.save(c_file, C_rgb)
#
#         # Convert RGB to YUV using the colourist
#         self.A = self.colourist._RGBtoYUV(C_rgb)
#
#     def do_block_partitioning(self, bsize: int) -> None:
#         # Assumes point cloud is morton ordered
#         base_block_size = np.log2(bsize)
#         assert np.all(np.floor(base_block_size) == base_block_size), "block size b should be a power of 2"
#         V_coarse = np.floor(self.V / bsize) * bsize
#         variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)
#         variation = np.concatenate(([1], variation))
#
#         start_indexes = np.nonzero(variation)[0]
#         Nlevel = self.V.shape[0]
#         end_indexes = np.concatenate((start_indexes[1:] - 1, np.array([Nlevel - 1])))
#         self.indexes = list(zip(start_indexes,end_indexes))  # Paired start and end indexes
#         # self.indexes = sorted(indexes, key=lambda x: x[1]-x[0], reverse=True)
#
#     def get_block(self, index: int) -> tuple[int, int]:
#         start_end_tuple = self.indexes[index]
#         return Block(idxs=start_end_tuple, block_num=index)
#
#     def get_all_blocks(self) -> list[Block]:
#         return [self.get_block(index) for index, _ in enumerate(self.indexes)]
if __name__ == "__main__":
    from .transforms import *
    from .visualization import *
    GFT_computer = GFT()
    visualizer = Visualizer()
    point_cloud = PointCloud()
    point_cloud(Path("res/longdress_vox10_1051.ply"))
    point_cloud.do_block_partitioning(bsize=16)
    block = point_cloud.get_block(200)
    graph = AttributeGraph(block.Vblock, block.Ablock,
                           sl_weight=5, block_fraction=0.05)
    visualizer(graph, block)
    visualizer.visualize_block()
    visualizer.add_selected_nodes()
    visualizer.visualize_coeffs(title="Attribute")
    visualizer(block.structural_graph, block)
    visualizer.visualize_coeffs(title="Structural")
    visualizer.display()


# class ADGFT():
#     def __init__(self, V, C):
#         pas
