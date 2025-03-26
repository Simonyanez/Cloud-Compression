# from utils.color import *
import ply as ply
import h5py
from uuid import *
from pathlib import Path
from graph import *
from typing import List, Dict
from color import *
import numpy as np

class BlockManager:
    def __init__(self, bsize: int, export_folder: Path, experiment_code: str, point_cloud_path: Path, rewrite=False):
        point_cloud_name = point_cloud_path.stem
        self.hdf5_path = export_folder / Path(f"{experiment_code}/{point_cloud_name}/block_size{bsize}_data.h5")
        if rewrite and self.hdf5_path.exists():
            self.hdf5_path.unlink()
            
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.hdf5_path, "a")
        if "blocks" not in self.file:
            self.file.create_group("blocks")
    
    def add_block(self, block: "Block") -> UUID:
        """Register a new block with metadata."""
        block_grp = self.file.create_group(f"blocks/{block.id}")
        block_grp.create_dataset("idxs", data=block.idxs)
        block_grp.create_group("graphs")  # Stores graph configurations

    def add_result(
        self,
        graph: Graph,
        result: tuple[np.ndarray, np.ndarray]):
        """Add a graph configuration + GFT results to a block."""

        graph_id = graph.id
        block_id = graph.block_id 
        graph_grp = self.file.create_group(f"blocks/{block_id}/graphs/{graph_id}")
        graph_grp.create_dataset("adjacency", data=graph.weights, compression="gzip")
        graph_grp.create_dataset("edges", data=graph.edges, compression="gzip")
        graph_grp.create_dataset("gft_mat", data=result[0], compression="gzip")
        graph_grp.create_dataset("coeffs", data=result[1], compression="gzip")

    def matched_metadata(self, graph: Graph, rewrite=False):
        if rewrite:
            return False
        graph_id = graph.id
        block_id = graph.block_id
        return f"blocks/{block_id}/graphs/{graph_id}" in self.file

    def get_config_data(
        self, block_id: int, graph_id: str
    ) -> Dict[str, np.ndarray]:
        """Load all data for a specific configuration."""
        graph_grp = self.file[f"blocks/{block_id}/graphs/{graph_id}"]
        return {
            "adjacency": graph_grp["adjacency"][:],
            "edges": graph_grp["edges"][:],
            "gft_mat": graph_grp["gft_mat"][:],
            "coeffs": graph_grp["coeffs"][:],
            
        }

    def get_data(self, block_id: int, graph_id: str, h5_key: str) -> np.ndarray:
        graph_grp = self.file[f"blocks/{block_id}/graphs/{graph_id}"]
        return graph_grp[h5_key][:]
    
    def get_graph_metadata(self, block_id: str, graph_id: str) -> np.ndarray:
        adjacency = self.get_data(block_id, graph_id, h5_key="adjacency")[:]
        diag = np.diag(adjacency)
        sl_pos = diag > 0
        sl_count = np.sum(sl_pos)
        sl_percentage = 100*sl_count/adjacency.shape[0]
        sl_weight = 0
        if sl_count > 0:
            sl_weight = diag[sl_pos][0]
        metadata = (sl_weight, sl_percentage, sl_count)
        return metadata

    def list_blocks(self) -> List[int]:
        return list(self.file["blocks"].keys())
    
    def list_graphs(self, block_id: UUID) -> List[UUID]:
        """List all graphs IDs for a block."""
        return list(self.file[f"blocks/{block_id}/graphs"].keys())

    def get_coefficients(self, block_id:UUID) -> List[np.ndarray]:
        return {graph_id:self.get_data(block_id, graph_id, h5_key='coeffs') for graph_id in self.list_graphs(block_id)}

    def close(self):
        self.file.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class Block():
    def __init__(self ,idxs: tuple[int, int], block_num: int | np.ndarray):
        self.id: int = block_num
        self.idxs: tuple[int, int] = idxs

    def __str__(self):
        return f""" Block at [{self.idxs[0], self.idxs[1]}] with UUID: {self.id}"""

    def _init_data(self, V: np.ndarray, A: np.ndarray):
        self.Vblock = V[self.as_index()]
        self.Ablock = A[self.as_index()]

    def _init_auxiliary(self, Vblock: np.ndarray, Ablock:np.ndarray, subidxs: np.ndarray):
        self.Vblock = Vblock
        self.Ablock = Ablock
        self.subidxs = subidxs

    def _del_data(self):
        self.Vblock = None
        self.Ablock = None

    def as_index(self):
        return np.arange(start=self.idxs[0], stop=self.idxs[1]+1) # Include end index       

class PointCloud():
    def __init__(self) -> None:
        # FIXME: Is ADCOlor really necessary for one operation
        self.colourist = Colourist()
        self.V: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None 

    def __call__(self, point_cloud_path: Path):
        self._read_point_cloud(point_cloud_path)
        
    def _read_point_cloud(self, point_cloud_path: Path):
        # Set the point cloud name (stem of the file path)
        self.point_cloud_name = point_cloud_path.stem

        # Define the save directory and file paths
        save_dir = Path("res/npy")
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        v_file = save_dir / f"{self.point_cloud_name}_V.npy"
        c_file = save_dir / f"{self.point_cloud_name}_C.npy"

        # Check if the .npy files already exist
        if v_file.exists() and c_file.exists():
            # Load the existing .npy files
            self.V = np.load(v_file)
            C_rgb = np.load(c_file)
        else:
            # Read the point cloud from the original file
            self.V, C_rgb, _ = ply.ply_read8i(point_cloud_path)

            # Save the point cloud data as .npy files
            np.save(v_file, self.V)
            np.save(c_file, C_rgb)

        # Convert RGB to YUV using the colourist
        self.A = self.colourist._RGBtoYUV(C_rgb)

    def do_block_partitioning(self, bsize: int) -> None:
        # Assumes point cloud is morton ordered
        base_block_size = np.log2(bsize) 
        assert np.all(np.floor(base_block_size) == base_block_size), "block size b should be a power of 2"
        V_coarse = np.floor(self.V / bsize) * bsize
        variation = np.sum(np.abs(V_coarse[1:] - V_coarse[:-1]), axis=1)
        variation = np.concatenate(([1], variation))

        start_indexes = np.nonzero(variation)[0]
        Nlevel = self.V.shape[0]
        end_indexes = np.concatenate((start_indexes[1:] - 1, np.array([Nlevel - 1])))
        self.indexes = list(zip(start_indexes,end_indexes))  # Paired start and end indexes
        # self.indexes = sorted(indexes, key=lambda x: x[1]-x[0], reverse=True)
    
    def get_block(self, index: int) -> tuple[int, int]:
        start_end_tuple = self.indexes[index]
        return Block(idxs=start_end_tuple, block_num=index) 

    def get_all_blocks(self) -> list[Block]:
        return [self.get_block(index) for index, _ in enumerate(self.indexes)]

if __name__ == "__main__":
    from transforms import *
    from visualization import *
    GFT_computer = GFT()
    visualizer = Visualizer()
    point_cloud = PointCloud()
    point_cloud(Path("res/longdress_vox10_1051.ply"))
    point_cloud.do_block_partitioning(bsize = 16)
    block = point_cloud.get_block(200)
    graph = AttributeGraph(block.Vblock, block.Ablock, sl_weight=5, block_fraction=0.05)
    visualizer(graph, block)
    visualizer.visualize_block()
    visualizer.add_selected_nodes()
    visualizer.visualize_coeffs(title="Attribute")
    visualizer(block.structural_graph, block)
    visualizer.visualize_coeffs(title="Structural")
    visualizer.display()
    
    
        
# class ADGFT():
#     def __init__(self, V, C):
#         pass

