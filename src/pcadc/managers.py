import h5py
from pathlib import Path
from .graph import *
from typing import List, Dict, Union
from .color import *
import numpy as np
import logging
logging.basicConfig(filename="logs/objects.log", 
                    filemode="w", 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)


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
    
    def add_block(self, block: "Block") -> None:
        """Register a new block with metadata."""
        block_grp = self.file.create_group(f"blocks/{block.id}")
        block_grp.create_dataset("idxs", data=block.idxs)
        block_grp.create_group("graphs")  # Stores graph configurations

    def add_result(
        self,
        graph: StructuralGraph | AttributeGraph,
        result: tuple[np.ndarray, np.ndarray]):
        """Add a graph configuration + GFT results to a block."""

        graph_id = graph.id
        block_id = graph.block_id 
        graph_grp = self.file.create_group(f"blocks/{block_id}/graphs/{graph_id}")
        graph_grp.create_dataset("edges", data=graph.edges, compression="gzip")
        graph_grp.create_dataset("coeffs", data=result[1], compression="gzip")

    def add_decision(self, block: "Block", q_step: int, sel_graph_id: str, sel_coeff: np.ndarray):
        sl_weight, sl_percentage = map(float, sel_graph_id.split("_"))
        decision_grp = self.file.create_group(f"blocks/{block.id}/decision/{q_step}")
        decision_grp.create_dataset("sl_weight", data=sl_weight)
        decision_grp.create_dataset("sl_percentage", data=sl_percentage)
        decision_grp.create_dataset("coeffs", data=sel_coeff, compression="gzip") 

    def add_overall(self,q_step: int, psnr: float, bpv: float, bitcount: int):
        overall_grp = self.file.create_group(f"results/{q_step}")
        overall_grp.create_dataset("psnr", data=psnr)
        overall_grp.create_dataset("bpv", data=bpv)
        overall_grp.create_dataset("bitcount", data=bitcount)
        


    def matched_metadata(self, graph: StructuralGraph | AttributeGraph, rewrite=False):
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
            "edges": graph_grp["edges"][:],
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



if __name__ == "__main__":
    from .transforms import *
    from .visualization import *
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

