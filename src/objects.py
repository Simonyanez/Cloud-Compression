from utils.color import *
import utils.ply as ply
from pathlib import Path
# from graph.graph import *

class Block():
    def __init__(self,V: np.ndarray, A: np.ndarray ,idxs: list | np.ndarray):

        self.Vblock = V[idxs, :]
        self.Ablock = A[idxs, :]
        pass
    
    def get_structural_graph(self):
        pass
        

class PointCloud():
    def __init__(self) -> None:
        # FIXME: Is ADCOlor really necessary for one operation
        self.colourist = Colourist()
        self.V = None
        self.A = None 

    def __call__(self, point_cloud_path: Path):
        self._read_point_cloud(point_cloud_path)
        
    def _read_point_cloud(self, point_cloud_path: Path):
        self.point_cloud_name = point_cloud_path.stem
        save_path = Path("res/npy") / self.point_cloud_name
        file_conditions = Path.exists(f"{save_path}_V.npy") and Path.exists(f"{save_path}_C.npy")
        if file_conditions:
            self.V = np.load(f"{save_path}_V.npy")
            C_rgb = np.load(f"{save_path}_C.npy")
        else:
            self.V,C_rgb,_ = ply.ply_read8i(point_cloud_path)  
            np.save(f"{save_path}_V.npy",self.V)
            np.save(f"{save_path}_C.npy",C_rgb) 
        self.A = self.colourist.RGBtoYUV(C_rgb)

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
        indexes = list(zip(start_indexes,end_indexes))  # Paired start and end indexes
        self.indexes = sorted(indexes, key=lambda x: x[1]-x[0], reverse=True)
    
    def get_block(self, index: int) -> Block:
        start_idx, end_idx = self.indexes[index]
        idx = list(range(start_idx, end_idx))
        return Block(self.V, self.A, idx)

    def get_all_blocks(self) -> list[Block]:
        return [self.get_block(index) for index in self.indexes]

        
# class ADGFT():
#     def __init__(self, V, C):
#         pass

