from utils.color import *
# from graph.graph import *

class Block():
    def __init__(self,V: np.ndarray, A: np.ndarray ,idxs: list | np.ndarray):

        self.Vblock = V[idxs, :]
        self.Ablock = A[idxs, :]
        pass
    
    def get_structural_graph(self):
        pass
        

class ADPointCloud():
    def __init__(self, V: np.ndarray, C:np.ndarray, bsize:int):
        # FIXME: Is ADCOlor really necessary for one operation
        self.ad_color = ADColor()
        self.V = V
        self.A = self.ad_color.RGBtoYUV(C, rounding=True)
        self.block_partitioning(bsize=bsize)

    def block_partitioning(self, bsize: int) -> list:
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

