from visualization import *

class PointCloud():
    def __init__(self) -> None:
        # FIXME: Is ADCOlor really necessary for one operation
        self.visualizer = Visualizer()
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
        self.A = self.visualizer._RGBtoYUV(C_rgb)

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
    
    def get_block(self, index: int) -> Block:
        start_idx, end_idx = self.indexes[index]
        idxs = list(range(start_idx, end_idx+1))
        Vblock = self.V[idxs, :]
        Ablock = self.A[idxs, :]
        return Block(Vblock, Ablock, idxs)

    def get_all_blocks(self) -> list[Block]:
        return [self.get_block(index) for index,_ in enumerate(self.indexes)]