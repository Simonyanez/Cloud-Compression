from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import numpy as np

# NOTE: Strategy pattern
class PointCloudReader(ABC):

    @abstractmethod
    def read(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        pass

class PLYReader(PointCloudReader):

    def read(self, path: Path, dataset: str = "8iVFB") -> tuple[np.ndarray, np.ndarray]:
        if dataset == "8iVFB":
            return self.read_8iVFB(path)
        if dataset == "MVUB":
            return self.read_MVUB(path)
        else:
            raise ValueError("Dataset read method not available")


    def read_8iVFB(self, path: Path):
        with open(path, 'r', encoding='UTF-8') as fid:
            fid.readline()  # Read 'ply\n'
            fid.readline()  # Read 'format ascii 1.0\n'
            fid.readline()  # Read 'comment Version 2, Copyright 2017, 8i Labs, Inc.\n'
            fid.readline()  # Read 'comment frame_to_world_scale %g\n'
            fid.readline()  # Read 'comment frame_to_world_translation %g %g %g\n'
            w = int(fid.readline().split()[2])  # Read 'comment width %d\n'
            N = int(fid.readline().split()[2])  # Read 'element vertex %d\n'

            fid.readline()  # Read 'property float x\n'
            fid.readline()  # Read 'property float y\n'
            fid.readline()  # Read 'property float z\n'
            fid.readline()  # Read 'property uchar red\n'
            fid.readline()  # Read 'property uchar green\n'
            fid.readline()  # Read 'property uchar blue\n'
            fid.readline()  # Read 'end_header'

            lines = fid.readlines()
            A = [list(map(float, line.strip().split())) for line in lines]

        V = np.array([[row[0], row[1], row[2]] for row in A])           # As numpy array
        C = np.array([[row[3], row[4], row[5]] for row in A])           # As numpy array
        J = int(np.log2(w + 1))
        # TODO: What the hell is J

        return V, C

    def read_MVUB(self, path:Path):
        with open(path, 'r', encoding='UTF-8') as fid:
            fid.readline()  # Read 'ply\n'
            fid.readline()  # Read 'format ascii 1.0\n'
            N = int(fid.readline().split()[2])  # Read 'element vertex %d\n'

            fid.readline()  # Read 'property float x\n'
            fid.readline()  # Read 'property float y\n'
            fid.readline()  # Read 'property float z\n'
            fid.readline()  # Read 'property uchar red\n'
            fid.readline()  # Read 'property uchar green\n'
            fid.readline()  # Read 'property uchar blue\n'

            lines = fid.readlines()
            A = [list(map(float, line.strip().split())) for line in lines]

        V = np.array([[row[0], row[1], row[2]] for row in A])
        C = np.array([[row[3], row[4], row[5]] for row in A])
            
        return V, C

class NPYReader(PointCloudReader):
    def read(self, path: Path):
        V = np.load(path.with_suffix("_V.npy"))
        C = np.load(path.with_suffix("_C.npy"))
        return V, C

# NOTE: Strategy pattern
class PointCloudWriter(ABC):
    @abstractmethod
    def write(self, path: Path, V:np.ndarray, C:np.ndarray):
        pass    

class PLYWriter(PointCloudWriter):
    def write(self, path: Path, V:np.ndarray, C:np.ndarray, F: Optional[np.ndarray]=None):
        N = V.shape[0]
        M = F.shape[0] if F is not None else 0

        with open(path, 'w', encoding='UTF-8') as fid:
            fid.write('ply\n')
            fid.write('format ascii 1.0\n')
            fid.write(f'element vertex {N}\n')
            fid.write('property int x\n')
            fid.write('property int y\n')
            fid.write('property int z\n')
            fid.write('property uchar red\n')
            fid.write('property uchar green\n')
            fid.write('property uchar blue\n')
            if M > 0:
                fid.write(f'element face {M}\n')
                fid.write('property list uchar int vertex_index\n')
            fid.write('end_header\n')
            
            for i in range(N):
                fid.write(f'{V[i, 0]} {V[i, 1]} {V[i, 2]} {C[i, 0]} {C[i, 1]} {C[i, 2]}\n')

            if F is not None:
                for i in range(M):
                    fid.write(f'3 {F[i, 0]} {F[i, 1]} {F[i, 2]}\n')

# NOTE: Facade pattern
class PointCloudIO:
    readers = {"ply": PLYReader(), "npy": NPYReader()}
    writers = {"ply": PLYWriter()}
    
    @staticmethod
    def load(path: Path, fmt: str = "ply"):
        if fmt not in PointCloudIO.readers:
            raise ValueError(f"Unsupported format {fmt}")
        return PointCloudIO.readers[fmt].read(path)

    @staticmethod
    def save(path: Path, V, C, fmt: str = "ply", F=None):
        if fmt not in PointCloudIO.writers:
            raise ValueError(f"Unsupported format {fmt}")
        return PointCloudIO.writers[fmt].write(path, V, C, F)
