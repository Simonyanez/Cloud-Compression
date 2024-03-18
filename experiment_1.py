
import numpy as np
from utils import ply
from utils.visualization import visualization

if __name__ == "__main__":
    V,C,J = ply.ply_read8i("res/longdress_vox10_1051.ply")
    print(f"This is geometric shape {np.shape(V[:,1])}")
    aspect_ratio = visualization(V,C,1,1)