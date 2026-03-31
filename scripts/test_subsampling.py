import sys
from pathlib import Path
import numpy as np

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, PointCloudMetadata, MortonBlockPartition, Sampler
from pcadc.color import Colourist

def test_subsampling():
    # 1. Setup metadata and load point cloud
    pc_path = Path("res/longdress_vox10_1051.ply")
    if not pc_path.exists():
        print(f"Error: {pc_path} not found.")
        return

    metadata = PointCloudMetadata(
        dataset="8i",
        sequence="longdress",
        depth=10,
        frame=1051
    )
    
    print(f"Loading point cloud from {pc_path}...")
    pc = PointCloud.from_file(pc_path, "ply", metadata)
    print(f"Loaded {len(pc.V)} points.")

    # 2. Convert to YUV (needed for luminance-based sampling)
    print("Converting attributes to YUV...")
    colourist = Colourist()
    pc.transform_attributes(lambda A: colourist._RGBtoYUV(A))

    # 3. Partition into blocks
    bsize = 16
    print(f"Partitioning into blocks of size {bsize}...")
    partitioner = MortonBlockPartition()
    _, blocks = partitioner.partition(pc, bsize=bsize)
    print(f"Created {len(blocks)} blocks.")

    # 4. Test Sampler
    ratio = 0.1
    n_strata = 5
    print(f"Initializing Sampler with ratio={ratio}, n_strata={n_strata}...")
    sampler = Sampler(ratio=ratio, n_strata=n_strata)
    
    print("Performing subsampling...")
    sampled_blocks = sampler(pc.V, pc.A, blocks)
    
    # 5. Report results
    print("\nSubsampling Results:")
    print(f"Total blocks: {len(blocks)}")
    print(f"Sampled blocks: {len(sampled_blocks)}")
    print(f"Effective ratio: {len(sampled_blocks)/len(blocks):.4f} (Target: {ratio})")
    
    # Check if blocks are unique
    unique_ids = len(set(b.metadata.block_idx for b in sampled_blocks))
    print(f"Unique block IDs: {unique_ids}")
    
    if unique_ids != len(sampled_blocks):
        print("WARNING: Sampled blocks contain duplicates!")

    # Check some variances
    if sampled_blocks:
        variances = []
        for b in sampled_blocks[:5]:
            # Temporarily init data to check variance
            b.init_data(pc.V, pc.A)
            var = np.var(b.Ablock[:, 0])
            variances.append(var)
            b.clear_data()
        print(f"Sample variances (first 5): {['%.2f' % v for v in variances]}")

if __name__ == "__main__":
    test_subsampling()
