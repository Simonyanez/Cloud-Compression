import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, PointCloudMetadata, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator

def analyze_slopes():
    # 1. Setup
    pc_path = Path("res/longdress_vox10_1051.ply")
    metadata = PointCloudMetadata(dataset="8i", sequence="longdress", depth=10, frame=1051)
    colourist = Colourist()
    approximator = Approximator()
    
    pc = PointCloud.from_file(pc_path, "ply", metadata)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    partitioner = MortonBlockPartition()
    _, blocks = partitioner.partition(pc, bsize=16)
    
    # 2. Extract Oracle Slopes and RMSE for every block
    print(f"Analyzing slopes for {len(blocks)} blocks...")
    oracle_slopes = []
    oracle_rmses = []
    point_counts = []
    
    for block in blocks:
        block.init_data(pc.V, pc.A)
        fit = approximator(block) # This does the LSQ fit
        oracle_slopes.append(fit.coeffs[1:]) # Skip intercept if it exists or take the 3 coeffs
        oracle_rmses.append(fit.rmse)
        point_counts.append(block.Vblock.shape[0])
        block.clear_data()
        
    oracle_slopes = np.array(oracle_slopes)
    oracle_rmses = np.array(oracle_rmses)
    
    # 3. Statistical Analysis
    slope_means = np.mean(oracle_slopes, axis=0)
    slope_stds = np.std(oracle_slopes, axis=0)
    
    print("\n--- Slope Statistics (X, Y, Z components) ---")
    print(f"Mean Slope: {slope_means}")
    print(f"Std Dev:    {slope_stds}")
    print(f"Avg RMSE:   {np.mean(oracle_rmses):.4f}")
    
    # 4. Check for Natural Clustering in Slope Space
    # Using 8 clusters as a test
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(oracle_slopes)
    
    # Calculate "Quantization Error" in slope space
    # (How far is each block from its nearest cluster slope)
    closest_centroid = kmeans.cluster_centers_[kmeans.labels_]
    slope_errors = np.linalg.norm(oracle_slopes - closest_centroid, axis=1)
    
    print(f"\n--- Slope Space Clustering ({n_clusters} clusters) ---")
    print(f"Avg Slope Quantization Error: {np.mean(slope_errors):.4f}")
    print(f"Max Slope Quantization Error: {np.max(slope_errors):.4f}")

    # 5. Plotting Distribution
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Slope Distribution (Scatter X vs Y)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(oracle_slopes[:, 0], oracle_slopes[:, 1], oracle_slopes[:, 2], alpha=0.1)
    ax1.set_title("Oracle Slopes Distribution (3D)")
    
    # Plot 2: Histogram of RMSE (Residuals)
    ax2 = fig.add_subplot(132)
    ax2.hist(oracle_rmses, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title("Histogram of Fit RMSE (Residuals)")
    ax2.set_xlabel("RMSE")
    
    # Plot 3: Slope Error vs Point Count
    ax3 = fig.add_subplot(133)
    ax3.scatter(point_counts, oracle_rmses, alpha=0.1)
    ax3.set_title("RMSE vs Block Density")
    ax3.set_xlabel("Points in Block")
    ax3.set_ylabel("RMSE")
    
    plt.tight_layout()
    plt.savefig("logs/slope_diversity_analysis.png")
    print("\nAnalysis plot saved to logs/slope_diversity_analysis.png")

if __name__ == "__main__":
    analyze_slopes()
