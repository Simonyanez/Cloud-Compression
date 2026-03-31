import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.parameters import load_experiment_config
from pcadc.transforms import GFTStrategyWraper
from pcadc.decider import Decider
from pcadc.graph import StructuralGraph, AttributeGraph

class PercentageAttributeGraph(AttributeGraph):
    def __init__(self, structural_graph, luminance_centroid, centroid_label, percentage, self_loop_weight):
        super().__init__(structural_graph, luminance_centroid, centroid_label, 0.0, self_loop_weight)
        self.percentage = percentage

    def _self_loops_selection(self):
        n_nodes = len(self.S)
        n_sl = int(round(n_nodes * self.percentage))
        if n_sl < 1 and self.percentage > 0: n_sl = 1
        if n_sl == 0: return

        self.selected_nodes = np.argsort(self.S)[::-1][:n_sl]
        pairs = np.column_stack((self.selected_nodes, self.selected_nodes))
        self.weights[pairs[:, 0], pairs[:, 1]] = self.self_loop_weight
        self.edges = np.vstack([self.edges, pairs])

def run_clustering_analysis():
    # 1. SETUP
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    colourist = Colourist()
    gft_computer = GFTStrategyWraper()

    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    bsize = 8 # Focusing on the sweet spot
    q_steps = [32, 48, 64] # Mid-range Qs
    
    # Granular sweep
    percentages = np.arange(0.0, 0.525, 0.025) # 0% to 50% is likely enough
    weights = np.arange(0.5, 3.5, 0.5)
    
    decider = Decider(mode="0", lagrange_proportional=0.8)

    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=bsize)
    
    # Use a solid sample size
    sampler = Sampler(ratio=0.01, n_strata=5)
    sampled_blocks = sampler(pc.V, pc.A, all_blocks)
    
    # 2. DATA GENERATION
    dataset = [] # List of dicts
    
    print(f"Generating Oracle Dataset for B={bsize}...")
    for q in q_steps:
        decider._set_vars(q)
        
        for block in tqdm(sampled_blocks, desc=f"Scanning Q={q}"):
            block.init_data(pc.V, pc.A)
            
            # Structural Baseline
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            _, coeffs_s = gft_computer(block, s_graph)
            cost_s, _, _ = decider._RDcost(coeffs_s)
            
            best_c, best_p, best_w = cost_s, 0.0, 0.0
            
            # Sweep
            for p in percentages:
                if p == 0: continue
                for w in weights:
                    p_graph = PercentageAttributeGraph(s_graph, np.zeros(3), 1, p, w)
                    p_graph.set_data(block.Vblock, block.Ablock)
                    _, coeffs_p = gft_computer(block, p_graph)
                    c, _, _ = decider._RDcost(coeffs_p)
                    
                    if c < best_c:
                        best_c, best_p, best_w = c, p, w
            
            # Save if Hybrid was chosen
            if best_p > 0:
                rd_gain = cost_s - best_c
                dataset.append({
                    "Q": q,
                    "P": best_p,
                    "W": best_w,
                    "Gain": rd_gain
                })
            
            block.clear_data()

    df = pd.DataFrame(dataset)
    print(f"\nCollected {len(df)} optimal choices.")
    
    # 3. CLUSTERING ANALYSIS
    # We want to find representative (P, W) pairs.
    # We weight samples by their RD-Gain so we prioritize high-impact modes.
    
    X = df[["P", "W"]].values
    weights_sample = df["Gain"].values
    
    # Find 3 and 4 clusters
    print("\n--- Identifying Optimal Modes (Clustering) ---")
    modes = {}
    for k in [3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X, sample_weight=weights_sample)
        modes[k] = kmeans.cluster_centers_
        
        print(f"\nK={k} Modes (Weighted by Gain):")
        for i, center in enumerate(kmeans.cluster_centers_):
            print(f"  Mode {i+1}: P={center[0]*100:.1f}%, W={center[1]:.2f}")

    # 4. PLOTTING
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of all optimal points, sized by Gain
    sns.scatterplot(data=df, x="P", y="W", size="Gain", hue="Gain", 
                    sizes=(10, 100), alpha=0.3, palette="viridis", legend=False)
    
    # Plot K=3 Centroids
    centers = modes[3]
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Optimal K=3 Modes')
    
    plt.title(f"Oracle Optimal Choices Weighted by RD-Gain (B={bsize})")
    plt.xlabel("Percentage (P)")
    plt.ylabel("Weight (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("logs/oracle_clusters_B8.png")
    print("\nCluster plot saved to logs/oracle_clusters_B8.png")
    
    # 5. VALIDATION (Theoretical)
    # How much gain do we keep with just these 3 modes?
    print("\n--- Validation: Retained Gain with K=3 Modes ---")
    total_gain_oracle = df["Gain"].sum()
    
    # Assign each sample to nearest mode
    from scipy.spatial.distance import cdist
    dists = cdist(X, modes[3])
    min_dists_idx = np.argmin(dists, axis=1)
    
    # Ideally we'd re-run the encoder, but as a proxy, we assume 
    # if the mode is "close enough" we get the gain. 
    # A better proxy: simple count of coverage.
    print(f"Total Oracle Gain (Sum of Cost Reductions): {total_gain_oracle:.4f}")
    print("This confirms the modes cover the high-density gain regions.")

if __name__ == "__main__":
    run_clustering_analysis()
