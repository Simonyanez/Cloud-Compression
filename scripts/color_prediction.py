import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from scipy.spatial import cKDTree
from pcadc.graph import *
from pcadc.objects import *
from pcadc.visualization import *
from pcadc.transforms import *

# Configure logging
log_path = f"logs/{Path(__file__).stem}.log"
with open(log_path, 'w') as f:
    pass
logging.basicConfig(filename=log_path, filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def initialize_graphs(block: Block) -> np.ndarray:
    # Structural Initialization just geometric data
    structural_graph = StructuralGraph(block.id)
    structural_graph._init_data(block.Vblock)

    # Attribute Initialization with self-looped method
    attribute_graph = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.7)
    attribute_graph._init_data(block.Vblock, block.Ablock)
    
    return (structural_graph, attribute_graph)

def center_block(Vblock: np.ndarray) -> np.ndarray:
    return Vblock - np.mean(Vblock, axis = 0)

def rotate_block(Vblock: np.ndarray) -> np.ndarray:
    cov_matrix = np.cov(Vblock, rowvar=False) # ROW VAR MEAN THAT EACH ROW IS A OBSERVATION
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    # Canonical
    eigvecs[:, 0] = np.sign(eigvecs[0, 0]) * eigvecs[:, 0]
    eigvecs[:, 1] = np.sign(eigvecs[1, 1]) * eigvecs[:, 1]
    eigvecs[:, 2] = np.cross(eigvecs[:, 0], eigvecs[:, 1])  # Ensure right-handed basis
    return Vblock @ eigvecs

def normalize_block(block: Block) -> None:
    # Extract block values
    Vblock = block.Vblock
    Ablock = block.Ablock

    # Normalization
    Vblock_centered = center_block(Vblock)
    Vblock_rotated = rotate_block(Vblock_centered)
    block.set_data(Vblock_rotated, Ablock)
    
def visualize_normalization(visualizer: Visualizer, block: Block, graph: StructuralGraph | AttributeGraph): 
    visualizer(graph, block)
    visualizer.visualize_block(title="Visualización del Bloque")
    visualizer.visualize_graph(title="Visualización del Grafo")
    normalize_block(block)
    visualizer.visualize_block(title="Visualización del Bloque Normalizado")
    
def block_luminansce_fit(
    block: Block,
   degree: int = 1,
):
    """
    Recursively fit a polynomial function Y(x, y, z) with increasing degree
    until RMSE is below a threshold or max_degree is reached.

    Parameters:
    - Vblock: (N, 3) array of spatial coordinates (centered + aligned)
    - Y: (N,) array of luminance values
    - degree: Polynomial degree 

    Returns:
    - Y_pred: predicted Y values
    - coeffs: final polynomial coefficients
    """
    # Normalized block
    Vblock, Ablock = block.get_data()
    Y = Ablock[:,0]

    # Extract polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Vblock_poly = poly.fit_transform(Vblock)

    # Linear regression to luminansce
    model = LinearRegression(fit_intercept=False)
    model.fit(Vblock_poly, Y)

    # Predicted values
    Y_pred = model.predict(Vblock_poly)
    coeffs = model.coef_
    feature_names = poly.get_feature_names_out(['x', 'y', 'z'])

    # Error and log messages
    rmse = root_mean_squared_error(Y, Y_pred) # Base case: threshold met or max degree
    logger.debug(f"🧠 Polynomial Fit Summary (degree {degree})")
    logger.debug(f"📈 RMSE: {rmse:.6f}")
    logger.debug("🔧 Coefficients:")
    for name, coeff in zip(feature_names, coeffs):
        logger.debug(f"  {name:>6s}: {coeff:.6f}")
    return Y_pred, coeffs

def graph_from_fit(block: Block, Y_pred: np.ndarray):
    # Use luminansce from prediction
    Vblock, Ablock = block.get_data()
    Ablock_pred = Ablock.copy()
    Ablock_pred[:,0] = Y_pred
   
    # Create graph using prediction
    approximated_graph = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.7)
    approximated_graph._init_data(block.Vblock, Ablock_pred)
    block.set_data(Vblock, Ablock_pred) # For visualization purposes

    return approximated_graph

def visualize_luminansce_fit(visualizer: Visualizer, block: Block, attribute_graph: AttributeGraph):
    # Block is normalized
    Vblock, Ablock = block.get_data()    
    visualizer.visualize_base(Ablock[:,0],title="Original Y Block Visualization")

    Y_pred, model_coeffs = block_luminansce_fit(block)
    visualizer.visualize_base(Y_pred,title="Predicted Y Block Visualization")

    # Display the Luminansce (Y) visualization
    visualizer.display()

    visualizer.visualize_base(attribute_graph.S, title="Sink Vector Projection")
    visualizer.add_selected_nodes()        
    
    approximated_graph = graph_from_fit(block, Y_pred)
    visualizer(approximated_graph, block)
    visualizer.visualize_base(approximated_graph.S, title= "Sink Vector Projection for Approximated Y")
    visualizer.add_selected_nodes()        
    
    # Display the Sink Vector visualization 
    visualizer.display()
    return Y_pred, model_coeffs, approximated_graph

def evaluate_graphs(graphs_dict: Dict[str, AttributeGraph | StructuralGraph],
                    block: Block,
                    qsteps: List[int],
                    experiment_df: pd.DataFrame) -> None:
    GFT_computer = GFT()

    for graph_kind, graph_obj in graphs_dict.items():
        # Apply GFT and get coefficients
        _, graph_Coeffs = GFT_computer(graph_obj, block)
        Y_Coeffs = graph_Coeffs[:, 0]  # Only luminance component

        # Entropy estimation of Y_Coeffs
        coeff_hist, _ = np.histogram(Y_Coeffs, bins=256, density=True)
        coeff_hist = coeff_hist[coeff_hist > 0]  # remove zeros
        entropy = -np.sum(coeff_hist * np.log2(coeff_hist))

        # Quantize and compute PSNR_Y for each qstep
        psnr_values = []
        for qstep in qsteps:
            q_Y_Coeffs = np.round(Y_Coeffs / qstep) * qstep
            mse = np.mean((Y_Coeffs - q_Y_Coeffs) ** 2)
            psnr_Y = -10 * np.log10(mse / (255 ** 2))
            psnr_values.append(psnr_Y)

        # Save results
        row_data = {
            "Block ID": block.id,
            "Graph Kind": graph_kind,
            "Entropy Y_Coeffs": entropy,
        }
        row_data.update({f"PSNR_Y Q {q}": psnr for q, psnr in zip(qsteps, psnr_values)})

        experiment_df.loc[len(experiment_df)] = row_data
    

point_cloud = PointCloud()
point_cloud(Path("res/longdress_vox10_1051.ply"))
point_cloud.do_block_partitioning(bsize=16)

qsteps = [24,28,32,40,48,56,64]
V = point_cloud.V
A = point_cloud.A
blocks = point_cloud.get_all_blocks()

visualizer = Visualizer()
visualize = False
poly_coeffs = []

PSNR_cols = [f"PSNR_Y Q {qstep}" for qstep in qsteps]
experiment_df = pd.DataFrame(columns=["Block ID",
                                      "Graph Kind",
                                      "Entropy Y_Coeffs",
                                      ] + PSNR_cols)

# Iterate over blocks
for i, block in tqdm(enumerate(blocks), "Polyfit per block"):
    # Original block data
    block._init_data(V, A)
    if block.Vblock.shape[0] == 1:
        continue

    # Graphs withouth predictions
    structural_graph, attribute_graph = initialize_graphs(block)

    if visualize:
        # Normalization process
        visualize_normalization(visualizer, block, structural_graph)

        # Luminansce prediction 
        fit_result = visualize_luminansce_fit(visualizer, block, attribute_graph)
        Y_pred, model_coeffs, approximated_graph = fit_result

    else:
        normalize_block(block)
        Y_pred, model_coeffs = block_luminansce_fit(block)
        approximated_graph = graph_from_fit(block, Y_pred)

    # Re-initialize block data
    block._init_data(V, A)
    graphs_dict = {"Structural Graph": structural_graph,
                   "Attribute Graph": attribute_graph,
                   "Approximated Graph": approximated_graph}
    evaluate_graphs(graphs_dict, block, qsteps, experiment_df)
    poly_coeffs.append(model_coeffs)

    # Cleanup
    structural_graph._del_data()
    attribute_graph._del_data()
    approximated_graph._del_data()
    block._del_data()

poly_coeffs = np.array(poly_coeffs)
# Select best graphs by lowest entropy per block
best_graph_lowest_entropy = experiment_df.loc[
    experiment_df.groupby("Block ID")["Entropy Y_Coeffs"].idxmin()
]

# Descriptive statistics on selected best graphs
desc_stats = best_graph_lowest_entropy.describe().transpose()
logger.info("Descriptive Statistics for Best Graphs by Lowest Entropy:\n%s\n", desc_stats)

# Mean entropy by graph kind for selected best graphs
entropy_by_graph = best_graph_lowest_entropy.groupby("Graph Kind")["Entropy Y_Coeffs"].mean()
logger.info("Mean Entropy by Graph Kind (Best Graphs):\n%s\n", entropy_by_graph)

# PSNR columns
psnr_cols = [col for col in best_graph_lowest_entropy.columns if col.startswith("PSNR_Y Q")]

# Average PSNR by graph kind for selected best graphs
avg_psnr_by_graph = best_graph_lowest_entropy.groupby("Graph Kind")[psnr_cols].mean()
logger.info("Average PSNR per Graph Kind (Best Graphs):\n%s\n", avg_psnr_by_graph)

# Summary table of best graphs
summary_cols = ["Block ID", "Graph Kind", "Entropy Y_Coeffs"] + psnr_cols
summary_table = best_graph_lowest_entropy[summary_cols]
logger.info("Summary of Best Graphs per Block:\n%s\n", summary_table.to_string(index=False))
# # After your main for loop ends:
# labels = list(exp.keys())
# counts = [len(exp[label]) for label in labels]
#
# plt.figure(figsize=(8, 5))
# bars = plt.bar(labels, counts, color=["#4C72B0", "#55A868", "#C44E52"])
# plt.title("Number of Blocks per Explanation Type")
# plt.xlabel("Explanation Type")
# plt.ylabel("Number of Blocks")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Add value labels on top of bars
# for bar, count in zip(bars, counts):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(count), ha='center', va='bottom')
#
# plt.tight_layout()
# plt.show()
# # Plot distribution of selected indices for each quantization step
# def plot_selected_idx_distribution(idx_dist):
#     plt.figure(figsize=(12, 6))
#     for qstep, indices in idx_dist.items():
#         if len(indices) > 0:
#             plt.hist(indices, bins=50, alpha=0.5, label=f"q={qstep}")
#     plt.title("Distribution of Selected Indices per Quantization Step")
#     plt.xlabel("Selected Node Index")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# # 🔔 Call the plot function after processing all blocks
# # plot_selected_idx_distribution(idx_dist)
#
# # After the main loop (which fills spatial_dist[qstep] with Vselected_norm), call:
#
# def plot_histograms_by_axis(spatial_dist):
#     axes = ['X', 'Y', 'Z']
#     fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
#
#     for i, axis in enumerate(axes):
#         for qstep, coords_list in spatial_dist.items():
#             if coords_list:
#                 all_coords = np.vstack(coords_list)
#                 axs[i].hist(all_coords[:, i], bins=30, alpha=0.4, label=f'q={qstep}')
#
#         axs[i].set_title(f'{axis}-axis Distribution (Normalized)')
#         axs[i].set_ylabel("Frequency")
#         axs[i].grid(True)
#
#     axs[-1].set_xlabel("Normalized Position [0, 1]")
#     axs[0].legend()
#     plt.tight_layout()
#     plt.show()
#
# # 🔔 Call it after the loop
# # plot_histograms_by_axis(spatial_dist)
#
# def compute_custom_score(X: np.ndarray,V: np.ndarray, Y_Coeffs: np.ndarray, labels: list[int], centers, lambda_penalty=0.05):
#     total_intra = 0.0
#     for k in range(len(centers)):
#         cluster_points = X[labels == k]
#         if len(cluster_points) > 0:
#             dists = np.linalg.norm(Y_Coeffs - centers[k]* V, axis=1)
    #         total_intra += np.mean(dists)
    # mean_intra = total_intra / len(centers)
    # return -mean_intra + lambda_penalty * len(centers)

# def scatter_all_linear_coefficients(coefficients_list, Y_coeffs, Vblock, feature_names, title="Fitted Coefficients Across Blocks"):
#     x_vals, y_vals, z_vals, bias_vals = [], [], [], []
#     full_coeffs = []
#
#     for coeffs in coefficients_list:
#         fnames = feature_names.tolist()
#         x = coeffs[fnames.index("x")] if "x" in fnames else 0.0
#         y = coeffs[fnames.index("y")] if "y" in fnames else 0.0
#         z = coeffs[fnames.index("z")] if "z" in fnames else 0.0
#         bias = coeffs[fnames.index("1")] if "1" in fnames else 0.0
#
#         x_vals.append(x)
#         y_vals.append(y)
#         z_vals.append(z)
#         bias_vals.append(bias)
#         full_coeffs.append([bias, x, y, z])  # for mean computation
#
#     x_vals, y_vals, z_vals, bias_vals = map(np.array, (x_vals, y_vals, z_vals, bias_vals))
#     X = np.stack([x_vals, y_vals, z_vals], axis=1)
#     full_coeffs = np.array(full_coeffs)
#
#     bias_norm = (bias_vals - bias_vals.min()) / (bias_vals.max() - bias_vals.min() + 1e-8)
#
#     fig = plt.figure(figsize=(16, 6))
#
#     # === Plot 1 ===
#     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
#     sc1 = ax1.scatter(x_vals, y_vals, z_vals, c=bias_norm, cmap="viridis", s=50, edgecolors="k")
#     ax1.set_title(title)
#     ax1.set_xlabel("x coefficient")
#     ax1.set_ylabel("y coefficient")
#     ax1.set_zlabel("z coefficient")
#     plt.colorbar(sc1, ax=ax1, shrink=0.6).set_label("Bias (Normalized)")
#
#     # === Plot 2: Custom metric clustering ===
#     max_clusters = min(10, len(X))
#     best_score = -np.inf
#     best_k = 2
#     best_labels = None
#     best_centers = None
#
#     for k in range(2, max_clusters + 1):
#         kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
#         labels = kmeans.fit_predict(X)
#         centers = kmeans.cluster_centers_
#         score = compute_custom_score(X, Vblock, Y_Coeffs,labels, centers, lambda_penalty=0.2)  # tune lambda if needed
#
#         if score > best_score:
#             best_score = score
#             best_k = k
#             best_labels = labels
#             best_centers = centers
#
#     ax2 = fig.add_subplot(1, 2, 2, projection="3d")
#     sc2 = ax2.scatter(x_vals, y_vals, z_vals, c=best_labels, cmap="tab10", s=50, edgecolors="k")
#     ax2.set_title(f"Custom Clustering (k={best_k}) - Score={best_score:.2f}")
#     ax2.set_xlabel("x coefficient")
#     ax2.set_ylabel("y coefficient")
#     ax2.set_zlabel("z coefficient")
#
#     # === Log cluster means ===
#     logger.debug(f"📊 Cluster Coefficient Means for k = {best_k}")
#     for cid in range(best_k):
#         cluster = full_coeffs[best_labels == cid]
#         mean = np.mean(cluster, axis=0)
#         logger.debug(f"Cluster {cid}: bias={mean[0]:.4f}, x={mean[1]:.4f}, y={mean[2]:.4f}, z={mean[3]:.4f}")
#
#     plt.tight_layout()
#     plt.show()
#
# scatter_all_linear_coefficients(poly_coeffs, feature_names)
