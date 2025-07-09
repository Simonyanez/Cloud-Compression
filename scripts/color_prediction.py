import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import logging
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

def fit_polynomial_function_recursive(
    Vblock: np.ndarray,
    Y: np.ndarray,
    degree: int = 1,
    rmse_threshold: float = 5.0,
    max_degree: int = 1,
):
    """
    Recursively fit a polynomial function Y(x, y, z) with increasing degree
    until RMSE is below a threshold or max_degree is reached.

    Parameters:
    - Vblock: (N, 3) array of spatial coordinates (centered + aligned)
    - Y: (N,) array of luminance values
    - degree: Starting polynomial degree
    - rmse_threshold: Stop if RMSE is below this
    - max_degree: Stop increasing if this degree is reached

    Returns:
    - Y_pred: predicted Y values
    - degree: final polynomial degree used
    - coeffs: final polynomial coefficients
    - rmse: final RMSE
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Vblock_poly = poly.fit_transform(Vblock)

    model = LinearRegression(fit_intercept=False)
    model.fit(Vblock_poly, Y)

    Y_pred = model.predict(Vblock_poly)
    coeffs = model.coef_
    feature_names = poly.get_feature_names_out(['x', 'y', 'z'])

    rmse = root_mean_squared_error(Y, Y_pred)
    # Base case: threshold met or max degree
    if rmse <= rmse_threshold or degree >= max_degree:
        logger.debug(f"🧠 Polynomial Fit Summary (degree {degree})")
        logger.debug(f"📈 RMSE: {rmse:.6f}")
        logger.debug("🔧 Coefficients:")
        for name, coeff in zip(feature_names, coeffs):
            logger.debug(f"  {name:>6s}: {coeff:.6f}")
        return Y_pred, feature_names, coeffs
    else:
        logger.debug(f"At degree {degree}, threshold not met. RMSE: {rmse:.6f}")
        return fit_polynomial_function_recursive(
            Vblock, Y,
            degree=degree + 1,
            rmse_threshold=rmse_threshold,
            max_degree=max_degree
        )


point_cloud = PointCloud()
point_cloud(Path("res/longdress_vox10_1051.ply"))
point_cloud.do_block_partitioning(bsize=16)
qsteps = [24,28,32,40,48,56,64]
V = point_cloud.V
A = point_cloud.A
blocks = point_cloud.get_all_blocks()
visualizer = Visualizer()
degree = 4
visualize = False
poly_coeffs = []
bits_needed = dict(zip(qsteps, [0]*len(qsteps)))
idx_dist = {q: [] for q in qsteps}
spatial_dist = {q: [] for q in qsteps}
mapping = {0:"Structural", 1:"Attribute", 2:"Approximation"}
exp = {"Structural": [], "Attribute": [], "Approximation": []}
for i, block in tqdm(enumerate(blocks), "Polyfit per block"):
    block._init_data(V, A)

    # Center block
    Vblock = block.Vblock  # shape: (N, D)
    if Vblock.shape[0] == 1:
        continue
    Ablock = block.Ablock
    Vblock_centered = center_block(Vblock) 
    # Use AttributeGraph just to get access to graph.S
    sgraph = StructuralGraph(block.id)
    sgraph._init_data(block.Vblock)
    graph = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.7)
    graph._init_data(block.Vblock, block.Ablock)
    visualizer(graph, block)
    # visualizer(sgraph, block)
    # visualizer.set_Vblock(Vblock_centered)
    if visualize:
        visualizer.visualize_block(title="Visualización del Bloque")
        visualizer.visualize_graph(title="Visualización del Grafo")
        visualizer.display()
    Vblock_rotated = rotate_block(Vblock_centered)
    if visualize:
        visualizer.set_Vblock(Vblock_rotated)
        visualizer.visualize_block(title="Visualización del Bloque Normalizado")
    Y_pred, feature_names, coeffs = fit_polynomial_function_recursive(Vblock_rotated, Ablock[:,0], degree=1)
    Ablock_pred = Ablock
    Ablock_pred[:,0] = Y_pred
    # visualizer.set_Ablock(Ablock_pred)
    if visualize:
        visualizer.visualize_base(Ablock[:,0],title="Y Block Visualization")
        visualizer.visualize_base(Y_pred,title="Predicted Y Block Visualization")
        visualizer.display()
        visualizer.visualize_base(graph.S, title="Sink Vector Projection")
        visualizer.add_selected_nodes()        
    graph_pred = AttributeGraph(block.id, sl_weight=1.2, sl_threshold=0.7)
    graph_pred._init_data(block.Vblock,Ablock_pred)
    block._init_auxiliary(Vblock_rotated, Ablock_pred, subidxs=[-1])
    if visualize:
        visualizer(graph_pred, block)
        visualizer.visualize_base(graph_pred.S, title= "Sink Vector Projection for Approximated Y")
        visualizer.add_selected_nodes()        
        visualizer.display()
    block._init_data(V, A)
    GFT_computer = GFT()
    s_result = GFT_computer(sgraph, block)
    a_result = GFT_computer(graph, block)
    p_result = GFT_computer(graph_pred, block)
    if visualize:
        visualizer.visualize_block_coeffs(s_result, title=f"Energy Compaction for Structural - Block Nº {i}")
        visualizer.visualize_block_coeffs(a_result, title=f"Energy Compaction for Attribute - Block Nº {i}")
        visualizer.visualize_block_coeffs(p_result, title=f"Energy Compaction for Approximation - Block Nº {i}")
        visualizer.display()
    _, s_coeffs = s_result
    _, a_coeffs = a_result
    _, p_coeffs = p_result
    sDC = s_coeffs[0,0]
    aDC = a_coeffs[0,0]
    pDC = p_coeffs[0,0]
    best = np.argmax(np.array([sDC, aDC, pDC]))
    if mapping[best] in ["Attribute","Approximation"]:
        poly_coeffs.append(coeffs)
    exp[mapping[best]].append(i)
    sgraph._del_data()
    graph._del_data()
    graph_pred._del_data()
    block._del_data()
# After your main for loop ends:
labels = list(exp.keys())
counts = [len(exp[label]) for label in labels]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts, color=["#4C72B0", "#55A868", "#C44E52"])
plt.title("Number of Blocks per Explanation Type")
plt.xlabel("Explanation Type")
plt.ylabel("Number of Blocks")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar, count in zip(bars, counts):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()
# Plot distribution of selected indices for each quantization step
def plot_selected_idx_distribution(idx_dist):
    plt.figure(figsize=(12, 6))
    for qstep, indices in idx_dist.items():
        if len(indices) > 0:
            plt.hist(indices, bins=50, alpha=0.5, label=f"q={qstep}")
    plt.title("Distribution of Selected Indices per Quantization Step")
    plt.xlabel("Selected Node Index")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 🔔 Call the plot function after processing all blocks
# plot_selected_idx_distribution(idx_dist)

# After the main loop (which fills spatial_dist[qstep] with Vselected_norm), call:

def plot_histograms_by_axis(spatial_dist):
    axes = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for i, axis in enumerate(axes):
        for qstep, coords_list in spatial_dist.items():
            if coords_list:
                all_coords = np.vstack(coords_list)
                axs[i].hist(all_coords[:, i], bins=30, alpha=0.4, label=f'q={qstep}')
        
        axs[i].set_title(f'{axis}-axis Distribution (Normalized)')
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    axs[-1].set_xlabel("Normalized Position [0, 1]")
    axs[0].legend()
    plt.tight_layout()
    plt.show()

# 🔔 Call it after the loop
# plot_histograms_by_axis(spatial_dist)

def compute_custom_score(X, labels, centers, lambda_penalty=0.05):
    total_intra = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            dists = np.linalg.norm(cluster_points - centers[k], axis=1)
            total_intra += np.mean(dists)
    mean_intra = total_intra / len(centers)
    return -mean_intra + lambda_penalty * len(centers)

def scatter_all_linear_coefficients(coefficients_list, feature_names, title="Fitted Coefficients Across Blocks"):
    x_vals, y_vals, z_vals, bias_vals = [], [], [], []
    full_coeffs = []

    for coeffs in coefficients_list:
        fnames = feature_names.tolist()
        x = coeffs[fnames.index("x")] if "x" in fnames else 0.0
        y = coeffs[fnames.index("y")] if "y" in fnames else 0.0
        z = coeffs[fnames.index("z")] if "z" in fnames else 0.0
        bias = coeffs[fnames.index("1")] if "1" in fnames else 0.0

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        bias_vals.append(bias)
        full_coeffs.append([bias, x, y, z])  # for mean computation

    x_vals, y_vals, z_vals, bias_vals = map(np.array, (x_vals, y_vals, z_vals, bias_vals))
    X = np.stack([x_vals, y_vals, z_vals], axis=1)
    full_coeffs = np.array(full_coeffs)

    bias_norm = (bias_vals - bias_vals.min()) / (bias_vals.max() - bias_vals.min() + 1e-8)

    fig = plt.figure(figsize=(16, 6))

    # === Plot 1 ===
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax1.scatter(x_vals, y_vals, z_vals, c=bias_norm, cmap="viridis", s=50, edgecolors="k")
    ax1.set_title(title)
    ax1.set_xlabel("x coefficient")
    ax1.set_ylabel("y coefficient")
    ax1.set_zlabel("z coefficient")
    plt.colorbar(sc1, ax=ax1, shrink=0.6).set_label("Bias (Normalized)")

    # === Plot 2: Custom metric clustering ===
    max_clusters = min(10, len(X))
    best_score = -np.inf
    best_k = 2
    best_labels = None
    best_centers = None

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        score = compute_custom_score(X, labels, centers, lambda_penalty=0.2)  # tune lambda if needed

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centers = centers

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc2 = ax2.scatter(x_vals, y_vals, z_vals, c=best_labels, cmap="tab10", s=50, edgecolors="k")
    ax2.set_title(f"Custom Clustering (k={best_k}) - Score={best_score:.2f}")
    ax2.set_xlabel("x coefficient")
    ax2.set_ylabel("y coefficient")
    ax2.set_zlabel("z coefficient")

    # === Log cluster means ===
    logger.debug(f"📊 Cluster Coefficient Means for k = {best_k}")
    for cid in range(best_k):
        cluster = full_coeffs[best_labels == cid]
        mean = np.mean(cluster, axis=0)
        logger.debug(f"Cluster {cid}: bias={mean[0]:.4f}, x={mean[1]:.4f}, y={mean[2]:.4f}, z={mean[3]:.4f}")

    plt.tight_layout()
    plt.show()

scatter_all_linear_coefficients(poly_coeffs, feature_names)