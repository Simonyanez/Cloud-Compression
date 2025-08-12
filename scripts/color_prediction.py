import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import pandas as pd
import seaborn as sns
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
from pcadc.decider import *

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

def initialize_graphs(block: Block) -> tuple[StructuralGraph, AttributeGraph]:
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
    - RMSE: the root mean squared error for the linear prediction
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
    return Y_pred, coeffs, rmse

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

    Y_pred, model_coeffs, rmse = block_luminansce_fit(block)
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
    return Y_pred, model_coeffs,rmse, approximated_graph

def evaluate_graphs(
    graphs_dict: Dict[str, AttributeGraph | StructuralGraph],
    block: Block,
    qsteps: List[int],
    decider: Decider,
    evaluation_res: List,
) -> None:
    GFT_computer = GFT()

    entropy_by_kind = {}
    coeff_dict = {}

    for graph_kind, graph_obj in graphs_dict.items():
        _, graph_coeffs = GFT_computer(graph_obj, block)
        coeff_dict[graph_kind] = graph_coeffs

        # Entropy of Y channel
        Y_Coeffs = graph_coeffs[:, 0]
        coeff_hist, _ = np.histogram(Y_Coeffs, bins=256, density=True)
        coeff_hist = coeff_hist[coeff_hist > 0]
        entropy = -np.sum(coeff_hist * np.log2(coeff_hist))
        entropy_by_kind[graph_kind] = entropy

        logger.info(f"Block ID: {block.id}")
        logger.info(f"Graph Kind: {graph_kind}")
        logger.info(f"Entropy: {entropy}")
        logger.info(f"PSNR Values:      ")

        # PSNR for each QStep
        psnr_list = []
        for qstep in qsteps:
            q_Y = np.round(Y_Coeffs / qstep) * qstep
            mse = np.mean((Y_Coeffs - q_Y) ** 2)
            psnr = -10 * np.log10(mse / (255.0 ** 2)) if mse != 0 else float('inf')
            logger.info(f"  For Q Step = {qstep} -> {psnr}")


    # RD selection part
    for qstep in qsteps:
        selected_name, selected_coeffs, min_cost = decider(qstep, coeff_dict)

        structural_rd_cost = None
        rd_gain = None
        if "Structural Graph" in coeff_dict:
            structural_rd_cost = decider._RDcost(coeff_dict["Structural Graph"])
            rd_gain = structural_rd_cost - min_cost

        logger.info(
            f"Block {block.id} | QStep {qstep} | Best: {selected_name} | "
            f"RD Cost: {min_cost:.6f} | Gain: {rd_gain:.6f}"
        )

        evaluation_res.append({
            "Block ID": block.id,
            "Q Step": qstep,
            "Best Graph": selected_name,
            "RD Cost": min_cost,
            "RD Gain": rd_gain,
                            })

def analyze_rdcost_by_qstep(experiment_df: pd.DataFrame):
    """
    Analyze RD cost performance by QStep.
    """
    df = experiment_df.copy()

    if df.empty or "QStep" not in df.columns:
        logger.warning("No RD cost data found for qstep analysis.")
        return

    # Group by QStep and Best Graph Kind
    grouped = df.groupby(["QStep", "Best Graph Kind"]).agg(
        Count=("RD Cost", "count"),
        Mean_RD_Cost=("RD Cost", "mean"),
        Mean_RD_Gain=("RD Gain", "mean")
    ).sort_index()

    logger.info("Best RD Graphs per QStep:\n%s", grouped.to_string())

    # Best RD Graph per QStep
    best_per_qstep = (
        grouped.reset_index()
        .sort_values(["QStep", "Mean_RD_Cost"])
        .groupby("QStep")
        .first()
    )
    logger.info("Best Performing Graph by RD Cost per QStep:\n%s", best_per_qstep.to_string(index=True))

    # === PLOTS ===
    import seaborn as sns
    sns.set(style="whitegrid")

    # RD Cost vs QStep
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=grouped.reset_index(),
        x="QStep", y="Mean_RD_Cost", hue="Best Graph Kind", marker="o"
    )
    plt.title("Mean RD Cost per QStep")
    plt.xlabel("QStep")
    plt.ylabel("Mean RD Cost")
    plt.legend(title="Best Graph Kind")
    plt.tight_layout()
    plt.show()

    # RD Gain vs QStep
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=grouped.reset_index(),
        x="QStep", y="Mean_RD_Gain", hue="Best Graph Kind", marker="o"
    )
    plt.title("Mean RD Gain vs Structural per QStep")
    plt.xlabel("QStep")
    plt.ylabel("Mean RD Gain")
    plt.legend(title="Best Graph Kind")
    plt.tight_layout()
    plt.show()

    # Selection count per graph
    plt.figure(figsize=(10, 4))
    sns.barplot(
        data=grouped.reset_index(),
        x="Best Graph Kind", y="Count", hue="QStep"
    )
    plt.title("Selection Count of Best Graphs per QStep")
    plt.xlabel("Graph Kind")
    plt.ylabel("Selection Count")
    plt.legend(title="QStep")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_entropy_psnr(experiment_df: pd.DataFrame):
    """
    Analyze entropy and PSNR by Graph Kind.
    """
    df = experiment_df.copy()

    if df.empty:
        logger.warning("No data available to analyze.")
        return

    entropy_df = df[df["Entropy Y_Coeffs"].notna()].copy()
    if entropy_df.empty:
        logger.warning("No entropy data to analyze.")
        return

    # Group by Best Graph Kind
    avg_entropy = entropy_df.groupby("Best Graph Kind")["Entropy Y_Coeffs"].mean()
    psnr_cols = [col for col in entropy_df.columns if col.startswith("PSNR_Y Q")]
    avg_psnr = entropy_df.groupby("Best Graph Kind")[psnr_cols].mean()

    logger.info("Average Entropy per Best Graph Kind:\n%s", avg_entropy.to_string())
    logger.info("Average PSNR per Best Graph Kind:\n%s", avg_psnr.to_string())

    # Plot entropy
    if not avg_entropy.empty:
        avg_entropy.plot(kind='bar', title='Average Entropy per Best Graph Kind')
        plt.ylabel("Entropy (bits)")
        plt.tight_layout()
        plt.show()

    # Plot PSNR per QStep
    if not avg_psnr.empty:
        avg_psnr.T.plot(title="Average PSNR per Best Graph Kind")
        plt.ylabel("PSNR (dB)")
        plt.xlabel("QStep")
        plt.tight_layout()
        plt.show()

def run_processing_analysis(processing_df: pd.DataFrame):
    if processing_df.empty:
        logger.warning("Processing DataFrame is empty.")
        return

    logger.info("=== Processing DataFrame Summary ===")
    logger.info(processing_df.describe(include='all').to_string())

    # RMSE histogram
    plt.figure(figsize=(8,4))
    sns.histplot(processing_df['RMSE'], bins=40, kde=True)
    plt.title('Distribution of RMSE per Block')
    plt.xlabel('RMSE')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Luminance STD histogram
    plt.figure(figsize=(8,4))
    sns.histplot(processing_df['Luminansce STD'], bins=40, kde=True)
    plt.title('Distribution of Luminance STD per Block')
    plt.xlabel('Luminance STD')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Coefficient norms or specific coeff histograms
    coeffs = np.vstack(processing_df['Fit Coeffs'])
    coeff_names = [f'Coeff_{i}' for i in range(coeffs.shape[1])]
    coeffs_df = pd.DataFrame(coeffs, columns=coeff_names)

    plt.figure(figsize=(12,6))
    sns.boxplot(data=coeffs_df)
    plt.title("Distribution of Polynomial Fit Coefficients")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    run_coeffs_logs(coeffs)  # Your existing logging function

def run_evaluation_analysis(evaluation_df: pd.DataFrame):
    if evaluation_df.empty:
        logger.warning("Evaluation DataFrame is empty.")
        return

    logger.info("=== Evaluation DataFrame Summary ===")
    logger.info(evaluation_df.describe(include='all').to_string())

    # Ensure column naming consistency
    if 'Q Step' in evaluation_df.columns:
        evaluation_df = evaluation_df.rename(columns={'Q Step': 'QStep'})
    if 'Best Graph' in evaluation_df.columns:
        evaluation_df = evaluation_df.rename(columns={'Best Graph': 'Best Graph Kind'})

    analyze_rdcost_by_qstep(evaluation_df)


def run_coeffs_logs(poly_coeffs: np.ndarray) -> None:
    means = poly_coeffs.mean(axis=0)
    stds = poly_coeffs.std(axis=0)
    mins = poly_coeffs.min(axis=0)
    maxs = poly_coeffs.max(axis=0)

    logger.info("Luminansce Fit Coefficients Stats: ")
    logger.info(f"Means: {means}")
    logger.info(f"Stds: {stds}")
    logger.info(f"Mins: {mins}")
    logger.info(f"Maxs: {maxs}")
    logger.info(f"Correlation matrix:\n{np.corrcoef(poly_coeffs.T)}")


def run_slopes_visualization(slope_matrix: np.ndarray) -> None:
    """
    Plots the distribution and correlation of the slope (coefficient) variables.
    
    Parameters:
        slope_matrix (np.ndarray): Array of shape (N, 3), where each row contains the slope coefficients
                                   (e.g., beta_1, beta_2, beta_3) for a single point cloud block.
    """
    if slope_matrix.shape[1] != 3:
        raise ValueError("Expected slope_matrix with shape (N, 3), representing (β₁, β₂, β₃).")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution Plot
    labels = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$']
    for i in range(3):
        sns.kdeplot(slope_matrix[:, i], ax=axes[0], label=labels[i], fill=True, linewidth=2)
    axes[0].set_title("Distribution of Slope Coefficients")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Correlation Heatmap
    corr = np.corrcoef(slope_matrix.T)
    sns.heatmap(corr, annot=True, cmap="coolwarm", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Correlation Between Slope Coefficients")

    plt.tight_layout()
    plt.show()

def normalize_coefficients(
    slope_matrix: np.ndarray, 
    plot: bool = True,
    method: str = "l2"
) -> np.ndarray:
    """
    Normalizes each coefficient vector in the slope matrix and logs statistics.

    Args:
        slope_matrix (np.ndarray): Matrix of shape (N, 3) with slope coefficients.
        plot (bool): Whether to show a diagnostic plot of norms before/after.
        method (str): Normalization method: 'l2' or 'max'.

    Returns:
        np.ndarray: Normalized slope matrix.
    """
    if method not in ("l2", "max"):
        raise ValueError("method must be 'l2' or 'max'")

    norms = np.linalg.norm(slope_matrix, axis=1) if method == "l2" else np.max(np.abs(slope_matrix), axis=1)
    zero_norms = norms == 0
    if np.any(zero_norms):
        logger.warning(f"{np.sum(zero_norms)} rows have zero norm and will be left unchanged.")
        norms[zero_norms] = 1.0  # prevent division by zero

    normalized = slope_matrix / norms[:, np.newaxis]

    # Log statistics
    logger.info(f"Original norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
    new_norms = np.linalg.norm(normalized, axis=1) if method == "l2" else np.max(np.abs(normalized), axis=1)
    logger.info(f"New norms ({method}) - min: {new_norms.min():.4f}, max: {new_norms.max():.4f}, mean: {new_norms.mean():.4f}")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(norms, bins=30)
        ax[0].set_title("Original Norms")
        ax[1].hist(new_norms, bins=30)
        ax[1].set_title(f"Normalized Norms ({method})")
        plt.suptitle("Coefficient Vector Norms")
        plt.tight_layout()
        plt.show()

    return normalized

def fixed_centroid_kmeans(X: np.ndarray, n_clusters: int,
                          fixed_center: np.ndarray = np.array([0, 0, 0]),
                          max_iter: int = 300, tol: float = 1e-4, verbose=False):
    assert n_clusters > 1, "At least 2 clusters are needed (one fixed, one+ dynamic)"
    
    # Initialize centroids: one is fixed, the others are randomly sampled from the data
    rng = np.random.default_rng(seed=42)
    other_centers = rng.choice(X, size=n_clusters - 1, replace=False)
    centers = np.vstack([fixed_center, other_centers])

    for it in range(max_iter):
        # Assign each point to the closest center
        labels = pairwise_distances_argmin_min(X, centers)[0]
        
        new_centers = [fixed_center]  # fixed center remains unchanged
        
        for k in range(1, n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                new_centers.append(members.mean(axis=0))
            else:
                # re-initialize dead centroid to a random point
                new_centers.append(rng.choice(X))
        
        new_centers = np.vstack(new_centers)
        shift = np.linalg.norm(centers - new_centers)
        if verbose:
            print(f"Iteration {it}, centroid shift: {shift}")
        if shift < tol:
            break
        centers = np.array(new_centers)
    
    logger.info(f"Custom Kmeans Clustering Results: \n {centers}")

    return centers, labels

from sklearn.metrics import r2_score


def plot_cluster_centers_3d(
    centers: np.ndarray,
    labels: np.ndarray,
    betas: np.ndarray,
    point_labels=True,
    point_color='red',
    title='Cluster Centers in 3D Beta Space'
):
    """
    Plots cluster centers in 3D space with axes beta_1, beta_2, beta_3.
    Logs membership statistics and a representation metric (R²).

    Parameters:
    - centers: numpy array of shape (K,3) with cluster centers
    - labels: numpy array of shape (N,) with cluster assignment for each point
    - betas: numpy array of shape (N,3) with the points' beta values
    - point_labels: bool, if True labels each point with C0, C1, ...
    - point_color: color string for the points
    - title: title of the plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot cluster centers
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color=point_color, s=50)

    for i, (x, y, z) in enumerate(centers):
        num_members = np.sum(labels == i)
        ax.text(x, y, z, f'C{i} ({num_members})', fontsize=10)

        cluster_points = betas[labels == i]
        if cluster_points.size > 0:
            mean_vals = np.mean(cluster_points, axis=0)
            std_vals = np.std(cluster_points, axis=0)
            min_vals = np.min(cluster_points, axis=0)
            max_vals = np.max(cluster_points, axis=0)

            # R² goodness-of-fit between center and members
            repeated_center = np.tile(centers[i], (cluster_points.shape[0], 1))
            r2 = r2_score(cluster_points, repeated_center)

            logger.info(f"Cluster {i} statistics:")
            logger.info(f"  Members: {num_members}")
            logger.info(f"  Centroid: {centers[i]}")
            logger.info(f"  Mean β:  {mean_vals}")
            logger.info(f"  Std β:   {std_vals}")
            logger.info(f"  Min β:   {min_vals}")
            logger.info(f"  Max β:   {max_vals}")
            logger.info(f"  R² fit:  {r2:.4f}")
            logger.info("-" * 40)

    ax.set_xlabel('beta_1 (x)')
    ax.set_ylabel('beta_2 (y)')
    ax.set_zlabel('beta_3 (z)')
    ax.set_title(title)

    plt.show()

def compute_luminansce_estimate(block: Block, poly_coeff: np.ndarray) -> tuple[StructuralGraph, AttributeGraph]:
    Vblock, Ablock = block.get_data()
    Y_estimate = Vblock @ poly_coeff
    bias = Ablock[:,0].mean() - Y_estimate.mean()
    Y_estimate_biased = Y_estimate + bias
    logger.info(f"RMSE of estimated luminansce {root_mean_squared_error(Ablock[:,0], Y_estimate_biased)}")
    Ablock_estimate = Ablock.copy()
    Ablock_estimate[:,0] = Y_estimate
    block.set_data(Vblock, Ablock_estimate)
    structural_graph, estimated_graph = initialize_graphs(block)
    return structural_graph, estimated_graph

def compute_custom_score(X: np.ndarray,V: np.ndarray, Y_Coeffs: np.ndarray, 
                         labels: list[int], centers, lambda_penalty=0.05):
    total_intra = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            dists = np.linalg.norm(Y_Coeffs - centers[k]* V, axis=1)
            total_intra += np.mean(dists)
    mean_intra = total_intra / len(centers)
    return -mean_intra + lambda_penalty * len(centers)

if __name__ == "__main__":
    rewrite_processing_df = True
    rewrite_cluster_df = False


    point_cloud = PointCloud()
    point_cloud(Path("res/longdress_vox10_1051.ply"))
    point_cloud.do_block_partitioning(bsize=8)

    qsteps = [24,28,32,40,48,56,64]
    V = point_cloud.V
    A = point_cloud.A
    blocks = point_cloud.get_all_blocks()

    visualizer = Visualizer()
    visualize = False

    decider = Decider(mode="0")

    experiment_path = Path.cwd() / "tmp/color_prediction"
    experiment_path.mkdir(exist_ok=True)
    processing_path = experiment_path / "processing_df.parquet"
    evaluation_path = experiment_path / "evaluation_df.parquet"
    processing_res = []
    evaluation_res = []

    # --- Process each block ---
    for i, block in tqdm(enumerate(blocks), "Polyfit per block"):
        if processing_path.exists() and not rewrite_processing_df:
            processing_df = pd.read_parquet(processing_path)
            evaluation_df = pd.read_parquet(evaluation_path)
            break

        block._init_data(V, A)
        if block.Vblock.shape[0] == 1:
            continue

        # Graph construction
        structural_graph, attribute_graph = initialize_graphs(block)

        if visualize:
            visualize_normalization(visualizer, block, structural_graph)
            Y_pred, model_coeffs, rmse, approximated_graph = visualize_luminansce_fit(visualizer, block, attribute_graph)
        else:
            normalize_block(block)
            Y_pred, model_coeffs, rmse = block_luminansce_fit(block)
            approximated_graph = graph_from_fit(block, Y_pred)

        # Restart block original data
        block._init_data(V, A)
        
        # Initiliaze dict for comparison
        graphs_dict = {
            "Structural Graph": structural_graph,
            "Attribute Graph": attribute_graph,
            "Approximated Graph": approximated_graph
        }
        evaluate_graphs(graphs_dict, block, qsteps, decider, evaluation_res)

        _, Ablock = block.get_data()
        Y_std = Ablock[:,0].std()
        processing_res.append({"Block ID": block.id,
                               "Number of points": Ablock.shape[0],
                               "RMSE": rmse,
                               "Fit Coeffs": model_coeffs,
                               "Luminansce STD": Y_std,
        })

        # Cleanup
        structural_graph._del_data()
        attribute_graph._del_data()
        approximated_graph._del_data()
        block._del_data()

    if not processing_path.exists() or rewrite_processing_df:
        processing_df = pd.DataFrame(data=processing_res)
        processing_df.to_parquet(processing_path)
        evaluation_df = pd.DataFrame(data=evaluation_res)
        evaluation_df.to_parquet(evaluation_path)

    # Run summary
    run_processing_analysis(processing_df)
    run_evaluation_analysis(evaluation_df)

    # # --- Cluster evaluation ---
    # slope_matrix = np.array(poly_coeffs)[:,1:]  # skip bias
    # run_coeffs_logs(poly_coeffs)
    # run_slopes_visualization(slope_matrix)
    # slope_matrix = normalize_coefficients(slope_matrix)

    # n_clusters = 16
    # centers, labels = fixed_centroid_kmeans(slope_matrix, n_clusters)
    # plot_cluster_centers_3d(centers, labels, slope_matrix)
    #
    # # Cluster fit
    # clusterfit_df = pd.DataFrame(columns=[
    #     "Block ID",
    #     "Graph Kind",
    #     "Entropy Y_Coeffs",
    #     "Best Graph Kind",
    #     "RD Cost",
    #     "RD Gain",
    #     "QStep",
    # ] + PSNR_cols)
    # clusterfit_path = experiment_path / "clusterfit_df.csv"
    # j = 0
    #
    # for i, block in tqdm(enumerate(blocks), "Best cluster per block"):
    #     if clusterfit_path.exists() and not rewrite_cluster_df:
    #         clusterfit_df = pd.read_csv(clusterfit_path)
    #         break
    #
    #     block._init_data(V, A)
    #     if block.Vblock.shape[0] == 1:
    #         continue
    #
    #     block_poly_coeff = centers[labels[j]]
    #     structural_graph, estimated_graph = compute_luminansce_estimate(block, block_poly_coeff)
    #     block._init_data(V, A)
    #
    #     graphs_dict = {
    #         "Structural Graph": structural_graph,
    #         "Estimated Graph": estimated_graph
    #     }
    #
    #     evaluate_graphs(graphs_dict, block, qsteps, decider, clusterfit_df)
    #
    #     structural_graph._del_data()
    #     estimated_graph._del_data()
    #     block._del_data()
    #     j += 1
    #
    # if not clusterfit_path.exists() or rewrite_coeffs_df:
    #     clusterfit_df.to_csv(clusterfit_path, index=False)
    #
    # run_experiment_logs(clusterfit_df)
    #
