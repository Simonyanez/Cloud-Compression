import sys
import os
from pathlib import Path

# Add src to path using absolute resolution
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import json
import numpy as np
from argparse import ArgumentParser
from pcadc.parameters import load_experiment_config
from pcadc.pointcloud import PointCloud, MortonBlockPartition, Sampler
from pcadc.color import Colourist
from pcadc.rd_cluster.main import run_rd_clustering
from pcadc.rd_cluster.clusterer import RDClusterer
from joblib import Parallel, delayed
from tqdm import tqdm

# --- WORKER FUNCTION ---
def _evaluate_all_clusters_worker(i, block, vertices, attributes, slopes, 
                                  slw_vals, slp_vals, qstep_value, structural_coeffs, 
                                  decider_mode, lagrange_proportional):
    """
    Computes RD costs for ALL clusters. 
    Index 0 is always structural.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Imports for worker context
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from pcadc.decider import Decider
    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator

    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(qstep_value)
    gft_computer = GFTStrategyWraper()
    block.init_data(vertices, attributes)
    
    costs, rates, dists = [], [], []
    
    for k, slope in enumerate(slopes):
        if k == 0:
            # Cluster 0 is the Structural Baseline
            c, r, d = decider._RDcost(structural_coeffs)
        else:
            # Cluster k > 0 are Adaptive Clusters
            slw, slp = slw_vals[k], slp_vals[k]
            s_graph = StructuralGraph(block.metadata)
            s_graph.set_data(block.Vblock)
            a_graph = AttributeGraph(s_graph, slope, k, slp, slw)
            
            V_rot = Approximator()._spatial_norm(block.Vblock)
            A_app = block.Ablock.copy()
            A_app[:, 0] = V_rot @ slope.T
            a_graph.set_data(block.Vblock, A_app)
            
            _, coeffs = gft_computer(block, a_graph)
            c, r, d = decider._RDcost(coeffs)
            
        costs.append(c)
        rates.append(r)
        dists.append(d)
    
    block.clear_data()
    return costs, rates, dists

# --- MONKEYPATCH ---
def patched_assignment_step(self, blocks, state, vertices, attributes, iteration):
    num_blocks = len(blocks)
    num_c = len(state.slopes) # num_clusters
    new_labels = np.zeros(num_blocks, dtype=int)
    self.decider._set_vars(state.qstep_value)
    
    # 1. Parallel evaluation of all candidates
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_all_clusters_worker)(
            i, block, vertices, attributes, state.slopes, 
            state.self_loop_weights, state.self_loop_percentages, 
            state.qstep_value, self.gft_cache.get_coeffs(block.block_id),
            self.decider.mode, self.decider.lagrange_proportional
        ) for i, block in tqdm(enumerate(blocks), total=num_blocks, desc=f"Cost Matrix (Iter {iteration})")
    )

    cost_matrix = np.zeros((num_blocks, num_c))
    rate_matrix = np.zeros((num_blocks, num_c))
    dist_matrix = np.zeros((num_blocks, num_c))

    for i, (costs, rates, dists) in enumerate(results):
        cost_matrix[i, :] = costs
        rate_matrix[i, :] = rates
        dist_matrix[i, :] = dists

    # 2. Sequential Pass with Beta Penalty (Spatial Smoothness)
    beta = getattr(self.sequential_parameters, 'beta', 0.0)
    new_labels[0] = np.argmin(cost_matrix[0, :])
    for i in range(1, num_blocks):
        adj_costs = cost_matrix[i, :].copy()
        mask = np.ones(num_c, dtype=bool)
        mask[new_labels[i-1]] = False
        adj_costs[mask] += beta
        new_labels[i] = np.argmin(adj_costs)

    # 3. Aggregate Final Results
    total_cost = 0
    all_rates, all_distortions, all_gains = np.zeros(num_blocks), np.zeros(num_blocks), np.zeros(num_blocks)
    for i in range(num_blocks):
        l = new_labels[i]
        total_cost += cost_matrix[i, l]
        all_rates[i] = rate_matrix[i, l]
        all_distortions[i] = dist_matrix[i, l]
        all_gains[i] = cost_matrix[i, 0] - cost_matrix[i, l]

    return new_labels, total_cost, all_rates, all_distortions, all_gains

RDClusterer._assignment_step = patched_assignment_step

# --- MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--q_step", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--lambda_prop", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=500.0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_code = f"B{args.block_size}_C{args.clusters}_L{args.lambda_prop}_Beta{args.beta}"
    
    if args.skip_existing and (out_dir / f"result_{experiment_code}.json").exists():
        print(f"[SKIP] {experiment_code} already exists.")
        return

    params = load_experiment_config(Path(args.config))
    params.clustering_params.max_iterations = args.max_iters
    params.clustering_params.number_of_clusters = args.clusters
    params.sequential_params.block_size = args.block_size
    params.sequential_params.sample_percentage = args.sample_rate
    params.sequential_params.quantization_steps = [args.q_step]
    params.sequential_params.lagrange_proportional = args.lambda_prop
    setattr(params.sequential_params, 'beta', args.beta)

    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    _, blocks = MortonBlockPartition().partition(pc, bsize=args.block_size)
    sampled_blocks = Sampler(params.sequential_params.sample_percentage, n_strata=5, seed=42)(pc.V, pc.A, blocks)
    
    final_state, history = run_rd_clustering(sampled_blocks, pc.V, pc.A, params)
    
    with open(out_dir / f"result_{experiment_code}.json", "w") as f:
        json.dump({
            "experiment_code": experiment_code,
            "active_clusters": len(np.unique(final_state.labels)),
            "final_cost": final_state.total_cost,
            "final_slopes": final_state.slopes.tolist(),
            "final_slw": final_state.self_loop_weights.tolist(),
            "final_slp": final_state.self_loop_percentages.tolist(),
            "iterations": len(history.states),
            "lambda_prop": args.lambda_prop,
            "beta": args.beta
        }, f, indent=4)

if __name__ == "__main__":
    main()
