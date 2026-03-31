import sys
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from pcadc.pointcloud import PointCloud, PointCloudMetadata, MortonBlockPartition, Sampler
from pcadc.color import Colourist, Approximator
from pcadc.parameters import load_experiment_config
from pcadc.rd_cluster.main import run_rd_clustering
from pcadc.rd_cluster.clusterer import RDClusterer
from pcadc.rd_cluster.gft_cache import InMemoryCacheStrategy
from pcadc.transforms import GFTStrategyWraper
from pcadc.rd_cluster.optimizer import SlopeOptimizer
from pcadc.rd_cluster.convergence import ConvergenceChecker
from pcadc.decider import Decider
from pcadc.rd_cluster.states import RDClusterState, ClusteringHistory

class InstrumentedRDClusterer(RDClusterer):
    def _calculate_rmse(self, block, slope):
        """Calculate RMSE of a block's luminance against a specific slope."""
        V_norm = Approximator()._spatial_norm(block.Vblock)
        Y_true = block.Ablock[:, 0]
        Y_pred = V_norm @ slope.T
        return np.sqrt(np.mean((Y_true - Y_pred)**2))

    def _assignment_step_with_logging(self, blocks, state, vertices, attributes):
        new_labels = np.zeros(len(blocks), dtype=int)
        slopes = state.slopes
        self.decider._set_vars(state.qstep_value)

        total_rd_cost = 0
        total_real_rmse = 0
        
        for i, block in enumerate(blocks):
            block.init_data(vertices, attributes)
            costs = []
            rmses = []
            
            for k, slope in enumerate(slopes):
                # Calculate RD Cost on REAL data
                coeffs = self.gft_cache.get_coeffs(block.block_id)
                if k != 0:
                    coeffs = self._compute_adaptive_gft(block, slope, k)
                rd_cost, _, _ = self.decider._RDcost(coeffs)
                costs.append(rd_cost)
                
                # Calculate RMSE for monitoring
                rmse = self._calculate_rmse(block, slope)
                rmses.append(rmse)
            
            costs = np.array(costs)
            # Use Annealing to choose (with cooling temp)
            chosen_cluster = self.annealing_scheduler.choose(costs, len(slopes))

            new_labels[i] = chosen_cluster
            total_rd_cost += costs[chosen_cluster]
            total_real_rmse += rmses[chosen_cluster]
            block.clear_data()
            
        print(f"  Avg Real RMSE: {total_real_rmse/len(blocks):.4f}")
        print(f"  Total RD Cost: {total_rd_cost:.4f}")
            
        return new_labels, total_rd_cost, None, None, None

    def _fit_full(self, blocks, vertices, attributes):
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        self._precompute_structural_gfts(blocks, vertices, attributes)
        
        state = self._initialize_state(blocks, vertices, attributes)
        print(f"\n[INIT] {len(blocks)} blocks, {self.num_clusters} clusters.")
        
        prev_labels = state.labels.copy()
        history = ClusteringHistory()
        history.add_state(state)
        
        history_jumps = []
        
        for iteration in range(self.convergence_checker.max_iterations):
            print(f"\n--- Iteration {iteration} (Temp: {self.annealing_scheduler.temperature:.4f}) ---")
            
            # Assignment Step
            new_labels, total_cost, _, _, _ = self._assignment_step_with_logging(blocks, state, vertices, attributes)
            
            # TRACK JUMPS
            jumps = np.sum(new_labels != prev_labels)
            jump_percentage = (jumps / len(blocks)) * 100
            history_jumps.append(jumps)
            
            print(f"  Total Cost: {total_cost:.4f}")
            print(f"  Block Jumps: {jumps} ({jump_percentage:.2f}%)")
            
            # Recalculate Slopes
            new_slopes = self.slope_optimizer.recalculate_slopes(
                blocks, new_labels, vertices, attributes, self.num_clusters, state.slopes
            )
            
            # Update State
            state = RDClusterState(
                labels=new_labels,
                slopes=new_slopes,
                qstep_value=state.qstep_value,
                lambda_step=state.lambda_step,
                iteration=iteration + 1,
                total_cost=total_cost
            )
            history.add_state(state)
            
            prev_labels = new_labels.copy()
            self.annealing_scheduler.cool_down()
            
            if history.get_cost_stable(num_of_iters=3, threshold=0.001) and iteration > 5:
                print("  COST STABLE - Stopping.")
                break
                
        return state, history_jumps

def main():
    # 1. Load Config
    config_path = Path("config/base_config.yaml")
    params = load_experiment_config(config_path)
    
    # 2. Setup Point Cloud
    colourist = Colourist()
    pc = PointCloud.from_file(params.sequential_params.point_cloud_path, "ply", params.pointcloud)
    pc.transform_attributes(colourist._RGBtoYUV)
    
    # 3. Partition and Sample
    partitioner = MortonBlockPartition()
    _, all_blocks = partitioner.partition(pc, bsize=params.sequential_params.block_size)
    
    # Use 10% sample
    sample_ratio = 0.1 
    sampler = Sampler(ratio=sample_ratio, n_strata=5)
    blocks = sampler(pc.V, pc.A, all_blocks)
    print(f"Using {len(blocks)} blocks for RD clustering test.")

    # 4. Initialize Instrumented Clusterer
    gft_computer = GFTStrategyWraper()
    slope_optimizer = SlopeOptimizer(learning_rate=params.clustering_params.learning_rate)
    convergence_checker = ConvergenceChecker(
        sequential_parameters=params.sequential_params,
        clustering_parameters=params.clustering_params,
        decider=Decider(mode=params.sequential_params.decider_mode,
                        lagrange_proportional=params.sequential_params.lagrange_proportional)
    )
    
    # 32 clusters
    params.clustering_params.number_of_clusters = 32
    params.clustering_params.max_iterations = 15
    params.clustering_params.cooling_rate = 0.8
    params.clustering_params.initial_temperature = 1.0
    
    clusterer = InstrumentedRDClusterer(
        sequential_parameters=params.sequential_params,
        clusterer_parameters=params.clustering_params,
        decider=Decider(mode=params.sequential_params.decider_mode,
                        lagrange_proportional=params.sequential_params.lagrange_proportional),
        gft_cache=InMemoryCacheStrategy(),
        gft_computer=gft_computer,
        slope_optimizer=slope_optimizer,
        convergence_checker=convergence_checker,
        temp_folder=params.metadata.temp_folder,
        use_two_stage=False
    )

    # 5. Run and Monitor
    final_state, jump_history = clusterer._fit_full(blocks, pc.V, pc.A)
    
    print("\nSummary:")
    print(f"Jump History: {jump_history}")

if __name__ == "__main__":
    main()
