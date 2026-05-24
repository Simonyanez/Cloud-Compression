from pcadc.rd_cluster.gft_cache import InMemoryCacheStrategy
from pcadc.rd_cluster.optimizer import SlopeOptimizer
from pcadc.rd_cluster.convergence import ConvergenceChecker
from pcadc.rd_cluster.clusterer import RDClusterer
from pcadc.blocks import Block
from pcadc.parameters import ExperimentConfig
from pcadc.decider import Decider
from pcadc.transforms import GFTStrategyWraper
from typing import List
from pathlib import Path

import numpy as np
import logging
import os

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "rd_cluster_main.log")

logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_rd_clustering(blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray, params: ExperimentConfig):
    
    decider = Decider(params.sequential_params.decider_mode, params.sequential_params.lagrange_proportional)
    gft_computer = GFTStrategyWraper()
    
    # Instantiate SlopeOptimizer with learning rate
    slope_optimizer = SlopeOptimizer()
    
    # Instantiate ConvergenceChecker with clustering_parameters
    convergence_checker = ConvergenceChecker(
        sequential_parameters=params.sequential_params,
        clustering_parameters=params.clustering_params,
        decider=Decider(mode=params.sequential_params.decider_mode,
                        lagrange_proportional=params.sequential_params.lagrange_proportional)
    )
    
    clusterer = RDClusterer(
        sequential_parameters=params.sequential_params,
        clusterer_parameters=params.clustering_params,
        decider=Decider(mode=params.sequential_params.decider_mode,
                        lagrange_proportional=params.sequential_params.lagrange_proportional),
        gft_cache=InMemoryCacheStrategy(),
        gft_computer=gft_computer,
        slope_optimizer=slope_optimizer,
        convergence_checker=convergence_checker,
        temp_folder=params.metadata.temp_folder,
        use_two_stage=False # Set use_two_stage to False
    )
    
    final_state, clustering_history = clusterer.fit(blocks, vertices, attributes)
    print(f"Final history {clustering_history}")
    logger.info(f"Final State: \n {final_state}")
    # cache.cleanup()
    
    return final_state, clustering_history


if __name__ == "__main__":
    import logging
    import os
    from pcadc.main import Researcher
    from pcadc.parameters import load_experiment_config

    # Centralized logging configuration
    # This will catch logs from all modules and write them to a single file.
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "debug.log")

    logging.basicConfig(
        filename=log_file_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    params = load_experiment_config(Path("config/config.yaml"))
    researcher = Researcher()
    researcher._init_experiment(params)
    indexes, blocks = researcher._block_partitioning()
    vertices = researcher.V
    attributes = researcher.A
    final_state, clustering_history = run_rd_clustering(blocks, vertices, attributes, params)
    print(final_state)

    
    
    # TODO: Load your data
    # blocks, vertices, attributes = load_data()
    # final_state = run_rd_clustering(blocks, vertices, attributes, config)


