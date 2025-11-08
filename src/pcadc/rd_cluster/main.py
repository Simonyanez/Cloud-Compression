from pathlib import Path
from src.pcadc.rd_cluster.gft_cache import InMemoryCacheStrategy
from src.pcadc.rd_cluster.optimizer import SlopeOptimizer
from src.pcadc.rd_cluster.convergence import ConvergenceChecker
from src.pcadc.rd_cluster.clusterer import RDClusterer
from src.pcadc.blocks import Block
from src.pcadc.parameters import ExperimentConfig
from src.pcadc.decider import Decider
from src.pcadc.transforms import GFTStrategyWraper
from typing import List

import numpy as np


def run_rd_clustering(blocks: List[Block], vertices: np.ndarray, attributes: np.ndarray, experiment_params: ExperimentConfig):
    
    decider = Decider(experiment_params.sequential_params.decider_mode, experiment_params.sequential_params.lagrange_proportional)
    gft_computer = GFTStrategyWraper()
    # Create components
    cache = InMemoryCacheStrategy()
    slope_optimizer = SlopeOptimizer(add_intercept=True)
    convergence_checker = ConvergenceChecker(
        experiment_params.sequential_params,
        experiment_params.clustering_params,
        decider)
    # TODO: Make a training set for the transforms
    # training_selector = TrainingSetSelector(
    #     selection_ratio=config['training_ratio']
    # )
    
    clusterer = RDClusterer(
        sequential_parameters=experiment_params.sequential_params,
        clusterer_parameters=experiment_params.clustering_params,
        decider=decider,
        gft_cache=cache,
        gft_computer=gft_computer,
        slope_optimizer=slope_optimizer,
        convergence_checker=convergence_checker,
        training_selector=None,
        use_two_stage=False
        # training_selector=training_selector,
        # use_two_stage=config['use_two_stage']
    )
    
    final_state, clustering_history = clusterer.fit(blocks, vertices, attributes)
    print(f"Final history {clustering_history}")
    cache.cleanup()
    
    return final_state, clustering_history


if __name__ == "__main__":
    from pcadc.main import Researcher
    from pcadc.parameters import load_experiment_config

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


