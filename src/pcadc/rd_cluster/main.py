from pathlib import Path
import numpy as np


def run_rd_clustering(blocks, vertices, attributes, config):
    
    # Create components
    cache = InMemoryCacheStrategy()
    slope_optimizer = SlopeOptimizer(add_intercept=True)
    convergence_checker = ConvergenceChecker(
        max_iterations=config['max_iterations'],
        label_change_threshold=config['label_threshold']
    )
    training_selector = TrainingSetSelector(
        selection_ratio=config['training_ratio']
    )
    
    lambda_schedule = np.logspace(
        np.log10(config['lambda_min']),
        np.log10(config['lambda_max']),
        config['lambda_steps']
    )
    
    clusterer = RDClusterer(
        num_clusters=config['num_clusters'],
        lambda_schedule=lambda_schedule,
        gft_cache=cache,
        slope_optimizer=slope_optimizer,
        convergence_checker=convergence_checker,
        training_selector=training_selector,
        use_two_stage=config['use_two_stage']
    )
    
    final_state = clusterer.fit(blocks, vertices, attributes)
    cache.cleanup()
    
    return final_state


if __name__ == "__main__":
    config = {
        'num_clusters': 8,
        'lambda_min': 0.1,
        'lambda_max': 10.0,
        'lambda_steps': 5,
        'max_iterations': 20,
        'label_threshold': 0.01,
        'training_ratio': 0.1,
        'use_two_stage': True
    }
    
    # TODO: Load your data
    # blocks, vertices, attributes = load_data()
    # final_state = run_rd_clustering(blocks, vertices, attributes, config)


