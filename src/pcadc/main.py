import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Configure logging
from tqdm import tqdm
from utils.bj_delta import bj_delta
from line_profiler import profile
from joblib import dump, load
import tempfile
import numpy as np
import json
import dataclasses
import yaml
from .rd_cluster.main import run_rd_clustering
from .factories import *
from .observer import *
from .transforms import *
from .pointcloud import *
from .io import *
from .decider import *
from .encoder import *
from .graph import *
from .blocks import *
from .parameters import *

# Setup logging to an absolute path to ensure it works when run as a module
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "main.log")

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)

class ExperimentResults:
    def __init__(self):
        self._results = []

    def add(self, encode_result: EncodeResult):
        self._results.append(encode_result)

    def as_dict(self):
        pass


# This is one subject for observer pattern
class Researcher():
    def __init__(self):
        self.cache_dir = Path(tempfile.mkdtemp(prefix="coeffs_cache_"))
        self.observers: List[ExperimentObserver] = []

    def attach(self, observer: ExperimentObserver):
        self.observers.append(observer)

    def _notify(self, event: ExperimentEvent) -> None:
        for observer in self.observers:
            observer.update(event)

    @profile
    def run(self, params: ExperimentConfig):
        self._init_experiment(params)
        indexes, blocks = self._block_partitioning()
        codebook = self._cluster_codebook(blocks)

        coeffs_stream = self._compute_coeffs(blocks, codebook)  # generator
        for result in self._exec_encoding(coeffs_stream, indexes, codebook):
            # save each result right away
            self._save_experiment(result, params)
            
    def _init_experiment(self, params: ExperimentConfig):
        self.experiment_params = params
        self.sequence_params = params.sequential_params
        self.pc_metadata = params.pointcloud
        self.debugging = params.debugging
        self.rewrite = params.rewrite_results
        self.metadata = params.metadata

    def _block_partitioning(self):
        colourist = Colourist()
        point_cloud_path = self.sequence_params.point_cloud_path
        point_cloud = PointCloud.from_file(
            path=point_cloud_path, fmt="ply", metadata=self.pc_metadata)
        point_cloud.transform_attributes(colourist._RGBtoYUV)
        self.V = point_cloud.vertices
        self.A = point_cloud.attributes
        mortonpartition = MortonBlockPartition()
        return mortonpartition.partition(point_cloud, bsize=self.sequence_params.block_size)

    def _cluster_codebook(self,blocks: List[Block]):
        final_state, clustering_history = run_rd_clustering(blocks, self.V, self.A, self.experiment_params)
        
        # Notify observers with the clustering history
        self._notify(ClusteringHistoryEvent(
            experiment_code=self.metadata.experiment_code,
            block_size=self.sequence_params.block_size,
            history=clustering_history
        ))
        
        return Codebook(final_state.slopes, final_state.labels, final_state.labels)

    def _compute_coeffs(self, blocks: List[Block], codebook: Codebook):
        gft_strategy_wraper = GFTStrategyWraper()
        coeffs_paths = []

        for block_idx, block in tqdm(enumerate(blocks), "Init graphs + coeffs per block"):
            graphs_list = self._init_graphs(block, block_idx, codebook)
            block.init_data(self.V, self.A)

            coeffs_list = []
            metadata_list = []
            for graph in graphs_list:
                GFT_mat, coeffs = gft_strategy_wraper(block, graph)
                coeffs_list.append(coeffs)
                metadata_list.append(graph.metadata)
                self._notify(CoeffsEvent(block, graph, coeffs, GFT_mat))
                graph.clear_data()
            block.clear_data()

            coeff_container =CoeffsContainer(
                block, metadata_list, coeffs_list)

            # dump to disk
            path = self.cache_dir / f"coeff_{block_idx}.pkl"
            dump(coeff_container, path)
            coeffs_paths.append(path)

        return coeffs_paths

    def _init_graphs(self, block: Block, block_idx: int, codebook: Codebook):
        # luminance_centroid = codebook.get_assigned_centroid(block_idx=block_idx)
        codebook_label = codebook.labels[block_idx]
        luminance_centroid = codebook.centroids[codebook_label]
        graphblock_factory = GraphBlockCreator(
            self.V, self.A, block,
            codebook_label,
            luminance_centroid,
            self.sequence_params
        )
        block.init_data(self.V, self.A)
        self._notify(CodebookEvent(block, codebook))
        block.clear_data()
        return graphblock_factory.get_all_products()

    def _exec_encoding(self, coeffs_paths, indexes: List, codebook: Codebook):
        encoder = Encoder()
        rdo_decider = Decider(
            mode="0", lagrange_proportional=self.sequence_params.lagrange_proportional)
        experiment_results = ExperimentResults()

        for q_step in tqdm(self.sequence_params.quantization_steps, "Iterating over quantization steps"):
            assignation_with_qstep = codebook.assignation.copy()
            Coeffs = np.zeros(self.A.shape, dtype=np.float64)

            for i, path in tqdm(enumerate(coeffs_paths), "RDO decider"):
                coeff_container = load(path)  # lazy load per block
                rdo_decision = rdo_decider(q_step, coeff_container)
                self._notify(RDOEvent(q_step, rdo_decision, coeff_container))
                # FIXME: What is this?
                assignation_with_qstep[i] = rdo_decision.get_label_from_luminance(
                    codebook)
                Coeffs[coeff_container.block.as_index(
                ), :] = rdo_decision.selected_coeffs

            result = encoder(Coeffs, q_step, indexes, assignation_with_qstep)
            self._notify(EncodeEvent(self.metadata.experiment_code, result))
            experiment_results.add(result)
            yield result


    def _save_experiment(self, encode_result: EncodeResult, params: ExperimentConfig):
        # Define the base export path
        export_base_path = Path(
            self.metadata.export_folder) / self.metadata.experiment_code
        export_base_path.mkdir(parents=True, exist_ok=True)

        # Create a run-specific folder if it doesn't exist
        run_path = export_base_path
        run_path.mkdir(parents=True, exist_ok=True)

        # Save the master config file once per run
        config_file_path = run_path / "config.json"
        if not config_file_path.exists():
            config_dict = params.to_dict()
            with open(config_file_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

        # Save the results for the current q_step
        results_dir = run_path / "results"
        results_dir.mkdir(exist_ok=True)
        result_file_path = results_dir / f"qstep_{encode_result.q_step}.json"

        result_dict = dataclasses.asdict(encode_result)

        with open(result_file_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

        logger.info(
            f"Saved results for q_step {encode_result.q_step} to {result_file_path}")


def run_experiments(params: ExperimentConfig, enable_viz: bool = False):
    researcher = Researcher()
    
    # Create the directory structure first
    db_dir = params.metadata.export_folder / Path(params.metadata.experiment_code)
    db_dir.mkdir(exist_ok=True, parents=True)
    
    # Then create the database file path
    db_path = db_dir / f"{params.metadata.experiment_code}.db"
    
    # Save the final configuration file in the result folder
    config_save_path = db_dir / "final_config.yaml"
    params.save_to_yaml(config_save_path)
    
    print(f"[*] Configuration saved to: {config_save_path}")
    logger.info(f"Configuration saved to: {config_save_path}")

    database_observer = SQLiteSink(db_path)
    visualization_observer = DiagnosticVisualizer()
    if not enable_viz:
        visualization_observer.disable()
    
    researcher.attach(database_observer)
    researcher.attach(visualization_observer)
    researcher.run(params)

def main():
    base_config_path = Path("config/base_config.yaml")
    if not base_config_path.exists():
        print(f"[!] Error: Base config file not found at {base_config_path}")
        sys.exit(1)

    # Load base parameters to use as defaults
    base_params = load_experiment_config(base_config_path)

    parser = argparse.ArgumentParser(description='Run PCADC experiments with automated code generation')
    
    # Mandatory overrides (made mandatory to avoid silent bugs as requested)
    parser.add_argument('--block_size', type=int, required=True, help='Mandatory: Block size for partitioning')
    parser.add_argument('--clusters', type=int, required=True, help='Mandatory: Number of clusters')
    
    # Optional overrides with defaults from base_config
    parser.add_argument('--slt', type=float, default=base_params.sequential_params.self_loop_threshold, help='Self-loop threshold')
    parser.add_argument('--slw', type=float, default=base_params.sequential_params.self_loop_weight, help='Self-loop weight')
    parser.add_argument('--qsteps', type=int, nargs='+', default=base_params.sequential_params.quantization_steps, help='Quantization steps list')
    
    # Execution flags
    parser.add_argument('--viz', action='store_true', help='Enable diagnostic visualization (Optional)')
    parser.add_argument('--rewrite', action='store_true', help='Rewrite existing results (Optional)')
    
    # Identification
    parser.add_argument('--name', type=str, default=base_params.metadata.experiment_name, help='Experiment descriptive name')

    args = parser.parse_args()

    # Apply overrides to base_params
    base_params.sequential_params.block_size = args.block_size
    base_params.clustering_params.number_of_clusters = args.clusters
    base_params.sequential_params.self_loop_threshold = args.slt
    base_params.sequential_params.self_loop_weight = args.slw
    base_params.sequential_params.quantization_steps = args.qsteps
    base_params.rewrite_results = args.rewrite
    base_params.metadata.experiment_name = args.name

    # Automatic code generation: RD-B{Block_Size}-C{Cluster_Num}-SW_{Self_loop_weight}-ST_{Self_loop_threshold}
    # Sanitize float strings for filenames (replace . with _)
    sw_str = str(args.slw).replace('.', '_')
    st_str = str(args.slt).replace('.', '_')
    base_params.metadata.experiment_code = f"RD-B{args.block_size}-C{args.clusters}-SW_{sw_str}-ST_{st_str}"

    print("-" * 50)
    print(f"[*] Starting Experiment: {base_params.metadata.experiment_name}")
    print(f"[*] Experiment Code:     {base_params.metadata.experiment_code}")
    print(f"[*] Block Size:          {base_params.sequential_params.block_size}")
    print(f"[*] Clusters:            {base_params.clustering_params.number_of_clusters}")
    print(f"[*] SL Weight:           {base_params.sequential_params.self_loop_weight}")
    print(f"[*] SL Threshold:        {base_params.sequential_params.self_loop_threshold}")
    print(f"[*] Q-Steps:             {base_params.sequential_params.quantization_steps}")
    print(f"[*] Visualization:       {'Enabled' if args.viz else 'Disabled'}")
    print(f"[*] Rewrite Results:     {'Enabled' if args.rewrite else 'Disabled'}")
    print("-" * 50)

    logger.info(f"Starting experiment {base_params.metadata.experiment_code} with parameters: {base_params}")

    run_experiments(base_params, enable_viz=args.viz)
    
    print(f"[!] Experiment {base_params.metadata.experiment_code} completed successfully.")

if __name__ == "__main__":
    main()
