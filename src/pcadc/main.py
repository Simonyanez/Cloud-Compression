# Configure logging
from tqdm import tqdm
from utils.bj_delta import bj_delta
from line_profiler import profile
from joblib import dump, load
import tempfile
import numpy as np
import json
import dataclasses
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
import logging
import os

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
# from .visualization import *
# from memory_profiler import profile

"""https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634"""

# Custom logging handler for tqdm


# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.WARNING):
#         super().__init__(level)
#
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             # Use tqdm.write to print logs above the progress bar
#             tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record)


# Add custom handler for tqdm output
# logger.addHandler(TqdmLoggingHandler())

# Add a file handler to write to the log file
# file_handler = logging.FileHandler('logs/main.log', mode='w')
# file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


class ExperimentResults:
    def __init__(self):
        self._results = []

    def add(self, encode_result: EncodeResult):
        self._results.append(encode_result)

    def as_dict(self):
        pass


# TODO: Experiment Results Summary Dataclass

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

    # def _cluster_codebook(self, blocks: List[Block]):
    #     fit_collection = FitCollection()
    #     approximator = Approximator()
    #     for block in tqdm(blocks, "Fitting luminansce by linear approximation"):
    #         block.init_data(self.V, self.A)
    #         fit_result = approximator(block)
    #         fit_collection.add(fit_result)
    #         self._notify(FitEvent(block, fit_result))
    #         block.clear_data()
    #     clusterer = Clusterer(
    #         self.sequence_params.number_of_clusters, self.sequence_params.normalize_slopes)
    #     codebook = clusterer(fit_collection)
    #     codebook.assign(blocks, self.V, self.A)
    #     return codebook

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
        # Assuming `encode_result` contains a `q_step` attribute
        # and has a method to convert itself to a dictionary.

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
            # Convert your ExperimentConfig dataclass to a dictionary
            # You might need a helper function for this
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


# class Analyst():
#     def __init__(self):
#         self.visualizer = Visualizer()
#         self.visualizer._init_2d_figure()
#         self.stored = {}
#         pass
#
#     def __call__(self, h5_path: Path, id: str):
#         self.h5path = h5_path
#         self.id = id
#
#     def decision_stats(self):
#         decision_df = self._load_decisions()
#         decision_counts = (
#             decision_df.groupby(["q_step", "sl_weight", "sl_percentage"])
#             .agg(block_count=("block_id", "nunique"))
#             .reset_index()
#             .sort_values(by=["q_step", "block_count"], ascending=[True, False])
#         )
#         q_steps = sorted(decision_counts["q_step"].unique())
#
#         # TODO: Move this to visualization
#         for q in q_steps:
#             df_q = decision_counts[decision_counts["q_step"] == q].copy()
#
#             # ======= BAR PLOT =======
#             df_q["decision"] = df_q.apply(
#                 lambda row: f"w:{row['sl_weight']}, p:{row['sl_percentage']}", axis=1)
#
#             plt.figure(figsize=(10, 5))
#             sns.barplot(data=df_q, x="decision",
#                         y="block_count", palette="Blues_d")
#             plt.title(f"Decision Counts - q_step {q}")
#             plt.xticks(rotation=45, ha="right")
#             plt.ylabel("Block Count")
#             plt.xlabel("Self-loop Decision (weight, percentage)")
#             plt.tight_layout()
#             plt.show()
#
#     def _load_decisions(self):
#         stats = []
#         with h5py.File(self.h5path, "r") as f:
#             for block_id in tqdm(f["blocks"].keys(), "Checking block decisions"):
#                 block_path = f["blocks"][block_id]
#                 decision_group = block_path["decision"]
#                 for q_step in decision_group.keys():
#                     grp = decision_group[q_step]
#                     sl_weight = grp["sl_weight"][()]
#                     sl_percentage = grp["sl_percentage"][()]
#                     stats.append({
#                         "block_id": int(block_id),
#                         "q_step": int(q_step),
#                         "sl_weight": sl_weight,
#                         "sl_percentage": sl_percentage
#                     })
#         return pd.DataFrame(stats)
#
#     def rate_distortion_curve(self, label: str, color: str, linestyle: str):
#         rd_data = {}
#         with h5py.File(self.h5path, "r") as f:
#             results_group = f["results"]
#             for q_step in results_group.keys():
#                 bpv = results_group[q_step]["bpv"][()]
#                 PSNR = results_group[q_step]["psnr"][()]
#                 bitcount = results_group[q_step]["bitcount"][()]
#                 rd_data[int(q_step)] = (float(bpv), float(PSNR), int(bitcount))
#
#         sorted_qsteps = sorted(rd_data.keys())
#         bpv_values = [rd_data[q][0] for q in sorted_qsteps]
#         psnr_values = [rd_data[q][1] for q in sorted_qsteps]
#         bitcount_values = [rd_data[q][2] for q in sorted_qsteps]
#         self.stored[self.id] = {"qsteps": sorted_qsteps, "bpv": bpv_values,
#                                 "PSNR": psnr_values, "bitcount": bitcount_values}
#         self.visualizer.add_rd_data(
#             sorted_qsteps, bpv_values, psnr_values, color=color, label=label, linestyle=linestyle)
#
#     def cleanup_stored(self):
#         self.stored = {}
#
#     def plot_rd_curve(self):
#         self.visualizer.visualize_rd()
#
#     def bjontegaard_delta(self, id_1, id_2):
#         # Assuming you compare same number of q steps
#         qsteps = self.stored[id_1]["qsteps"]
#         R1 = self.stored[id_1]["bpv"]
#         PSNR1 = self.stored[id_1]["PSNR"]
#         bitcount1 = self.stored[id_1]["bitcount"]
#         R2 = self.stored[id_2]["bpv"]
#         PSNR2 = self.stored[id_2]["PSNR"]
#         bitcount2 = self.stored[id_2]["bitcount"]
#         bd_psnr = bj_delta(R1, PSNR1, R2, PSNR2, mode=0)
#         bd_rate = bj_delta(R1, PSNR1, R2, PSNR2, mode=1)
#         for i, q in enumerate(qsteps):
#             print(
#                 f"Quantization Step: {q} - Bitcount diff {abs(bitcount2[i]-bitcount1[i])}\n ==================================================")
#         result_str = f"Bjontegaard Metrics: \n BD-PSNR: {bd_psnr} - BD-Rate: {bd_rate} \n =================================================="
#         print(result_str)


def run_experiments(config_path: Path):
    params = load_experiment_config(config_path)
    researcher = Researcher()
    
    # Create the directory structure first
    db_dir = params.metadata.export_folder / Path(params.metadata.experiment_code)
    db_dir.mkdir(exist_ok=True, parents=True)
    
    # Then create the database file path
    db_path = db_dir / f"{params.metadata.experiment_code}.db"
    
    database_observer = SQLiteSink(db_path)
    visualization_observer = DiagnosticVisualizer()
    researcher.attach(database_observer)
    researcher.attach(visualization_observer)
    researcher.run(params)
#
#
# def run_results():
#     experiments = ["BE01", "TH01"]
#     descriptions = ["Standard", "Dynamic"]
#     linestyles = ["solid", "dashed"]
#     description_map = dict(zip(experiments, descriptions))
#     linestyle_map = dict(zip(experiments,  linestyles))
#     bsizes = ["16", "8", "4"]
#     colors = ["red", "green", "blue"]
#     color_map = dict(zip(bsizes,  colors))
#     analyst = Analyst()
#     for bsize in bsizes:
#         for experiment in experiments:
#             analyst(Path(
#                 f"/media/simao/TOSHIBA EXT/Experiments/{experiment}/longdress_vox10_1051/block_size{bsize}_data.h5"), id=f"b{bsize}-{experiment}")
#             analyst.rate_distortion_curve(
#                 label=f"Block {bsize} - {description_map[experiment]} GFT", color=color_map[bsize], linestyle=linestyle_map[experiment])
#             analyst.decision_stats()
#             plt.show()
#         print(
#             f"Analysis for Block of size {bsize}\n ==================================================")
#         analyst.bjontegaard_delta(
#             f"b{bsize}-{experiments[0]}", f"b{bsize}-{experiments[1]}")
#     analyst.plot_rd_curve()
#     plt.show()


if __name__ == "__main__":
    run_experiments(Path("config/config_medium_b16_c8.yaml"))
    run_experiments(Path("config/config_medium_b8_c8.yaml"))
    run_experiments(Path("config/config_medium_b4_c8.yaml"))
    pass
