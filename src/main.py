from parameters import *
from graph import *
from encoder import *
from decider import *
from transforms import *
from objects import *
from visualization import *
from itertools import product
from line_profiler import profile
# from memory_profiler import profile
import logging
import shutil
from tqdm import tqdm

"""https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634"""

# Custom logging handler for tqdm
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.WARNING):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Use tqdm.write to print logs above the progress bar
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(filename="logs/main.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add custom handler for tqdm output
logger.addHandler(TqdmLoggingHandler())

# Add a file handler to write to the log file
file_handler = logging.FileHandler('logs/main.log', mode='w')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Researcher():
    # TODO: Use less RAM, currently up to 9Gb of RAM
    def __init__(self):
        self.point_cloud =  PointCloud()
        self.GFT_computer = GFT()
        self.encoder = Encoder()
        self.decider = Decider(mode="0")
        self.visualizer = Visualizer()
        # self.graphs = []  # TODO: Don't know what was intended here
        self.blocks = None
        self.param = [None] *5
        self.graphs_dict: Optional[dict[UUID,AttributeGraph | StructuralGraph]] = {}

    @profile
    def __call__(self, params: ExperimentParameters):
        param_combinations = self._generate_combinations(params)
        export_folder = Path(params.export_folder)
        experiment_code = params.experiment_code
        self.debugging = params.debugging
        for i,param in enumerate(tqdm(param_combinations, desc="Running parameters: ")):
            logger.info(self._params_msg(param))
            if param[0] != self.param[0]:
                self.point_cloud(param[0])
            if param[1] != self.param[1]:
                self.block_manager = BlockManager(bsize=param[1], export_folder=export_folder, experiment_code=experiment_code, point_cloud_path=param[0], rewrite=params.rewrite_results)
                if self.blocks:
                    self._exec_encoding(params.quantization_steps)
                self.point_cloud.do_block_partitioning(bsize=param[1])   # Restart blocks
            if param[2] != self.param[2] or param[3] != self.param[3]:
                # self.graphs.clear()
                self._block_processing(self_loop_weight=param[2], self_loop_percentage=param[3])
            if i == len(param_combinations)-1:
                self._exec_encoding(params.quantization_steps)
            self.param = param
        pass

        
    @profile
    def _exec_encoding(self, q_steps: List[int]):
        for q_step in tqdm(q_steps, desc= "Iterating over quantization steps: "):
            logger.info(f"Quantization Step: {q_step}")
            Coeffs, graph_ids = self._per_block_decider(q_step)
            # selected_graphs = [self.graphs[i] for i,graph_id in enumerate(graph_ids) if self.graphs[i] == graph_id]
            selected_graphs = []
            indexes = self.point_cloud.indexes
            PSNR, bpv, bsize = self.encoder(Coeffs, selected_graphs, q_step, indexes)
            self.block_manager.add_overall(q_step=q_step, psnr=PSNR, bpv=bpv, bitcount=bsize)
            logger.info(self._result_msg(PSNR, bpv, bsize))

    def _result_msg(self, PSNR: float, bpv:float, bsize: int):
        log_msg = f"""Results: 
                    Total bitstream = {bsize}
                    Bits per voxel = {bpv}
                    Peak Signal-to-Noise Ratio = {PSNR}"""
        return log_msg
        
    @profile
    def _per_block_decider(self,q_step: int):
        selected_graph_ids = []
        Coeffs = np.zeros(self.point_cloud.A.shape, dtype=np.float64)    
        for block in tqdm(self.blocks, desc=f"Rate-Distorsion Optimization for Quantization Step {q_step}: "):
            selected_graph_id, selected_coeff = self._block_decider(block, q_step)
            self.block_manager.add_decision(block, q_step, selected_graph_id, selected_coeff)
            Coeffs[block.as_index(),:] = selected_coeff
            selected_graph_ids.append(selected_graph_id)
        return Coeffs, selected_graph_ids
    
    def _block_decider(self, block: Block, q_step: int):
        coeffs_dict = self.block_manager.get_coefficients(block.id)
        selected_graph_id, selected_coeff = self.decider(q_step, coeffs_dict, block.id)
        logger.info(f"Selected {self._decision_msg(selected_graph_id, block.id)}")
        return selected_graph_id, selected_coeff
            
    def _decision_msg(self, selected_graph_id: str, block_id:str):
        # TODO: Create Object factory and avoid circular import
        sl_weight, sl_percentage = map(float , selected_graph_id.split("_"))
        edges = self.block_manager.get_data(block_id, selected_graph_id, "edges")
        sl_count = np.sum(edges[:,0] == edges[:,1])
        if sl_percentage > 0: # Self-loop percentage
            log_msg = f""" Attribute Graph: 
                Self-Loop Weight: {sl_weight}
                Self-Loop Percentage: {sl_percentage}
                Self-Loop Count: {sl_count}
            """
        else:
            log_msg = f""" Structural Graph"""
        return log_msg
    
    def _generate_combinations(self, params: ExperimentParameters):
        multiple_params = [
            params.point_cloud_path,
            params.block_size,
            params.self_loop_weight,
            params.self_loop_percentage,
        ]
        return list(product(*multiple_params))
    
    @profile
    def _block_processing(self, self_loop_weight: float, self_loop_percentage: float):
        self.blocks = self.point_cloud.get_all_blocks()         # Start end tuples
        V = self.point_cloud.V
        A = self.point_cloud.A
       
        self.bugs_idx = []
        for block in tqdm(self.blocks, desc="Processing blocks: "):
            self._process_block(V, A, block, self_loop_weight, self_loop_percentage)
        logger.info(f"Bad working blocks {self.bugs_idx}")

    @profile
    def _process_block(self,V: np.ndarray, A:np.ndarray, block: Block, sl_weight: float, sl_percentage: float):
        block._init_data(V, A)
        if block.id not in self.block_manager.list_blocks(): # Check if block is in file
            struct_graph = StructuralGraph(block.id)
            # self.graphs.append(struct_graph)
            struct_graph._init_data(V=block.Vblock)
            self._add_block(block)
            self._add_graph(struct_graph, block)
            struct_graph._del_data()
        attr_graph = AttributeGraph(block.id, sl_weight=sl_weight, sl_percentage=sl_percentage)
        attr_graph._init_data(block.Vblock, block.Ablock)
        # self.graphs.append(attr_graph)
        self._add_graph(attr_graph, block)
        attr_graph._del_data()
        block._del_data()

    def _visualize_transform(self, result: tuple[np.ndarray, np.ndarray], vis_gft:bool):
        self.visualizer.visualize_block_coeffs(result=result, title=f"Coeffs for block", vis_gft = vis_gft)
        plt.show(block=True)


    def _add_block(self, block: Block):
        if str(block.id) not in self.block_manager.list_blocks():
            self.block_manager.add_block(block)

    @profile
    def _add_graph(self, graph: Graph, block: Block):
        """Process and store graph results with configurable debugging.
        
        Args:
            graph: Graph object to process
            block: Associated block data
            debug: If True, enables detailed logging (default: False)
        """
        self.visualizer(graph, block)
        if not self.block_manager.matched_metadata(graph):
            # Compute GFT transform
            gft_mat, coeffs = self.GFT_computer(graph, block)
            
            # Visualization trigger condition (DC component check)
            dc_check = np.sum(coeffs[0,0] < coeffs[:,0]) >= 1
            if self.debugging and dc_check:
                self._block_debugger(graph, block,gft_mat, coeffs)
                self.visualizer.visualize_coeffs(coeffs)
                self.visualizer.visualize_gft(gft_mat)
                self.visualizer.display()
                # self._visualize_transform((gft_mat, coeffs), vis_gft=True)
            self.visualizer.close()
            # Store results
            self.block_manager.add_result(graph, (gft_mat, coeffs))

    def _block_debugger(self, graph: Graph, block: Block, gft_mat: np.ndarray, coeffs: np.ndarray):
        logger.debug("Visualization triggered - DC component not dominant")
        logger.debug("\n=== GFT Computation Debug ===")
        logger.debug(f"Graph ID: {graph.id}")
        logger.debug(f"Graph Type: {type(graph).__name__}")
        logger.debug(f"Block Size: {block.Vblock.shape}")
        logger.debug(f"GFT Matrix Shape: {gft_mat.shape}")
        logger.debug(f"Coefficients Shape: {coeffs.shape}")
         
        # Channel statistics
        for i, ch in enumerate(['Y', 'U', 'V']):
            logger.debug(f"{ch} Channel - Min: {np.min(coeffs[:,i]):.4f} "
                    f"DC: {coeffs[0,i]:.4f} "
                    f"Max: {np.max(coeffs[:,i]):.4f} at pos {np.where(coeffs[:,i] == np.max(coeffs[:,i]))}"
                    f"Mean: {np.mean(coeffs[:,i]):.4f}")
        
        # Data quality checks
        if np.any(np.isnan(coeffs)):
            logger.warning("NaN values detected in coefficients!")
        if np.any(np.abs(coeffs) > 1e6):
            logger.warning("Extremely large coefficient values detected!")

        logger.debug("DC coefficients non dominant -> Visualization triggered")
        self.bugs_idx.append(block.id)

    def _encode_coeffs(self, Coeffs: np.ndarray, q_step: int):
        self.encoder(Coeffs, q_step)
        

    def _params_msg(self,param: list[Path, int, float, float]):
        log_msg = f"""Currently running with parameters:
        ================================================    
            Point Cloud: {param[0].stem}
            Block Size: {param[1]}
            Self-loop Weight: {param[2]}
            Self-loop Percentage: {param[3]}
        ================================================    
        """
        return log_msg

    def _save_config(self, params: ExperimentParameters):
        pass

    def _save_data(self):
        pass

    def _save_plots(self):
        pass

    
class Analyst():
    def __init__(self):
        self.visualizer = Visualizer()
        self.visualizer._init_2d_figure()
        pass

    def __call__(self, h5_path: Path, label: str, color: str, decisions: bool = False):
        self.h5path = h5_path
        self.label = label
        self.color = color
        if decisions:
            self.decision_stats()
        self.rate_distortion_curve()

    def decision_stats(self):
        decision_df = self._load_decisions()
        decision_counts = (
            decision_df.groupby(["q_step", "sl_weight", "sl_percentage"])
            .agg(block_count=("block_id", "nunique"))
            .reset_index()
            .sort_values(by=["q_step", "block_count"], ascending=[True, False])
        )
        q_steps = sorted(decision_counts["q_step"].unique())

        for q in q_steps:
            df_q = decision_counts[decision_counts["q_step"] == q].copy()

            # ======= BAR PLOT =======
            df_q["decision"] = df_q.apply(
                lambda row: f"w:{row['sl_weight']}, p:{row['sl_percentage']}", axis=1)

            plt.figure(figsize=(10, 5))
            sns.barplot(data=df_q, x="decision", y="block_count", palette="Blues_d")
            plt.title(f"Decision Counts - q_step {q}")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Block Count")
            plt.xlabel("Self-loop Decision (weight, percentage)")
            plt.tight_layout()
            plt.show()
        

    def _load_decisions(self):
        stats = []
        with h5py.File(self.h5path, "r") as f:
            for block_id in tqdm(f["blocks"].keys(), "Checking block decisions"):
                block_path = f["blocks"][block_id]
                decision_group = block_path["decision"]
                for q_step in decision_group.keys():
                    grp = decision_group[q_step]
                    sl_weight = grp["sl_weight"][()]
                    sl_percentage = grp["sl_percentage"][()]
                    stats.append({
                        "block_id": int(block_id),
                        "q_step": int(q_step),
                        "sl_weight": sl_weight,
                        "sl_percentage": sl_percentage
                    })
        return pd.DataFrame(stats)

    def rate_distortion_curve(self):
        rd_data = {}
        with h5py.File(self.h5path, "r") as f:
            results_group = f["results"]
            for q_step in results_group.keys():
                bpv = results_group[q_step]["bpv"][()]
                PSNR = results_group[q_step]["psnr"][()]
                rd_data[int(q_step)] = (float(bpv), float(PSNR))

        sorted_qsteps = sorted(rd_data.keys())
        bpv_values = [rd_data[q][0] for q in sorted_qsteps]
        psnr_values = [rd_data[q][1] for q in sorted_qsteps]
        self.visualizer.add_rd_data(sorted_qsteps, bpv_values, psnr_values, color=self.color, label=self.label) 

    def plot_rd_curve(self):
        self.visualizer.visualize_rd()

    def bjontegaard_delta(self):
        pass



if __name__ == "__main__":
    params = load_experiment_parameters(Path("config/config.yaml"))
    export_folder = Path(params.export_folder)
    shutil.copy2(Path("config/config.yaml"), export_folder)
    researcher = Researcher()
    researcher(params)
    analyst = Analyst()
    analyst(Path("/media/simao/TOSHIBA EXT/Experiments/BE01/longdress_vox10_1051/block_size16_data.h5"),label="Block 16 GFT Standard", color='red')
    analyst(Path("/media/simao/TOSHIBA EXT/Experiments/BE01/longdress_vox10_1051/block_size8_data.h5"),label="Block 8 GFT Standard", color='blue')
    # # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/BE01/longdress_vox10_1051/block_size4_data.h5"),label="Block 4 GFT Standard", color='green')
    analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE12/longdress_vox10_1051/block_size16_data.h5"),label="Block 16 GFT Dynamic", color='orange')
    analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE12/longdress_vox10_1051/block_size8_data.h5"),label="Block 8 GFT Dynamic", color='purple')
    analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE12/longdress_vox10_1051/block_size4_data.h5"),label="Block 4 GFT Dynamic", color='cyan')
    # analyst.decision_stats()
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/BE01/longdress_vox10_1051/block_size8_data.h5"),label="Block 8 GFT Standard", color='blue')
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE05/longdress_vox10_1051/block_size8_data.h5"),label="Block 8 GFT Dynamic", color='cyan')
    # analyst.decision_stats()

    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE03/longdress_vox10_1051/block_size16_data.h5"),label="Block GFT Standard", color='cyan')
    # analyst.decision_stats()
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE02/longdress_vox10_1051/block_size16_data.h5"),label="Block GFT Standard", color='cyan')

    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/TE01/longdress_vox10_1051/block_size16_data.h5"),label="Block GFT Standard", color='cyan')
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/NT01/longdress_vox10_1051/block_size16_data.h5"), label="Block GFT RDO Self-looped", color='red')
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/MR01/longdress_vox10_1051/block_size16_data.h5"), label="Block GFT Modified RDO Self-looped", color='green')
    # analyst.decision_stats()
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/MR02/longdress_vox10_1051/block_size16_data.h5"), label="Block GFT Modified RDO Self-looped", color='orange')
    # analyst(Path("/media/simao/TOSHIBA EXT/Experiments/MR06/longdress_vox10_1051/block_size16_data.h5"), label="Block GFT Modified RDO Self-looped", color='black')
# # 

    analyst.plot_rd_curve()
    plt.show()