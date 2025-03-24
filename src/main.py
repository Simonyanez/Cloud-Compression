from parameters import *
from graph import *
from encoder import *
from decider import *
from transforms import *
from objects import *
from visualization import *
from itertools import product
import logging
from tqdm import tqdm

"""https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634"""

# Custom logging handler for tqdm
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
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
logger.addHandler(TqdmLoggingHandler())
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Researcher():
    def __init__(self):
        self.point_cloud =  PointCloud()
        self.GFT_computer = GFT()
        self.encoder = Encoder()
        self.decider = Decider()
        self.visualizer = Visualizer()
        self.structural_graph = None
        self.blocks = None
        self.param = [None] *5
        self.graphs_dict: Optional[dict[UUID,AttributeGraph | StructuralGraph]] = {}

    def __call__(self, params: ExperimentParameters):
        param_combinations = self._generate_combinations(params)
        for i,param in enumerate(tqdm(param_combinations, desc="Running parameters: ")):
            logger.info(self._params_msg(param))
            if param[0] != self.param[0]:
                self.point_cloud(param[0])
            if param[1] != self.param[1]:
                if self.blocks:
                    self._exec_encoding(params.quantization_steps)
                self.point_cloud.do_block_partitioning(bsize=param[1])
            if param[2] != self.param[2] or param[3] != self.param[3]:
                self._block_processing(self_loop_weight=param[2], self_loop_percentage=param[3])
            if i == len(param_combinations)-1:
                self._exec_encoding(params.quantization_steps)
            self.param = param
        pass

        

    def _exec_encoding(self, q_steps: List[int]):
        for q_step in tqdm(q_steps, desc= "Iterating over quantization steps: "):
            logger.info(f"Quantization Step: {q_step}")
            Coeffs, graph_ids = self._per_block_decider(q_step)
            selected_graphs = [self.graphs_dict[graph_id] for graph_id in graph_ids]
            indexes = self.point_cloud.indexes
            bpv, PSNR = self.encoder(Coeffs, selected_graphs, q_step, indexes)
            logger.info(self._result_msg(bpv, PSNR))

    def _result_msg(self, bpv: float, PSNR: float):
        log_msg = f"""Results: 
                    Bits per voxel = {bpv}
                    Peak Signal-to-Noise Ratio = {PSNR}"""
        return log_msg
        
        
    def _per_block_decider(self,q_step: int):
        graph_ids = []
        Coeffs = np.zeros(self.point_cloud.A.shape)    
        for block in tqdm(self.blocks, desc="Rate-Distorsion Optimization: "):
            selected_coeff, selected_graph_id = self._block_decider(q_step, block=block)
            Coeffs[block.idxs] = selected_coeff
            graph_ids.append(selected_graph_id)
        return Coeffs, graph_ids

    def _block_decider(self, q_step: int, block: Optional[Block] = None, idx: Optional[int] = None):
        assert block is not None or idx is not None, "Block or block index missing"
        if idx:
            block = self.blocks[idx]
        coeffs_dict = block.get_coeffs_dict()
        selected_graph_id, selected_coeff = self.decider(q_step, coeffs_dict)
        logger.info(f"{str(block)} selected {self._decision_msg(selected_graph_id)}")
        return selected_coeff, selected_graph_id
            
    def _decision_msg(self, selected_graph_id):
        selected_graph = self.graphs_dict[selected_graph_id]
        print(type(selected_graph))
        if isinstance(selected_graph, AttributeGraph):
            diag = np.diag(selected_graph.weights)
            sl_pos = diag > 0
            sl_count = np.sum(sl_pos)
            sl_percentage = 100*sl_count/selected_graph.weights.shape[0]
            sl_weight = diag[sl_pos[0]]
            log_msg = f""" Attribute Graph: 
                Self-Loop Weight: {sl_weight}
                Self-Loop Percentage: {sl_percentage}
                Self-Loop Count: {sl_count}
            """
        if isinstance(selected_graph, StructuralGraph):
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
    
    def _block_processing(self, self_loop_weight, self_loop_percentage):
        self.blocks = self.point_cloud.get_all_blocks()
        for block in tqdm(self.blocks, desc="Processing blocks: "):
            if block.structural_graph.graph_id not in list(block.results.keys()):
                self.graphs_dict[block.structural_graph.graph_id] = block.structural_graph
                gft_mat, coeffs = self.GFT_computer(block.structural_graph, block)
                block.save_gft_result(block.structural_graph.graph_id, gft_mat, coeffs)
            attribute_graph = AttributeGraph(block.Vblock, block.Ablock, sl_weight= self_loop_weight, block_fraction=self_loop_percentage)
            self.graphs_dict[attribute_graph.graph_id] = attribute_graph
            gft_mat, coeffs = self.GFT_computer(attribute_graph, block)
            block.save_gft_result(attribute_graph.graph_id, gft_mat, coeffs)


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

    def _save_config(self):
        pass

    def _save_data(self):
        pass

    def _save_plots(self):
        pass

        


if __name__ == "__main__":
    params = load_experiment_parameters(Path("config/config.yaml"))
    researcher = Researcher()
    researcher(params)