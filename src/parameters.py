from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml
import logging
logging.basicConfig(filename="logs/parameters.log", 
                    filemode="w", 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )
logger = logging.getLogger(__name__)

@dataclass
class ExperimentParameters:
    # Experiment general data
    experiment_name: str
    experiment_date: str
    experiment_info: str

    # File and paths
    point_cloud_path: List[Path]  # Always a list of paths
    export_folder: Path

    # Method parameters
    self_loop_weight: List[float]  # Always a list of floats
    self_loop_percentage: List[float]  # Always a list of floats

    # Encoding ranges parameters
    quantization_steps: List[int]  # Always a list of ints
    block_size: List[int]  # Always a list of ints

    def __post_init__(self):
        # Ensure lists have consistent lengths (all lists should have the same length)
        max_len = max(len(self.point_cloud_path), len(self.self_loop_weight), len(self.self_loop_percentage), 
                      len(self.quantization_steps), len(self.block_size))

        self.point_cloud_path = self._normalize_to_list(self.point_cloud_path, max_len)
        self.self_loop_weight = self._normalize_to_list(self.self_loop_weight, max_len)
        self.self_loop_percentage = self._normalize_to_list(self.self_loop_percentage, max_len)
        self.quantization_steps = self._normalize_to_list(self.quantization_steps, max_len)
        self.block_size = self._normalize_to_list(self.block_size, max_len)

        # Validate ranges
        for value in self.self_loop_percentage:
            if value < 0 or value > 1:
                raise ValueError(f"Applied self-loop percentage should be in range [0,1], but got {value}")

        if not self.experiment_name:
            raise ValueError("Experiment must have a name.")

    def _normalize_to_list(self, param, max_len):
        # If the parameter is a single value, convert it into a list of the appropriate length
        if not isinstance(param, list):
            param = [param] * max_len
        return param

    def __str__(self):
            """
            Custom string representation for logging and display.
            """
            return f""""
                        ExperimentParameters
                ============================================
                  Name: {self.experiment_name}
                  Date: {self.experiment_date}
                  Info: {self.experiment_info}
                  Point Cloud: {self.point_cloud_path}
                  Export Path: {self.export_folder}
                  Self-loop Weights: {self.self_loop_weight}
                  Self-loop Percentages: {self.self_loop_percentage}
                  Quantization Steps: {self.quantization_steps}
                  Block Size: {self.block_size}
                """
        
# Parser method to load YAML into ExperimentParameters
def load_experiment_parameters(yaml_file: Path) -> ExperimentParameters:
    """
    Load experiment parameters from a YAML file and log the loaded parameters.
    """
    try:
        logger.info(f"Loading parameters from YAML file: {yaml_file}")
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)

        # Convert paths to Path objects
        config["point_cloud_path"] = [Path(p) for p in config["point_cloud_path"]]
        config["export_folder"] = Path(config["export_folder"])

        # Create an instance of ExperimentParameters
        params = ExperimentParameters(**config)

        # Log the loaded parameters
        logger.info("Successfully loaded parameters:")
        logger.info(params)

        return params
    except Exception as e:
        logger.error(f"Failed to load parameters from {yaml_file}: {e}")
        raise

# Example usage
if __name__ == "__main__":
    yaml_file = "experiment_config.yaml"
    params = load_experiment_parameters(yaml_file)
    print(params)