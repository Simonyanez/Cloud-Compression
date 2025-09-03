from .io import *
from .pointcloud import *
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Any
import yaml
import logging

# ------------------- Logging Setup -------------------
logging.basicConfig(
    filename="logs/parameters.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------- Data Classes -------------------


@dataclass
class SequentialParameters:
    """
    Parameters for a single experiment run (pipeline).
    """
    point_cloud_path: Path
    number_of_clusters: int
    normalize_slopes: bool
    self_loop_weight: float
    self_loop_threshold: float
    quantization_steps: List[int]
    lagrange_proportional: float
    block_size: int


@dataclass
class ExperimentMetadata:
    """
    Metadata for the experiment.
    """
    experiment_name: str
    experiment_code: str
    experiment_date: str
    experiment_info: str
    export_folder: Path


@dataclass
class ExperimentConfig:
    """
    Full experiment configuration: metadata + sequential parameters.
    """
    metadata: ExperimentMetadata
    sequential_params: SequentialParameters
    pointcloud: PointCloudMetadata
    rewrite_results: bool = False
    debugging: bool = False

    def __str__(self):
        return (
            f"ExperimentConfig\n"
            f"==============================\n"
            f"Name: {self.metadata.experiment_name}\n"
            f"Code: {self.metadata.experiment_code}\n"
            f"Date: {self.metadata.experiment_date}\n"
            f"Info: {self.metadata.experiment_info}\n"
            f"Export Path: {self.metadata.export_folder}\n"
            f"Point Cloud: {self.sequential_params.point_cloud_path}\n"
            f"Number of Clusters: {self.sequential_params.number_of_clusters}\n"
            f"Normalize Slopes: {self.sequential_params.normalize_slopes}\n"
            f"Self-loop Weight: {self.sequential_params.self_loop_weight}\n"
            f"Self-loop Threshold: {self.sequential_params.self_loop_threshold}\n"
            f"Quantization Steps: {self.sequential_params.quantization_steps}\n"
            f"Lagrange Proportional: {self.sequential_params.lagrange_proportional}\n"
            f"Block Size: {self.sequential_params.block_size}\n"
            f"Rewrite Results: {self.rewrite_results}\n"
            f"Debugging: {self.debugging}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representation of the ExperimentConfig.
        """
        # Start with the default dataclass-to-dict conversion
        config_dict = asdict(self)

        # Recursively convert Path objects to strings
        return self._convert_paths(config_dict)

    def _convert_paths(self, data: Any) -> Any:
        """Private helper to recursively convert Path objects to strings."""
        if isinstance(data, Path):
            return str(data)
        if isinstance(data, dict):
            return {k: self._convert_paths(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._convert_paths(item) for item in data]
        return data


# ------------------- YAML Loader -------------------


def load_experiment_config(yaml_file: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    try:
        logger.info(f"Loading configuration from {yaml_file}")
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        # Parse sections
        metadata = ExperimentMetadata(
            experiment_name=config["experiment_metadata"]["experiment_name"],
            experiment_code=config["experiment_metadata"]["experiment_code"],
            experiment_date=config["experiment_metadata"]["experiment_date"],
            experiment_info=config["experiment_metadata"]["experiment_info"],
            export_folder=Path(config["experiment_metadata"]["export_folder"]),
        )

        pointcloud = PointCloudMetadata(
            dataset=config["pointcloud_metadata"]["point_cloud_dataset"],
            sequence=config["pointcloud_metadata"]["point_cloud_sequence"],
            depth=config["pointcloud_metadata"]["point_cloud_depth"],
            frame=config["pointcloud_metadata"]["point_cloud_frame"],
        )

        sequential_params = SequentialParameters(
            point_cloud_path=Path(
                config["sequential_params"]["point_cloud_path"]),
            number_of_clusters=config["sequential_params"]["number_of_clusters"],
            normalize_slopes=config["sequential_params"]["normalize_slopes"],
            self_loop_weight=config["sequential_params"]["self_loop_weight"],
            self_loop_threshold=config["sequential_params"]["self_loop_threshold"],
            quantization_steps=config["sequential_params"]["quantization_steps"],
            lagrange_proportional=config["sequential_params"]["lagrange_proportional"],
            block_size=config["sequential_params"]["block_size"],
        )

        exp_config = ExperimentConfig(
            metadata=metadata,
            pointcloud=pointcloud,
            sequential_params=sequential_params,
            rewrite_results=config.get("rewrite_results", False),
            debugging=config.get("debugging", False),
        )

        logger.info(f"Configuration loaded successfully: {exp_config}")
        return exp_config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

# ------------------- Example Usage -------------------


if __name__ == "__main__":
    yaml_file = "experiment_config.yaml"
    exp_config = load_experiment_config(yaml_file)
    print(exp_config)
