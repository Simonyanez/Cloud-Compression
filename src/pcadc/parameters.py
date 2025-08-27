from .io import *
from dataclasses import dataclass
from pathlib import Path
from typing import List
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
class SequentialParameter:
    """
    Parameters for a single experiment run (pipeline).
    """
    point_cloud_path: Path
    number_of_clusters: int
    self_loop_weight: float
    self_loop_threshold: float
    quantization_steps: List[int]
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
    sequential_params: SequentialParameter
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
            f"Self-loop Weight: {self.sequential_params.self_loop_weight}\n"
            f"Self-loop Threshold: {self.sequential_params.self_loop_threshold}\n"
            f"Quantization Steps: {self.sequential_params.quantization_steps}\n"
            f"Block Size: {self.sequential_params.block_size}\n"
            f"Rewrite Results: {self.rewrite_results}\n"
            f"Debugging: {self.debugging}\n"
        )

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
            self_loop_weight=config["sequential_params"]["self_loop_weight"],
            self_loop_threshold=config["sequential_params"]["self_loop_threshold"],
            quantization_steps=config["sequential_params"]["quantization_steps"],
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
