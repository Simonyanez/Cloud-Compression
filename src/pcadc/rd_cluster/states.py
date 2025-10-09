from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class RDClusterState:
    """Encapsulates the state of RD clustering at any iteration"""
    labels: np.ndarray          # (N_blocks,) cluster assignment per block
    slopes: np.ndarray          # (N_clusters, slope_dim) cluster centroids
    lambda_value: float         # Current lambda for RD tradeoff
    lambda_step: int            # Current position in lambda schedule
    iteration: int              # Iteration counter
    total_cost: Optional[float] = None  # Total RD cost (if tracked)


@dataclass
class RDBlockCost:
    """Rate-distortion cost for a single block-cluster assignment"""
    cluster_id: int
    rate: float
    distortion: float
    cost: float  # rate + lambda * distortion


@dataclass
class ClusteringHistory:
    """Tracks clustering evolution over iterations"""
    states: List[RDClusterState]
    convergence_metrics: Dict[str, List[float]]  # e.g., {"label_changes": [...], "total_cost": [...]}
