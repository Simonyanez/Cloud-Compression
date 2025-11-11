from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from pcadc.parameters import SequentialParameters
from pcadc.decider import Decider

@dataclass
class RDClusterState:
    """Encapsulates the state of RD clustering at any iteration"""
    labels: np.ndarray          # (N_blocks,) cluster assignment per block
    slopes: np.ndarray          # (N_clusters, 3) cluster centroids
    qstep_value: int         # Current lambda for RD tradeoff
    lambda_step: int            # Current position in lambda schedule
    iteration: int              # Iteration counter
    total_cost: Optional[float] = None  # Total RD cost (if tracked)
    
    # New metrics
    cluster_entropy: Optional[float] = None
    avg_rate: Optional[float] = None
    avg_distortion: Optional[float] = None
    cluster_gains: Optional[Dict[int, Dict[str, float]]] = None # Richer gain stats per dynamic cluster

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RDClusterState to a dictionary for JSON serialization."""
        # Convert numpy int types in cluster_gains to standard python int
        serializable_gains = {}
        if self.cluster_gains:
            for k, stats in self.cluster_gains.items():
                serializable_gains[k] = {key: int(val) if 'blocks' in key else val for key, val in stats.items()}

        return {
            "labels": self.labels.tolist(),
            "slopes": self.slopes.tolist(),
            "qstep_value": self.qstep_value,
            "lambda_step": self.lambda_step,
            "iteration": self.iteration,
            "total_cost": self.total_cost,
            "cluster_entropy": self.cluster_entropy,
            "avg_rate": self.avg_rate,
            "avg_distortion": self.avg_distortion,
            "cluster_gains": serializable_gains,
        }

    def __repr__(self) -> str:
        n_blocks = len(self.labels)
        n_clusters = len(self.slopes)
        
        # Get cluster distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        n_active = len(unique_labels)
        n_empty = n_clusters - n_active
        
        # Statistics on cluster sizes
        min_size = counts.min() if len(counts) > 0 else 0
        max_size = counts.max() if len(counts) > 0 else 0
        avg_size = counts.mean() if len(counts) > 0 else 0
        
        cost_str = f"{self.total_cost:.6f}" if self.total_cost is not None else "None"
        
        # Format cluster distribution (show counts for each cluster)
        if n_active <= 10:
            # Show full distribution for small number of clusters
            dist_str = ", ".join([f"{label}:{count}" for label, count in zip(unique_labels, counts)])
            dist_info = f"dist=[{dist_str}]"
        else:
            # Show summary for many clusters
            dist_info = f"sizes=[min:{min_size}, avg:{avg_size:.1f}, max:{max_size}]"
        
        # Format slopes information
        slopes_info = self._format_slopes(unique_labels, counts)

        # New metrics info
        metrics_info = ""
        if self.cluster_entropy is not None:
            metrics_info += f"\n  Entropy: {self.cluster_entropy:.4f}"
        if self.avg_rate is not None:
            metrics_info += f", Avg Rate: {self.avg_rate:.4f}"
        if self.avg_distortion is not None:
            metrics_info += f", Avg Dist: {self.avg_distortion:.4f}"
        
        gains_info = ""
        if self.cluster_gains is not None and len(self.cluster_gains) > 0:
            gains_info += "\n  Gains (vs structural):"
            for k, stats in self.cluster_gains.items():
                gains_info += (
                    f"\n    Cluster {k}: avg={stats['avg_gain']:.2f} "
                    f"| #pos={stats['positive_gain_blocks']} "
                    f"#neg={stats['negative_gain_blocks']} "
                    f"| min={stats['min_gain']:.2f} "
                    f"max={stats['max_gain']:.2f}"
                )

        return (
            f"RDClusterState(\n"
            f"  iter={self.iteration}, qstep={self.qstep_value}, λ_step={self.lambda_step}\n"
            f"  blocks={n_blocks}, clusters={n_clusters} (active:{n_active}, empty:{n_empty})\n"
            f"  {dist_info}\n"
            f"  cost={cost_str}"
            f"{metrics_info}"
            f"{gains_info}\n"
            f"{slopes_info}"
            f")"
        )
    
    def _format_slopes(self, unique_labels: np.ndarray, counts: np.ndarray) -> str:
        """Format slope information for each cluster"""
        lines = ["  Cluster Slopes (3D):"]
        
        # Create a mapping from label to count for quick lookup
        label_to_count = dict(zip(unique_labels, counts))
        
        # Iterate through all clusters in order
        for cluster_id in range(len(self.slopes)):
            slope = self.slopes[cluster_id]
            slope_str = f"[{slope[0]:7.4f}, {slope[1]:7.4f}, {slope[2]:7.4f}]"
            
            if cluster_id in label_to_count:
                # Active cluster
                count = label_to_count[cluster_id]
                lines.append(f"    Cluster {cluster_id:2d} (n={count:4d}): {slope_str}")
            else:
                # Empty cluster
                lines.append(f"    Cluster {cluster_id:2d} (n=   0): {slope_str} [EMPTY]")
        
        return "\n".join(lines) + "\n"
    
    def is_last(self, sequential_parameters):
        last_qstep = sequential_parameters.quantization_steps[-1]
        return self.qstep_value == last_qstep

@dataclass
class RDBlockCost:
    """Rate-distortion cost for a single block-cluster assignment"""
    cluster_id: int
    rate: float
    distortion: float
    cost: float  # rate + lambda * distortion
    
    def __repr__(self) -> str:
        # Calculate lambda from the equation: cost = rate + lambda * distortion
        lambda_val = (self.cost - self.rate) / self.distortion if self.distortion != 0 else 0
        
        return (
            f"RDBlockCost(cluster={self.cluster_id}, "
            f"R={self.rate:.4f}, "
            f"D={self.distortion:.4f}, "
            f"cost={self.cost:.4f}, "
            f"λ={lambda_val:.2f})"
        )


@dataclass
class ClusteringHistory:
    """Tracks clustering evolution over iterations"""
    states: List[RDClusterState] = field(default_factory=list)
    
    def __repr__(self) -> str:
        n_states = len(self.states)
        if n_states == 0:
            return "ClusteringHistory(empty)"
        
        first_state = self.states[0]
        last_state = self.states[-1]
        
        # Cost progression
        cost_info = ""
        if last_state.total_cost is not None and first_state.total_cost is not None:
            cost_change = last_state.total_cost - first_state.total_cost
            pct_change = (cost_change / first_state.total_cost) * 100
            cost_info = f"\n  cost: {first_state.total_cost:.6f} → {last_state.total_cost:.6f} (Δ={cost_change:+.6f}, {pct_change:+.2f}%)"
        
        # Cluster evolution
        first_active = len(np.unique(first_state.labels))
        last_active = len(np.unique(last_state.labels))
        cluster_change = last_active - first_active
        
        # Check convergence
        is_converged = self.get_cost_stable(num_of_iters=min(5, n_states - 1)) if n_states > 1 else False
        convergence_str = " [CONVERGED]" if is_converged else ""
        
        return (
            f"ClusteringHistory({n_states} states{convergence_str})\n"
            f"  iterations: {first_state.iteration} → {last_state.iteration}\n"
            f"  qstep: {first_state.qstep_value} → {last_state.qstep_value}\n"
            f"  active_clusters: {first_active} → {last_active} (Δ={cluster_change:+d})"
            f"{cost_info}"
        )
    
    def add_state(self, state: RDClusterState):
        """Add a new state to history"""
        self.states.append(state)
    
    def get_cost_stable(self, num_of_iters: int, threshold: float = 1e-3) -> bool:
        """
        Check if cost has been stable for the last num_of_iters iterations
        
        Args:
            num_of_iters: Number of iterations to check
            threshold: Maximum relative change allowed
        
        Returns:
            True if cost changes are below threshold for all checked iterations
        """
        if len(self.states) < num_of_iters + 1:
            return False
        
        recent_states = self.states[-(num_of_iters + 1):]
        costs = [state.total_cost for state in recent_states]
        
        if any(c is None for c in costs):
            return False
        costs = np.array(costs)
        diffs = np.diff(costs)  # [cost[1]-cost[0], cost[2]-cost[1], ...]
        relative_changes = np.abs(diffs) / (np.abs(costs[:-1]) + 1e-10)
        return np.all(relative_changes < threshold)
