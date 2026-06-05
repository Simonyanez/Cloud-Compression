import numpy as np
import os
from typing import List, Callable, Tuple, Optional
from pcadc.blocks import Block
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

# --- TOP LEVEL WORKERS (Required for Process Pickling) ---

def _evaluate_rd_cost_standalone(block: Block, params: np.ndarray, cluster_idx: int, 
                                 q_step: int, decider_mode: str, lagrange_proportional: float):
    """Standalone evaluator that doesn't need 'self'."""
    from pcadc.decider import Decider
    from pcadc.graph import StructuralGraph, AttributeGraph
    from pcadc.transforms import GFTStrategyWraper
    from pcadc.color import Approximator

    slope = params[:3]
    slw = params[3]
    slp = params[4]
    
    decider = Decider(decider_mode, lagrange_proportional)
    decider._set_vars(q_step)
    gft_computer = GFTStrategyWraper()

    structural_graph = StructuralGraph(block.metadata)
    structural_graph.set_data(block.Vblock)
    attribute_graph = AttributeGraph(structural_graph, slope, cluster_idx, slp, slw)
    
    Vblock_rotated = Approximator()._spatial_norm(block.Vblock)
    Ablock_app = block.Ablock.copy()
    Ablock_app[:, 0] = Vblock_rotated @ slope.T
    attribute_graph.set_data(block.Vblock, Ablock_app)
    
    _, coeffs = gft_computer(block, attribute_graph)
    attribute_graph.clear_data()
    
    rd_cost, _, _ = decider._RDcost(coeffs)
    return rd_cost

def _cluster_objective_func(params, cluster_indices, blocks, vertices, attributes, 
                            q_step, decider_mode, lagrange_proportional, k):
    total_cost = 0.0
    for idx in cluster_indices:
        total_cost += _evaluate_rd_cost_standalone(blocks[idx], params, k, q_step, decider_mode, lagrange_proportional)
    return total_cost

def _worker_optimize_cluster(k, labels, blocks, vertices, attributes, old_slopes, old_slw, old_slp, 
                             q_step, decider_mode, lagrange_proportional, maxiter, popsize, polish, tol, mode):
    if k == 0:
        return 0, old_slopes[0], old_slw[0], old_slp[0]

    cluster_mask = (labels == k)
    cluster_indices = np.where(cluster_mask)[0]
    
    if len(cluster_indices) == 0:
        return k, old_slopes[k], old_slw[k], old_slp[k]
        
    print(f"[Cluster {k}] Starting {mode.upper()} Optimization (n={len(cluster_indices)} blocks)")
    
    for idx in cluster_indices:
        blocks[idx].init_data(vertices, attributes)

    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-20.0, 20.0), (0.01, 10.0), (0.01, 0.99)]
    
    from multiprocessing.pool import ThreadPool
    # Use a ThreadPool to fill up the remaining CPU cores during the population evaluation.
    # Since we are already in a Process, a ThreadPool here is very efficient for NumPy math.
    with ThreadPool(processes=os.cpu_count()) as pool:
        res = differential_evolution(
            _cluster_objective_func, 
            bounds=bounds,
            args=(cluster_indices, blocks, vertices, attributes, q_step, decider_mode, lagrange_proportional, k),
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0), 
            workers=pool.map, 
            updating="deferred",
            recombination=0.7,
            polish=polish,
            tol=tol,
            disp=False
        )
    
    for idx in cluster_indices:
        blocks[idx].clear_data()
        
    return k, res.x[:3], res.x[3], res.x[4]

class SlopeOptimizer:
    
    def __init__(self, add_intercept: bool = True, mode: str = "production"):
        self.add_intercept = add_intercept
        self.mode = mode.lower()
        
        if self.mode == "draft":
            self.maxiter = 2
            self.popsize = 2
            self.polish = False
            self.tol = 0.5
        else: # Production
            self.maxiter = 15 
            self.popsize = 10 
            self.polish = True 
            self.tol = 1e-3
    
    def recalculate_slopes(self, blocks: List[Block], labels: np.ndarray,
                           vertices: np.ndarray, attributes: np.ndarray,
                           num_clusters: int, old_slopes: np.ndarray,
                           old_slw: np.ndarray, old_slp: np.ndarray,
                           q_step: int, decider_mode: str, lagrange_proportional: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Parallelize across clusters using PROCESSES
        results = Parallel(n_jobs=-1)(
            delayed(_worker_optimize_cluster)(
                k, labels, blocks, vertices, attributes, old_slopes, old_slw, old_slp,
                q_step, decider_mode, lagrange_proportional,
                self.maxiter, self.popsize, self.polish, self.tol, self.mode
            ) for k in range(num_clusters)
        )
        
        new_slopes = old_slopes.copy()
        new_slw = old_slw.copy()
        new_slp = old_slp.copy()
        
        for k, slope, slw, slp in results:
            new_slopes[k] = slope
            new_slw[k] = slw
            new_slp[k] = slp
            
        return new_slopes, new_slw, new_slp
