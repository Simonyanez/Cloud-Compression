import numpy as np
from typing import List, Callable, Tuple
from pcadc.blocks import Block
from scipy.optimize import differential_evolution

class SlopeOptimizer:
    
    def __init__(self, add_intercept: bool = True, mode: str = "production"):
        self.add_intercept = add_intercept
        self.mode = mode.lower()
        # Bumping this up slightly to give the coarse search room to breathe
        self.max_nm_evals = 20
        
        # Configure optimizer settings based on mode
        if self.mode == "draft":
            self.maxiter = 2
            self.popsize = 2
            self.polish = False
            self.tol = 0.5
        else: # Production
            self.maxiter = 15 # Deep generational limits
            self.popsize = 10 # Large search population (10 * 5 = 50 candidates)
            self.polish = True # Enable local gradient fine-tuning
            self.tol = 1e-3
    
    def recalculate_slopes(self, blocks: List[Block], labels: np.ndarray,
                           vertices: np.ndarray, attributes: np.ndarray,
                           num_clusters: int, old_slopes: np.ndarray,
                           old_slw: np.ndarray, old_slp: np.ndarray,
                           rd_cost_fn: Callable[[Block, np.ndarray, int], float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        new_slopes = old_slopes.copy()
        new_slw = old_slw.copy()
        new_slp = old_slp.copy()
        
        for k in range(num_clusters):
            if k == 0:
                continue # Skip DC cluster

            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            print(f"\n[Cluster {k}] Starting {self.mode.upper()} Optimization (n={len(cluster_indices)} blocks)")
            print(f"[Cluster {k}] Initial Guess Slope: {old_slopes[k]}, SLW: {old_slw[k]}, SLP: {old_slp[k]}")
            
            eval_counter = 0
            
            # --- Objective Function ---
            def objective(params):
                nonlocal eval_counter
                total_cost = 0.0
                for idx in cluster_indices:
                    blocks[idx].init_data(vertices, attributes)
                    total_cost += rd_cost_fn(blocks[idx], params, k)
                    blocks[idx].clear_data()
                
                eval_counter += 1
                return total_cost

            # Bounds for [slope_x, slope_y, slope_z, slw, slp]
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-20.0, 20.0), (0.01, 10.0), (0.01, 0.99)]
            
            res = differential_evolution(
                objective, 
                bounds=bounds,
                maxiter=self.maxiter,
                popsize=self.popsize,
                mutation=(0.5, 1.0), 
                recombination=0.7,
                polish=self.polish,
                tol=self.tol,
                disp=False
            )
            print(f"[Cluster {k}] {self.mode.upper()} Optimization Finished | Success: {res.success} | Status: {res.message}")
            print(f"[Cluster {k}] Final Adopted Params: {res.x}\n")
            
            new_slopes[k] = res.x[:3]
            new_slw[k] = res.x[3]
            new_slp[k] = res.x[4]
            
        return new_slopes, new_slw, new_slp
