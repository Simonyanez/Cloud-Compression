import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def generate_gif_vectors(run_id, folder_path, output_path):
    root = Path(folder_path)
    # Find all iteration files for this run
    iter_files = sorted(list(root.glob(f"{run_id}_iter_*.json")), 
                       key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    
    if not iter_files:
        print(f"[!] No files found for {run_id} in {folder_path}")
        return

    history = []
    for it_f in iter_files:
        with open(it_f, 'r') as f:
            d = json.load(f)
        history.append(np.array(d.get('slopes', [])))
    
    history = np.array(history)
    num_iters, num_clusters, _ = history.shape
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate axis limits for stability
    min_val = np.min(history)
    max_val = np.max(history)
    
    colors = plt.cm.get_cmap('tab10', num_clusters)

    def update(frame):
        ax.clear()
        # Fix limits
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_zlim(min_val, max_val)
        ax.set_xlabel('Slope X')
        ax.set_ylabel('Slope Y')
        ax.set_zlabel('Slope Z')
        ax.set_title(f'RD-Clustering Slope Vectors: {run_id}\nIteration: {frame}', fontsize=14)
        
        # Current slopes
        current_slopes = history[frame]
        
        # Draw 3D vectors (quiver)
        for c in range(num_clusters):
            # Skip DC (cluster 0) if it's always [0,0,0]
            if c == 0 and np.allclose(current_slopes[c], 0):
                continue
                
            ax.quiver(0, 0, 0, 
                      current_slopes[c, 0], current_slopes[c, 1], current_slopes[c, 2],
                      color=colors(c), arrow_length_ratio=0.1, linewidth=2, label=f'Cluster {c}')
        
        ax.legend(loc='upper left', fontsize='small', ncol=2)

    print(f"[*] Animating {run_id} as GIF (Vectors)...")
    # interval=500ms for 0.5s per iteration
    anim = FuncAnimation(fig, update, frames=num_iters, interval=500, blit=False)
    
    anim.save(output_path, writer='pillow')
    plt.close()

if __name__ == "__main__":
    res_dir = Path("res")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Source mapping
    configs = [
        ("B4_C4_Q24", "fast_validated_sweep_20260524_235753"),
        ("B4_C6_Q24", "afternoon_validated_sweep_20260525_152602"),
        ("B4_C8_Q24", "afternoon_validated_sweep_20260525_152602"),
        ("B4_C10_Q24", "afternoon_validated_sweep_20260525_152602")
    ]
    
    for rid, folder in configs:
        out = res_dir / f"slopes_vectors_{rid}.gif"
        generate_gif_vectors(rid, folder, out)
        print(f"[+] Saved: {out}")
