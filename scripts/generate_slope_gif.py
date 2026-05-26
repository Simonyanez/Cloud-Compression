import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def generate_gif(run_id, folder_path, output_path):
    root = Path(folder_path)
    # Find all iteration files for this run
    iter_files = sorted(list(root.glob(f"{run_id}_iter_*.json")), 
                       key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    
    if not iter_files:
        print(f"[!] No files found for {run_id}")
        return

    # Extract all slopes history
    # shape: (num_iters, num_clusters, 3)
    history = []
    for it_f in iter_files:
        with open(it_f, 'r') as f:
            d = json.load(f)
        slopes = np.array(d.get('self_loop_weights', [])) # Use weights or slopes? 
        # User asked for "slopes" evolution. Slopes are the 3D vectors.
        slopes = np.array(d.get('slopes', []))
        history.append(slopes)
    
    history = np.array(history)
    num_iters, num_clusters, _ = history.shape
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate axis limits
    min_x, max_x = np.min(history[:, :, 0]), np.max(history[:, :, 0])
    min_y, max_y = np.min(history[:, :, 1]), np.max(history[:, :, 1])
    min_z, max_z = np.min(history[:, :, 2]), np.max(history[:, :, 2])
    
    # Add padding
    pad = 0.1
    ax.set_xlim(min_x - abs(min_x)*pad, max_x + abs(max_x)*pad)
    ax.set_ylim(min_y - abs(min_y)*pad, max_y + abs(max_y)*pad)
    ax.set_zlim(min_z - abs(min_z)*pad, max_z + abs(max_z)*pad)
    
    # Distinct colors for clusters
    colors = plt.cm.get_cmap('tab10', num_clusters)
    
    scatters = []
    lines = []
    
    # Initialize objects
    for c in range(num_clusters):
        # Scatter for the current point
        scat = ax.scatter([], [], [], color=colors(c), s=100, edgecolors='black', label=f'Cluster {c}')
        scatters.append(scat)
        # Line for the trajectory
        line, = ax.plot([], [], [], color=colors(c), alpha=0.5, linewidth=2)
        lines.append(line)
        
    ax.set_title(f'Evolution of Cluster Slopes (3D)\nRun: {run_id}', fontsize=14)
    ax.set_xlabel('Slope X')
    ax.set_ylabel('Slope Y')
    ax.set_zlabel('Slope Z')
    
    # Add a legend only for active clusters (exclude DC if c=0 and slope is 0)
    ax.legend(loc='upper left', fontsize='small', ncol=2)

    def update(frame):
        for c in range(num_clusters):
            # Current iteration point
            current_slope = history[frame, c, :]
            scatters[c]._offsets3d = (np.array([current_slope[0]]), np.array([current_slope[1]]), np.array([current_slope[2]]))
            
            # Trajectory up to current point
            traj = history[:frame+1, c, :]
            lines[c].set_data(traj[:, 0], traj[:, 1])
            lines[c].set_3d_properties(traj[:, 2])
            
        return scatters + lines

    print(f"[*] Animating {run_id} ({num_iters} frames)...")
    anim = FuncAnimation(fig, update, frames=num_iters, interval=200, blit=False)
    
    anim.save(output_path, writer='pillow')
    plt.close()

if __name__ == "__main__":
    folder = "afternoon_validated_sweep_20260525_152602"
    runs = ["B4_C6_Q24", "B4_C8_Q24", "B4_C10_Q24"]
    
    for r in runs:
        out = Path("docs/thesis/img/resultados") / f"slopes_evo_{r}.gif"
        generate_gif(r, folder, out)
        print(f"[+] Saved: {out}")
