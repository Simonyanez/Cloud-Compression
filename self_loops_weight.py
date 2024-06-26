from src.encoder import *
import matplotlib.pyplot as plt
import imageio


V,C_rgb,_ = ply.ply_read8i("res/longdress_vox10_1051.ply")  

structural_encoder = StructuralEncoder(V,C_rgb)
directional_encoder = DirectionalEncoder(V,C_rgb)
structural_encoder.block_indexes(block_size = 16)
directional_encoder.block_indexes(block_size = 16)

sorted_indexes = directional_encoder.std_sorted_indexes()

# Find borders by structural graph
trial = sorted_indexes[-100]
W, border_positions, border_fig, y_values = directional_encoder.find_borders(trial)
c_values = np.arange(0.1,6.0,0.1)
# Add weights to specific border
choosed_positions = [39,42,45,43,91,92,95,96,97,110,111,112,115,116,118,125,128]
first_component_results = {'Structural':[],'Self-Looped':[]}
total_energy = {'Structural':[],'Self-Looped':[]}


for c in c_values:
    c=round(c,2)
    choosed_weights = [c]*len(choosed_positions)
    idx_map = dict(zip(choosed_positions,choosed_weights))
    GFT, Gfreq, Ablockhat = directional_encoder.gft_transform(trial,W,idx_map)
    nGFT, _, nAblockhat = directional_encoder.gft_transform(trial,W,None)
    
    print(f'This is the sizee of the transformed block {np.shape(Ablockhat)}')
    energy_fig = directional_encoder.energy_block(Ablockhat,"self-looped",c)
    #energy_fig_2 = directional_encoder.energy_block(nAblockhat,"strctural")
    energy_fig.savefig(f'res/c_exp_frames/self_looped_energy_{c}.png')
    plt.close(energy_fig)
    
    first_component_results["Structural"].append(nAblockhat[0,0])
    first_component_results["Self-Looped"].append(Ablockhat[0,0])

    base_fig= directional_encoder.component_projection(trial,GFT[:,0],"self-looped",c)
    #nbase_fig = directional_encoder.component_projection(trial,nGFT[:,0],"structural",y_values)
    print(f"{np.min(GFT[:,0])},{np.max(GFT[:,0])}")
    print(f"{np.min(nGFT[:,0])},{np.max(nGFT[:,0])}")
    base_fig.savefig(f'res/c_exp_frames/self_looped_{c}.png')

    #total_energy["Structural"].append(nAblockhat[:,0])
    #total_energy["Self-Looped"].append(Ablockhat[:,0])
    plt.close(base_fig)

frames = []
frames2 = []
for c in c_values:
    c=round(c,2)
    frame_path = f'res/c_exp_frames/self_looped_{str(c)}.png'
    frames.append(imageio.imread(frame_path))
    frame_path_2 = f'res/c_exp_frames/self_looped_energy_{str(c)}.png'
    frames2.append(imageio.imread(frame_path_2))

imageio.mimsave(f'res/c_experiment.gif',frames,duration=4.0)
imageio.mimsave(f'res/c_experiment_2.gif',frames2,duration=10.0)
# Create a figure
fig, ax = plt.subplots()

# Plot the first set of data
ax.plot(c_values, first_component_results["Structural"], label='Structural', color='blue', linestyle='-', marker='o')

# Plot the second set of data
ax.plot(c_values, first_component_results["Self-Looped"], label='Self-Looped', color='red', linestyle='--', marker='x')

# Add a title
ax.set_title('Structural and Self-Looped method DC component over weight value')

# Add labels to the axes
ax.set_xlabel('Self-loop Weight')
ax.set_ylabel('Absolute Value')

# Add a legend
ax.legend()

# # Bar plot settings
# bar_width = 0.35
# index = np.arange(num_groups)

# # Plotting
# fig2, ax2 = plt.subplots()

# bar1 = ax2.bar(index, total_energy["Structural"], bar_width, label='Structural')
# bar2 = ax2.bar(index + bar_width, total_energy["Self-Looped"], bar_width, label='Self-Looped')

# # Add labels, title, and legend
# ax2.set_xlabel('Samples')
# ax2.set_ylabel('Total Energy')
# ax2.set_title('Comparison of Total Energy: Structural vs. Self-Looped')
# ax2.set_xticks(index + bar_width / 2)
# ax2.set_xticklabels([str(i) for i in range(num_groups)])
# ax2.legend()

# # Adding error bars
# ax2.errorbar(index, total_energy["Structural"], yerr=std_structural, fmt='none', capsize=5, label='Structural Std Dev')
# ax2.errorbar(index + bar_width, total_energy["Self-Looped"], yerr=std_selflooped, fmt='none', capsize=5, label='Self-Looped Std Dev')

# plt.tight_layout()
plt.show()
# # Create a figure
# fig2, ax2 = plt.subplots()

# # Plot the first set of data
# ax2.plot(c_values, total_energy["Structural"], label='Structural', color='blue', linestyle='-', marker='o')

# # Plot the second set of data
# ax2.plot(c_values, total_energy["Self-Looped"], label='Self-Looped', color='red', linestyle='--', marker='x')

# # Add a title
# ax2.set_title('Plot of Structural and Self-Looped method first component energy over weight value')

# # Add labels to the axes
# ax2.set_xlabel('Self-loop Weight')
# ax2.set_ylabel('Energy')

# # Add a legend
# ax2.legend()
plt.show()

