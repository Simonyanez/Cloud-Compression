import matplotlib.pyplot as plt
import pandas as pd
from utils.bj_delta import *

def extract_info(subdata):
    nPSNR_Y = subdata['nPSNR_Y']
    nbpv = subdata['Structural bpv']
    dPSNR_Y = subdata['dPSNR_Y']
    dbpv = subdata['Dynamic bpv']
    bd_psnr = bj_delta(nbpv, nPSNR_Y, dbpv, dPSNR_Y, mode=0)
    bd_rate = bj_delta(nbpv, nPSNR_Y, dbpv, dPSNR_Y,  mode=1)
    return nPSNR_Y, nbpv, dPSNR_Y, dbpv,bd_psnr,bd_rate
    
def PSNR_vs_bpv_plot(data):
    bsizes = [4, 8, 16]
    num_of_points = [1, 2, 4]
    markers = ['o', 's', '^']  # Different markers for each block size
    colors = ['blue', 'green', 'red']  # Different colors for each block size

    for num in num_of_points:
        plt.figure(figsize=(10, 6))
        for i, bsize in enumerate(bsizes):
            block_data = data[data['Block Size'] == bsize]
            blockpoint_data = block_data[block_data['Point Fraction'] == num]
            nPSNR_Y, nbpv, dPSNR_Y, dbpv, bd_psnr,bd_rate = extract_info(blockpoint_data)
            print(f"For size block {bsize} and {num} number of points:\n Bjontegaard Gain: {bd_psnr,bd_rate}")
            # Optionally plot dPSNR_Y vs. dbpv if available
            if len(dPSNR_Y) > 0 and len(dbpv) > 0:
                plt.plot(dbpv, dPSNR_Y, marker=markers[i], color=colors[i], linestyle='--', label=f'Dynamic - Block Size {bsize}', alpha=0.7)
            # Plot PSNR vs. bpv
            plt.plot(nbpv, nPSNR_Y, marker=markers[i], color=colors[i], linestyle='-', label=f'Structural - Block Size {bsize}', alpha=0.3)
        
        # Plot customization
        plt.title(f'PSNR vs. bpv for {num} self-looped nodes - single frame')
        plt.xlabel('bpv')
        plt.ylabel('PSNR_Y')
        plt.grid(True)
        plt.legend(title='Legend')
        plt.ylim(20, 45)  # Adjust according to your data range
        plt.xlim(0, 12)   # Adjust according to your data range
        plt.show()

if __name__ == "__main__":
    data = pd.read_csv('PSNR_experiment.csv')
    PSNR_vs_bpv_plot(data)