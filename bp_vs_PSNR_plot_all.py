import matplotlib.pyplot as plt
import pandas as pd
from utils.bj_delta import *

def extract_info(subdata):
    """
    Extract PSNR and bpv data from subdata and calculate Bjontegaard Delta metrics.
    """
    nPSNR_Y = subdata['N PSNR All'].values
    print(nPSNR_Y)
    nbpv = subdata['Average N Bits'].values
    dPSNR_Y = subdata['D PSNR All'].values
    dbpv = subdata['Average D Bits'].values
    bd_psnr = bj_delta(nbpv, nPSNR_Y, dbpv, dPSNR_Y, mode=0)
    bd_rate = bj_delta(nbpv, nPSNR_Y, dbpv, dPSNR_Y, mode=1)
    return nPSNR_Y, nbpv, dPSNR_Y, dbpv, bd_psnr, bd_rate

def PSNR_vs_bpv_plot(data):
    """
    Plot PSNR vs. bpv for different block sizes and point fractions.
    """
    bsizes = [4, 8, 16]
    num_of_points = [1, 2, 4, 8, 16]
    markers = ['o', 's', '^']  # Different markers for each block size
    colors = ['blue', 'green', 'red']  # Different colors for each block size

    for num in num_of_points:
        plt.figure(figsize=(10, 6))
        for i, bsize in enumerate(bsizes):
            # Filter data based on block size
            block_data = data[data['Block Size'] == bsize]
            if block_data.empty:
                print(f"No data for Block Size {bsize}")
                continue
            
            # Extract information
            nPSNR_Y, nbpv, dPSNR_Y, dbpv, bd_psnr, bd_rate = extract_info(block_data)
            print(f"For Block Size {bsize} and Point Fraction {num}:")
            print(f"  Bjontegaard PSNR Gain: {bd_psnr}")
            print(f"  Bjontegaard Rate Gain: {bd_rate}")

            # Plot Dynamic and Structural metrics
            if len(dPSNR_Y) > 0 and len(dbpv) > 0:
                plt.plot(dbpv, dPSNR_Y, marker=markers[i], color=colors[i], linestyle='--', label=f'Dynamic - Block Size {bsize}', alpha=0.7)
            plt.plot(nbpv, nPSNR_Y, marker=markers[i], color=colors[i], linestyle='-', label=f'Structural - Block Size {bsize}', alpha=0.3)
        
        # Customize and show plot
        plt.title(f'PSNR vs. bpv - 10 Frames')
        plt.xlabel('bpv')
        plt.ylabel('PSNR_Y')
        plt.grid(True)
        plt.legend(title='Legend')
        plt.ylim(20, 45)  # Adjust according to your data range
        plt.xlim(0, 12)   # Adjust according to your data range
        plt.show()

if __name__ == "__main__":
    # Load data from CSV
    data = pd.read_csv('results.csv')
    # Generate and display plots
    PSNR_vs_bpv_plot(data)