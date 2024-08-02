import pandas as pd
import matplotlib.pyplot as plt

def extract_info(subdata):
    PSNR_Y = subdata['PSNR_Y']
    bpv = subdata['Structural bpv']
    dPSNR_Y = subdata['dPSNR_Y']
    dbpv = subdata['Dynamic bpv']
    return PSNR_Y,bpv,dPSNR_Y,dbpv

def PSNR_vs_bpv_plot(data):
    bsizes = [4,8,16]
    markers = ['o', 's', '^']  # Different markers for each block size
    colors = ['blue', 'green', 'red']  # Different colors for each block size
    
    plt.figure(figsize=(10, 6))

    for i, bsize in enumerate(bsizes):
        block_data = data[data['Block Size'] == bsize]
        PSNR_Y, bpv, dPSNR_Y, dbpv = extract_info(block_data)

        # Plot PSNR vs. bpv
        plt.plot(bpv, PSNR_Y, marker=markers[i], color=colors[i], linestyle='-', label=f'Structural - Block Size {bsize}', alpha=0.3)
        # Optionally plot dPSNR_Y vs. dbpv if available
        if len(dPSNR_Y) > 0 and len(dbpv) > 0:
            plt.plot(dbpv, dPSNR_Y, marker=markers[i], color=colors[i], linestyle='--', label=f'Dynamic - Block Size {bsize}',alpha=0.7)

    # Plot customization
    plt.title('PSNR vs. bpv for Different Block Sizes')
    plt.xlabel('bpv_Y')
    plt.ylabel('PSNR_Y')
    plt.grid(True)
    plt.legend(title='Legend')
    plt.ylim(30, 60)  # Adjust according to your data range
    plt.xlim(0, 12)   # Adjust according to your data range
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('results.csv')
    PSNR_vs_bpv_plot(data)