import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def get_rd_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT bsize, q_step, point_count, struct_rate, struct_dist, oracle_rate, oracle_dist, best_p, best_w, gain FROM results", conn)
    conn.close()
    
    df['struct_sse'] = df['struct_dist']**2
    df['oracle_sse'] = df['oracle_dist']**2
    
    grouped = df.groupby(['bsize', 'q_step']).agg({
        'point_count': 'sum',
        'struct_rate': 'sum',
        'struct_sse': 'sum',
        'oracle_rate': 'sum',
        'oracle_sse': 'sum',
        'gain': 'mean'
    }).reset_index()
    
    grouped['R_s'] = grouped['struct_rate'] / grouped['point_count']
    grouped['R_o'] = grouped['oracle_rate'] / grouped['point_count']
    grouped['PSNR_s'] = 10 * np.log10(255**2 / (grouped['struct_sse'] / grouped['point_count']))
    grouped['PSNR_o'] = 10 * np.log10(255**2 / (grouped['oracle_sse'] / grouped['point_count']))
    
    return grouped, df

def main():
    abs_db = "results/oracle_upper_bound.db"
    lin_db = "results/linear_oracle_upper_bound.db"
    
    abs_rd, abs_full = get_rd_data(abs_db)
    lin_rd, lin_full = get_rd_data(lin_db)
    
    block_sizes = sorted(abs_rd['bsize'].unique())
    
    fig, axes = plt.subplots(1, len(block_sizes), figsize=(18, 6), sharey=False)
    if len(block_sizes) == 1: axes = [axes]

    for i, bsize in enumerate(block_sizes):
        d_abs = abs_rd[abs_rd['bsize'] == bsize].sort_values('q_step')
        d_lin = lin_rd[lin_rd['bsize'] == bsize].sort_values('q_step')
        
        ax = axes[i]
        # Structural Baseline (using abs_rd as they should be identical)
        ax.plot(d_abs['R_s'], d_abs['PSNR_s'], 'o--', label='Structural (Baseline)', color='gray', alpha=0.8)
        
        # Absolute Oracle
        ax.plot(d_abs['R_o'], d_abs['PSNR_o'], 's-', label='Absolute Oracle', color='blue')
        
        # Linear Oracle
        ax.plot(d_lin['R_o'], d_lin['PSNR_o'], '^-', label='Linear Oracle', color='red')
        
        ax.set_title(f'RD Curve - Block Size {bsize}')
        ax.set_xlabel('Rate (bits per point)')
        ax.set_ylabel('PSNR_Y (dB)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/rd_curves_comparison.png', dpi=300)
    print("Saved RD curves to results/rd_curves_comparison.png")

    # Second Plot: Gain Distribution / Parameters
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gain distribution comparison
    sns.kdeplot(abs_full['gain'], label='Absolute Oracle Gain', ax=axes[0], fill=True, color='blue')
    sns.kdeplot(lin_full['gain'], label='Linear Oracle Gain', ax=axes[0], fill=True, color='red')
    axes[0].set_title('RD-Cost Gain Distribution (Structural - Oracle)')
    axes[0].set_xlabel('Gain (Lower is better for cost)')
    axes[0].legend()

    # Best Percentage Distribution (p)
    sns.histplot(abs_full['best_p'], label='Abs Oracle p', ax=axes[1], color='blue', alpha=0.5, bins=20, stat='probability')
    sns.histplot(lin_full['best_p'], label='Lin Oracle p', ax=axes[1], color='red', alpha=0.5, bins=20, stat='probability')
    axes[1].set_title('Distribution of Optimal Node Selection % (best_p)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/gain_parameters_analysis.png', dpi=300)
    print("Saved analysis plots to results/gain_parameters_analysis.png")

if __name__ == "__main__":
    main()
