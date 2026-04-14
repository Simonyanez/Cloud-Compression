import sqlite3
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from pathlib import Path

def bjontegaard_metric(R1, PSNR1, R2, PSNR2, mode='rate'):
    """Calculates BD-PSNR (dB) or BD-Rate (%) between two RD curves."""
    # Convert to numpy arrays to avoid indexing issues
    R1, PSNR1 = np.array(R1), np.array(PSNR1)
    R2, PSNR2 = np.array(R2), np.array(PSNR2)
    
    lR1 = np.log10(R1)
    lR2 = np.log10(R2)

    # Find common overlap range
    if mode == 'psnr':
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))
        if min_int >= max_int: return 0.0
        f1 = interp1d(lR1, PSNR1, kind='cubic')
        f2 = interp1d(lR2, PSNR2, kind='cubic')
        int1 = quad(f1, min_int, max_int)[0]
        int2 = quad(f2, min_int, max_int)[0]
        return (int2 - int1) / (max_int - min_int)
    else:
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))
        if min_int >= max_int: return 0.0
        idx1 = np.argsort(PSNR1)
        idx2 = np.argsort(PSNR2)
        f1 = interp1d(PSNR1[idx1], lR1[idx1], kind='cubic')
        f2 = interp1d(PSNR2[idx2], lR2[idx2], kind='cubic')
        int1 = quad(f1, min_int, max_int)[0]
        int2 = quad(f2, min_int, max_int)[0]
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        return (10**avg_exp_diff - 1) * 100

def get_rd_data(db_path):
    conn = sqlite3.connect(db_path)
    # Dist is qerror (Euclidean norm), so Dist^2 is SSE
    df = pd.read_sql("SELECT bsize, q_step, point_count, struct_rate, struct_dist, oracle_rate, oracle_dist FROM results", conn)
    conn.close()
    
    df['struct_sse'] = df['struct_dist']**2
    df['oracle_sse'] = df['oracle_dist']**2
    
    # Aggregating over the whole point cloud
    grouped = df.groupby(['bsize', 'q_step']).agg({
        'point_count': 'sum',
        'struct_rate': 'sum',
        'struct_sse': 'sum',
        'oracle_rate': 'sum',
        'oracle_sse': 'sum'
    }).reset_index()
    
    # Rate in bits per point (bpp)
    grouped['R_s'] = grouped['struct_rate'] / grouped['point_count']
    grouped['R_o'] = grouped['oracle_rate'] / grouped['point_count']
    
    # PSNR_Y calculation
    grouped['PSNR_s'] = 10 * np.log10(255**2 / (grouped['struct_sse'] / grouped['point_count']))
    grouped['PSNR_o'] = 10 * np.log10(255**2 / (grouped['oracle_sse'] / grouped['point_count']))
    
    return grouped

def main():
    abs_db = "results/oracle_upper_bound.db"
    lin_db = "results/linear_oracle_upper_bound.db"
    
    print(f"Loading data from {abs_db} and {lin_db}...")
    abs_rd = get_rd_data(abs_db)
    lin_rd = get_rd_data(lin_db)
    
    block_sizes = sorted(abs_rd['bsize'].unique())
    
    for bsize in block_sizes:
        print(f"\n--- Detailed RD Results for B={bsize} ---")
        d_abs = abs_rd[abs_rd['bsize'] == bsize].sort_values('q_step')
        d_lin = lin_rd[lin_rd['bsize'] == bsize].sort_values('q_step')
        
        print(f"{'QStep':<6} | {'R_struct':<10} | {'P_struct':<10} | {'R_abs':<10} | {'P_abs':<10} | {'R_lin':<10} | {'P_lin':<10}")
        print("-" * 80)
        # Merge on q_step to align
        merged = pd.merge(d_abs, d_lin, on=['bsize', 'q_step'], suffixes=('_abs', '_lin'))
        for _, row in merged.iterrows():
            print(f"{row['q_step']:<6.0f} | {row['R_s_abs']:<10.4f} | {row['PSNR_s_abs']:<10.2f} | {row['R_o_abs']:<10.4f} | {row['PSNR_o_abs']:<10.2f} | {row['R_o_lin']:<10.4f} | {row['PSNR_o_lin']:<10.2f}")

    print("\n" + "="*95)
    print(f"{'BSize':<6} | {'Metric':<10} | {'Absolute Oracle vs Structural':<25} | {'Linear Oracle vs Structural':<25}")
    print("-" * 95)
    
    for bsize in block_sizes:
        # Abs
        d_abs = abs_rd[abs_rd['bsize'] == bsize].sort_values('q_step')
        bdr_abs = bjontegaard_metric(d_abs['R_s'], d_abs['PSNR_s'], d_abs['R_o'], d_abs['PSNR_o'], mode='rate')
        bdp_abs = bjontegaard_metric(d_abs['R_s'], d_abs['PSNR_s'], d_abs['R_o'], d_abs['PSNR_o'], mode='psnr')
        
        # Lin
        d_lin = lin_rd[lin_rd['bsize'] == bsize].sort_values('q_step')
        bdr_lin = bjontegaard_metric(d_lin['R_s'], d_lin['PSNR_s'], d_lin['R_o'], d_lin['PSNR_o'], mode='rate')
        bdp_lin = bjontegaard_metric(d_lin['R_s'], d_lin['PSNR_s'], d_lin['R_o'], d_lin['PSNR_o'], mode='psnr')
        
        print(f"{bsize:<6} | {'BD-Rate %':<10} | {bdr_abs:>27.4f}% | {bdr_lin:>27.4f}%")
        print(f"{'':<6} | {'BD-PSNR dB':<10} | {bdp_abs:>27.4f}  | {bdp_lin:>27.4f} ")
        print("-" * 95)

    print("\nEfficiency Analysis (Percentage of Absolute Gain captured by Linear Oracle):")
    for bsize in block_sizes:
        d_abs = abs_rd[abs_rd['bsize'] == bsize].sort_values('q_step')
        d_lin = lin_rd[lin_rd['bsize'] == bsize].sort_values('q_step')
        bdr_abs = bjontegaard_metric(d_abs['R_s'], d_abs['PSNR_s'], d_abs['R_o'], d_abs['PSNR_o'], mode='rate')
        bdr_lin = bjontegaard_metric(d_lin['R_s'], d_lin['PSNR_s'], d_lin['R_o'], d_lin['PSNR_o'], mode='rate')
        
        eff = (bdr_lin / bdr_abs * 100) if bdr_abs != 0 else 0
        print(f"B={bsize:<2}: {eff:.2f}% of possible bitrate reduction achieved.")

if __name__ == "__main__":
    main()
