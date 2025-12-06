#!/usr/bin/env python3
"""
Comparative plot script for PCADC experiment database.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse

class PCDCComparator:
    def __init__(self, experiment_dirs: list[Path]):
        self.experiment_dirs = experiment_dirs
        self.results = {}

    def load_data(self):
        """Load data from all experiments"""
        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            print(f"Processing experiment: {exp_name}")
            
            # Load from DB
            db_path = next(exp_dir.glob('*.db'))
            conn = sqlite3.connect(db_path)
            db_df = pd.read_sql_query("SELECT * FROM encoding", conn)
            conn.close()
            
            # Load from JSON files
            json_results = []
            results_dir = exp_dir / 'results'
            for json_file in results_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    json_results.append(data)
            json_df = pd.DataFrame(json_results)
            json_df.columns = json_df.columns.str.lower()
            
            self.results[exp_name] = {
                'db': db_df.sort_values('bpv'),
                'json': json_df.sort_values('bpv')
            }
            print(f"  - Found {len(db_df)} entries in DB")
            print(f"  - Found {len(json_df)} JSON files")

    def compare_results(self):
        """Provide a textual comparison of the results."""
        print("\n--- Comparative Analysis ---")
        
        # Group experiments by block size
        experiments_by_block = {}
        for exp_name, data in self.results.items():
            parts = exp_name.split('-')
            block_size = parts[-1] # e.g., 'B4', 'B8', 'B16'

            if block_size not in experiments_by_block:
                experiments_by_block[block_size] = {}

            if 'Baseline' in exp_name:
                experiments_by_block[block_size]['baseline'] = data['db']
            elif 'RD-Fast' in exp_name:
                experiments_by_block[block_size]['rd_fast'] = data['db']
        
        for block_size, exps in experiments_by_block.items():
            baseline_df = exps.get('baseline')
            rd_fast_df = exps.get('rd_fast')

            if baseline_df is None or rd_fast_df is None:
                print(f"  Skipping comparison for {block_size}: Missing baseline or RD-Fast data.")
                continue

            print(f"\nBlock Size: {block_size}")
            print("  Baseline (BPV, PSNR):")
            print(baseline_df[['bpv', 'psnr']].to_string(index=False))
            print("  RD-Fast (BPV, PSNR):")
            print(rd_fast_df[['bpv', 'psnr']].to_string(index=False))

            # Simple comparison: find closest BPV points and compare PSNR
            for _, row_fast in rd_fast_df.iterrows():
                closest_baseline = baseline_df.iloc[(baseline_df['bpv'] - row_fast['bpv']).abs().argsort()[:1]]
                if not closest_baseline.empty:
                    baseline_bpv = closest_baseline['bpv'].iloc[0]
                    baseline_psnr = closest_baseline['psnr'].iloc[0]
                    
                    psnr_diff = row_fast['psnr'] - baseline_psnr
                    bpv_diff_percent = ((row_fast['bpv'] - baseline_bpv) / baseline_bpv) * 100 if baseline_bpv != 0 else 0

                    print(f"    At RD-Fast BPV={row_fast['bpv']:.2f}, PSNR={row_fast['psnr']:.2f} dB:")
                    print(f"      Closest Baseline BPV={baseline_bpv:.2f}, PSNR={baseline_psnr:.2f} dB")
                    print(f"      PSNR difference: {psnr_diff:.2f} dB (RD-Fast vs Baseline)")
                    print(f"      BPV difference: {bpv_diff_percent:.2f}% (RD-Fast vs Baseline)")
                    
                    # Also compare speed if available (assuming 'encoding_time' or similar exists)
                    # This part is speculative as 'encoding_time' is not explicitly in the DB schema shown
                    # If it were, we'd do something like:
                    # fast_time = row_fast.get('encoding_time', np.nan)
                    # baseline_time = closest_baseline.get('encoding_time', np.nan)
                    # if not np.isnan(fast_time) and not np.isnan(baseline_time):
                    #     time_diff_percent = ((fast_time - baseline_time) / baseline_time) * 100
                    #     print(f"      Encoding Time difference: {time_diff_percent:.2f}%")

    def plot_comparison(self, output_path: Path):
        """Generate comparative plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.get_cmap('tab10', len(self.results) * 2)
        
        i = 0
        for exp_name, data in self.results.items():
            # Plot DB results
            ax.plot(data['db']['bpv'], data['db']['psnr'], 'o-',
                    label=f'{exp_name} (DB)', color=colors(i))
            # Plot JSON results
            ax.plot(data['json']['bpv'], data['json']['psnr'], 'x--',
                    label=f'{exp_name} (JSON)', color=colors(i+1))
            i += 2

        ax.set_title('Rate-Distortion Curve Comparison')
        ax.set_xlabel('Bits per Vertex (BPV)')
        ax.set_ylabel('PSNR (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📁 Comparative plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare PCADC experiment results')
    parser.add_argument('exp_dirs', type=Path, nargs='+', 
                       help='List of experiment directories to compare')
    parser.add_argument('--output', type=Path, default=Path('comparative_plot.png'), 
                       help='Output file for the plot')
    
    args = parser.parse_args()
    
    for exp_dir in args.exp_dirs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"❌ Directory not found: {exp_dir}")
            return
            
    comparator = PCDCComparator(args.exp_dirs)
    comparator.load_data()
    comparator.compare_results() # Call the new comparison method
    comparator.plot_comparison(args.output)

if __name__ == "__main__":
    main()