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
import json

class PCDCComparator:
    def __init__(self, experiment_dirs: list[Path]):
        self.experiment_dirs = experiment_dirs
        self.results = {}
        self.overheads = {}

    def load_data(self):
        """Load data from all experiments"""
        n_points = 765821  # Number of points for longdress_vox10_1051.ply
        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            print(f"Processing experiment: {exp_name}")

            # Load from DB
            db_path_glob = list(exp_dir.glob('*.db'))
            if not db_path_glob:
                print(f"  - No DB file found in {exp_dir}")
                continue
            db_path = db_path_glob[0]
            conn = sqlite3.connect(db_path)
            try:
                db_df = pd.read_sql_query("SELECT * FROM encoding", conn)
                
                overhead_bpv = 0
                if 'Baseline' not in exp_name:
                    huffman_df = pd.read_sql_query("SELECT * FROM cluster_codes", conn)
                    if not huffman_df.empty:
                        expected_length = (huffman_df['probability'] * huffman_df['code_length']).sum()
                        
                        code_estimation_df = pd.read_sql_query("SELECT * FROM code_estimation", conn)
                        if not code_estimation_df.empty:
                            n_blocks = code_estimation_df['total_blocks'].iloc[0]
                            overhead_bpv = expected_length * n_blocks / n_points
                
                self.results[exp_name] = db_df.sort_values('bpv')
                self.overheads[exp_name] = overhead_bpv
                print(f"  - Found {len(db_df)} entries in DB")
            except pd.io.sql.DatabaseError as e:
                print(f"  - Could not read table from {db_path}: {e}")
            finally:
                conn.close()

    def plot_comparison(self, output_path: Path, title: str):
        """Generate comparative plot"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Find common q_steps
        common_q_steps = None
        for exp_name, data in self.results.items():
            if common_q_steps is None:
                common_q_steps = set(data['q_step'])
            else:
                common_q_steps = common_q_steps.intersection(set(data['q_step']))

        colors = plt.cm.get_cmap('tab10', len(self.results))

        # Filter data and plot
        for i, (exp_name, data) in enumerate(self.results.items()):
            filtered_data = data[data['q_step'].isin(common_q_steps)].copy()
            color = colors(i)
            
            if 'Baseline' in exp_name:
                ax.plot(filtered_data['bpv'], filtered_data['psnr'], 'o-', label=exp_name, color=color)
            else:
                # Original curve
                ax.plot(filtered_data['bpv'], filtered_data['psnr'], 'o-', label=exp_name, color=color)
                
                # Curve with overhead
                overhead = self.overheads.get(exp_name, 0)
                filtered_data['bpv_with_overhead'] = filtered_data['bpv'] + overhead
                ax.plot(filtered_data['bpv_with_overhead'], filtered_data['psnr'], 'o--', label=f'{exp_name} + overhead', alpha=0.5, color=color)
                
                # Add text with overhead value
                last_point = filtered_data.iloc[-1]
                ax.text(last_point['bpv_with_overhead'] * 1.01, last_point['psnr'], f'+{overhead:.4f} bpv', fontsize=9, color=color)

        ax.set_title(title)
        ax.set_xlabel('Bits per Voxel (bpv)')
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
    parser.add_argument('--output', type=Path, required=True,
                       help='Output file for the plot')
    parser.add_argument('--title', type=str, default='Rate-Distortion Curve Comparison',
                       help='Title for the plot')

    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"❌ Directory not found: {exp_dir}")
            return

    comparator = PCDCComparator(args.exp_dirs)
    comparator.load_data()
    comparator.plot_comparison(args.output, args.title)

if __name__ == "__main__":
    main()
