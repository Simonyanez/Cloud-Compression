#!/usr/bin/env python3
"""
Combined comparative plot script for PCADC experiment database.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse

class CombinedComparator:
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
        fig, ax = plt.subplots(figsize=(20, 10))

        # Define a colorblind-friendly palette
        colors = {
            'Baseline': '#1f77b4',
            'Fast': '#ff7f0e',
            'Medium': '#2ca02c'
        }
        markers = {
            4: 'o',
            8: 's',
            16: '^'
        }
        linestyles = {
            'Baseline': '-',
            'Fast': '--',
            'Medium': '-.'
        }

        for exp_name, data in self.results.items():
            method = exp_name.split('-')[1] if 'RD' in exp_name else 'Baseline'
            block_size_str = exp_name.split('-B')[1].split('-')[0]
            block_size = int(block_size_str)

            color = colors.get(method, '#000000')
            marker = markers.get(block_size, 'x')
            linestyle = linestyles.get(method, ':')

            label = f'{method} - B{block_size}'

            if 'Baseline' in exp_name:
                ax.plot(data['bpv'], data['psnr'], marker=marker, linestyle=linestyle, label=label, color=color)
            else:
                # Original curve
                ax.plot(data['bpv'], data['psnr'], marker=marker, linestyle=linestyle, label=label, color=color)
                
                # Curve with overhead
                overhead = self.overheads.get(exp_name, 0)
                data['bpv_with_overhead'] = data['bpv'] + overhead
                ax.plot(data['bpv_with_overhead'], data['psnr'], marker=marker, linestyle=':', label=f'{label} + overhead', alpha=0.7, color=color)
                
                # Add text with overhead value
                if not data.empty:
                    last_point = data.iloc[-1]
                    ax.text(last_point['bpv_with_overhead'] * 1.01, last_point['psnr'], f'+{overhead:.4f} bpv', fontsize=9, color=color)

        ax.set_title(title)
        ax.set_xlabel('Bits per Voxel (bpv)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_xscale('log')
        ax.legend(loc='best', fontsize='medium')
        ax.grid(True, which="both", ls="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📁 Comparative plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare PCADC experiment results for multiple block sizes')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output file for the plot')
    parser.add_argument('--title', type=str, default='Rate-Distortion Curve Comparison (All Block Sizes)',
                       help='Title for the plot')

    args = parser.parse_args()

    base_dir = Path('results/PCADC_TEST')
    experiment_dirs = [
        base_dir / 'Baseline-B4',
        base_dir / 'RD-Fast-B4',
        base_dir / 'RD-Medium-B4-C4',
        base_dir / 'Baseline-B8',
        base_dir / 'RD-Fast-B8',
        base_dir / 'RD-Medium-B8-C4',
        base_dir / 'Baseline-B16',
        base_dir / 'RD-Fast-B16',
        base_dir / 'RD-Medium-B16-C4',
    ]

    for exp_dir in experiment_dirs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"❌ Directory not found: {exp_dir}")
            return

    comparator = CombinedComparator(experiment_dirs)
    comparator.load_data()
    comparator.plot_comparison(args.output, args.title)

if __name__ == "__main__":
    main()
