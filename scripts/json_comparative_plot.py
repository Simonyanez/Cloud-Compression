#!/usr/bin/env python3
"""
Comparative plot script for PCADC experiment database using only JSON files.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

class JSONComparator:
    def __init__(self, experiment_dirs: list[Path]):
        self.experiment_dirs = experiment_dirs
        self.results = {}

    def load_data(self):
        """Load data from all experiments from JSON files"""
        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            print(f"Processing experiment: {exp_name}")
            
            # Load from JSON files
            json_results = []
            results_dir = exp_dir / 'results'
            if not results_dir.exists():
                print(f"  - Skipping {exp_name}: 'results' directory not found.")
                continue

            for json_file in results_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    json_results.append(data)
            
            if not json_results:
                print(f"  - Skipping {exp_name}: No JSON files found.")
                continue

            json_df = pd.DataFrame(json_results)
            json_df.columns = json_df.columns.str.lower()
            
            self.results[exp_name] = json_df.sort_values('bpv')
            print(f"  - Found {len(json_df)} JSON files")

    def plot_comparison(self, output_path: Path):
        """Generate comparative plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.get_cmap('tab10', len(self.results))
        
        i = 0
        for exp_name, data in self.results.items():
            ax.plot(data['bpv'], data['psnr'], 'o-', 
                    label=exp_name, color=colors(i))
            i += 1

        ax.set_title('Rate-Distortion Curve Comparison (JSON only)')
        ax.set_xlabel('Bits per Vertex (BPV)')
        ax.set_ylabel('PSNR (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📁 Comparative plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare PCADC experiment results using only JSON files')
    parser.add_argument('exp_dirs', type=Path, nargs='+', 
                       help='List of experiment directories to compare')
    parser.add_argument('--output', type=Path, default=Path('json_comparative_plot.png'), 
                       help='Output file for the plot')
    
    args = parser.parse_args()
    
    for exp_dir in args.exp_dirs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"❌ Directory not found: {exp_dir}")
            return
            
    comparator = JSONComparator(args.exp_dirs)
    comparator.load_data()
    comparator.plot_comparison(args.output)

if __name__ == "__main__":
    main()
