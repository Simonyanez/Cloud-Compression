#!/usr/bin/env python3
"""
Quick analysis script for PCADC experiment database.
Run this to get immediate insights into your pipeline behavior.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse

class PCDCAnalyzer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        
    def load_data(self):
        """Load all tables into dataframes"""
        self.fit_df = pd.read_sql_query("SELECT * FROM fit", self.conn)
        self.coeffs_df = pd.read_sql_query("SELECT * FROM coeffs", self.conn)
        self.decision_df = pd.read_sql_query("SELECT * FROM decision", self.conn)
        self.encoding_df = pd.read_sql_query("SELECT * FROM encoding", self.conn)
        self.cluster_df = pd.read_sql_query("SELECT * FROM cluster", self.conn)
        
        print(f"Loaded data:")
        print(f"  - {len(self.fit_df)} fit results")
        print(f"  - {len(self.coeffs_df)} coefficient entries")
        print(f"  - {len(self.decision_df)} RDO decisions")
        print(f"  - {len(self.encoding_df)} encoding results")
        print(f"  - {len(self.cluster_df)} cluster assignments")
        
    def plot_overview(self):
        """Create overview dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCADC Pipeline Overview', fontsize=16)
        
        # 1. Fit Quality Distribution
        axes[0,0].hist(self.fit_df['rmse'], bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Linear Fit RMSE Distribution')
        axes[0,0].set_xlabel('RMSE')
        axes[0,0].set_ylabel('Block Count')
        axes[0,0].axvline(self.fit_df['rmse'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.fit_df["rmse"].mean():.3f}')
        axes[0,0].legend()
        
        # 2. Cluster Distribution
        cluster_counts = self.cluster_df['label'].value_counts().sort_index()
        axes[0,1].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='green')
        axes[0,1].set_title('Cluster Assignment Distribution')
        axes[0,1].set_xlabel('Cluster Label')
        axes[0,1].set_ylabel('Block Count')
        
        # 3. Coefficient Energy Distribution
        if not self.coeffs_df.empty:
            axes[0,2].hist(self.coeffs_df['energy'], bins=50, alpha=0.7, color='orange')
            axes[0,2].set_title('Coefficient Energy Distribution')
            axes[0,2].set_xlabel('Energy Compaction (top-10)')
            axes[0,2].set_ylabel('Count')
        
        # 4. RDO Cost Distribution
        if not self.decision_df.empty:
            axes[1,0].hist(self.decision_df['cost'], bins=50, alpha=0.7, color='purple')
            axes[1,0].set_title('RDO Cost Distribution')
            axes[1,0].set_xlabel('RD Cost')
            axes[1,0].set_ylabel('Decision Count')
        
        # 5. Rate-Distortion Curve
        if not self.encoding_df.empty:
            encoding_sorted = self.encoding_df.sort_values('bpv')
            axes[1,1].plot(encoding_sorted['bpv'], encoding_sorted['psnr'], 'o-', color='red')
            axes[1,1].set_title('Rate-Distortion Curve')
            axes[1,1].set_xlabel('Bits per Vertex (BPV)')
            axes[1,1].set_ylabel('PSNR (dB)')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Decision Analysis by Graph
        if not self.decision_df.empty:
            graph_decisions = self.decision_df.groupby('graph_descriptor').size()
            axes[1,2].bar(graph_decisions.index, graph_decisions.values, alpha=0.7, color='brown')
            axes[1,2].set_title('Graph Selection Frequency')
            axes[1,2].set_xlabel('Graph ID')
            axes[1,2].set_ylabel('Selection Count')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_analysis(self):
        """Detailed cluster analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis', fontsize=16)
        
        # Merge cluster and fit data
        cluster_fit = self.cluster_df.merge(self.fit_df, on='block_id')
        
        # 1. RMSE by Cluster
        sns.boxplot(data=cluster_fit, x='label', y='rmse', ax=axes[0,0])
        axes[0,0].set_title('Fit Quality by Cluster')
        axes[0,0].set_xlabel('Cluster Label')
        axes[0,0].set_ylabel('RMSE')
        
        # 2. Cluster Size vs Quality
        cluster_stats = cluster_fit.groupby('label').agg({
            'rmse': ['mean', 'std', 'count']
        }).round(3)
        cluster_stats.columns = ['Mean_RMSE', 'Std_RMSE', 'Block_Count']
        cluster_stats = cluster_stats.reset_index()
        
        scatter = axes[0,1].scatter(cluster_stats['Block_Count'], cluster_stats['Mean_RMSE'], 
                                  s=100, alpha=0.7, c=cluster_stats['label'], cmap='tab10')
        axes[0,1].set_title('Cluster Size vs Mean RMSE')
        axes[0,1].set_xlabel('Number of Blocks')
        axes[0,1].set_ylabel('Mean RMSE')
        plt.colorbar(scatter, ax=axes[0,1], label='Cluster Label')
        
        # 3. Energy by Cluster (if available)
        if not self.coeffs_df.empty:
            cluster_coeffs = self.cluster_df.merge(self.coeffs_df, on='block_id')
            sns.boxplot(data=cluster_coeffs, x='label', y='energy', ax=axes[1,0])
            axes[1,0].set_title('Coefficient Energy by Cluster')
            axes[1,0].set_xlabel('Cluster Label')
            axes[1,0].set_ylabel('Energy Compaction')
        
        # 4. Decision Cost by Cluster (if available)
        if not self.decision_df.empty:
            cluster_decisions = self.cluster_df.merge(self.decision_df, on='block_id')
            sns.boxplot(data=cluster_decisions, x='label', y='cost', ax=axes[1,1])
            axes[1,1].set_title('RDO Cost by Cluster')
            axes[1,1].set_xlabel('Cluster Label')
            axes[1,1].set_ylabel('RD Cost')
        
        plt.tight_layout()
        return fig
    
    def plot_rdo_analysis(self):
        """RDO decision analysis"""
        if self.decision_df.empty:
            print("No RDO decision data available")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RDO Decision Analysis', fontsize=16)
        
        # 1. Rate vs Distortion trade-off
        axes[0,0].scatter(self.decision_df['rate_diff'], self.decision_df['dist_diff'], 
                         alpha=0.6, s=20)
        axes[0,0].set_title('Rate-Distortion Trade-off')
        axes[0,0].set_xlabel('Rate Difference')
        axes[0,0].set_ylabel('Distortion Difference')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Cost vs Rate difference
        axes[0,1].scatter(self.decision_df['rate_diff'], self.decision_df['cost'], 
                         alpha=0.6, s=20, color='orange')
        axes[0,1].set_title('RD Cost vs Rate Difference')
        axes[0,1].set_xlabel('Rate Difference')
        axes[0,1].set_ylabel('RD Cost')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Graph selection patterns
        graph_selection = self.decision_df['graph_id'].value_counts().sort_index()
        axes[1,0].bar(graph_selection.index, graph_selection.values, alpha=0.7, color='green')
        axes[1,0].set_title('Graph Selection Frequency')
        axes[1,0].set_xlabel('Graph ID')
        axes[1,0].set_ylabel('Selection Count')
        
        # 4. Entropy difference distribution
        if 'entropy_diff' in self.decision_df.columns:
            axes[1,1].hist(self.decision_df['entropy_diff'].dropna(), bins=50, 
                          alpha=0.7, color='purple')
            axes[1,1].set_title('Entropy Difference Distribution')
            axes[1,1].set_xlabel('Entropy Difference')
            axes[1,1].set_ylabel('Count')
            axes[1,1].axvline(0, color='red', linestyle='--', label='No Change')
            axes[1,1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_encoding_results(self):
        """Encoding results analysis"""
        if self.encoding_df.empty:
            print("No encoding results available")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Encoding Results Analysis', fontsize=16)
        
        # Sort by q_step for proper ordering
        encoding_sorted = self.encoding_df.sort_values('q_step')
        
        # 1. Rate-Distortion curve
        axes[0,0].plot(encoding_sorted['bpv'], encoding_sorted['psnr'], 'o-', 
                      linewidth=2, markersize=8, color='red')
        axes[0,0].set_title('Rate-Distortion Curve')
        axes[0,0].set_xlabel('Bits per Vertex (BPV)')
        axes[0,0].set_ylabel('PSNR (dB)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add q_step annotations
        for _, row in encoding_sorted.iterrows():
            axes[0,0].annotate(f'q={int(row["q_step"])}', 
                              (row['bpv'], row['psnr']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. PSNR vs Quantization Step
        axes[0,1].plot(encoding_sorted['q_step'], encoding_sorted['psnr'], 'o-', 
                      color='blue', linewidth=2, markersize=8)
        axes[0,1].set_title('PSNR vs Quantization Step')
        axes[0,1].set_xlabel('Quantization Step')
        axes[0,1].set_ylabel('PSNR (dB)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Overhead analysis
        axes[1,0].plot(encoding_sorted['q_step'], encoding_sorted['overhead_bpv'], 'o-', 
                      color='orange', linewidth=2, markersize=8)
        axes[1,0].set_title('Overhead vs Quantization Step')
        axes[1,0].set_xlabel('Quantization Step')
        axes[1,0].set_ylabel('Overhead (BPV)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Efficiency (PSNR per bit)
        efficiency = encoding_sorted['psnr'] / encoding_sorted['bpv']
        axes[1,1].plot(encoding_sorted['q_step'], efficiency, 'o-', 
                      color='green', linewidth=2, markersize=8)
        axes[1,1].set_title('Coding Efficiency (PSNR/BPV)')
        axes[1,1].set_xlabel('Quantization Step')
        axes[1,1].set_ylabel('PSNR per BPV')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """Generate text summary of key findings"""
        print("\n" + "="*50)
        print("PCADC EXPERIMENT SUMMARY REPORT")
        print("="*50)
        
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   Total blocks processed: {len(self.fit_df)}")
        print(f"   Number of clusters: {self.cluster_df['label'].nunique()}")
        if not self.encoding_df.empty:
            print(f"   Quantization steps tested: {sorted(self.encoding_df['q_step'].unique())}")
        
        print(f"\n🎯 FIT QUALITY:")
        print(f"   Mean RMSE: {self.fit_df['rmse'].mean():.4f}")
        print(f"   RMSE std: {self.fit_df['rmse'].std():.4f}")
        print(f"   Best fit RMSE: {self.fit_df['rmse'].min():.4f}")
        print(f"   Worst fit RMSE: {self.fit_df['rmse'].max():.4f}")
        
        print(f"\n🏷️ CLUSTERING:")
        cluster_balance = self.cluster_df['label'].value_counts()
        print(f"   Most used cluster: {cluster_balance.index[0]} ({cluster_balance.iloc[0]} blocks)")
        print(f"   Least used cluster: {cluster_balance.index[-1]} ({cluster_balance.iloc[-1]} blocks)")
        print(f"   Cluster balance ratio: {cluster_balance.iloc[0]/cluster_balance.iloc[-1]:.2f}:1")
        
        if not self.decision_df.empty:
            print(f"\n⚖️ RDO DECISIONS:")
            graph_selection = self.decision_df['graph_id'].value_counts()
            print(f"   Most selected graph: {graph_selection.index[0]} ({graph_selection.iloc[0]} times)")
            print(f"   Graph selection diversity: {len(graph_selection)} different graphs used")
            print(f"   Mean RD cost: {self.decision_df['cost'].mean():.6f}")
        
        if not self.encoding_df.empty:
            print(f"\n📈 ENCODING PERFORMANCE:")
            best_quality = self.encoding_df.loc[self.encoding_df['psnr'].idxmax()]
            best_compression = self.encoding_df.loc[self.encoding_df['bpv'].idxmin()]
            print(f"   Best quality: {best_quality['psnr']:.2f} dB @ {best_quality['bpv']:.3f} BPV (q={int(best_quality['q_step'])})")
            print(f"   Best compression: {best_compression['bpv']:.3f} BPV @ {best_compression['psnr']:.2f} dB (q={int(best_compression['q_step'])})")
        
        print("="*50)
    
    def save_all_plots(self, output_dir: Path):
        """Generate and save all analysis plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Overview
        fig1 = self.plot_overview()
        if fig1:
            fig1.savefig(output_dir / "01_overview.png", dpi=150, bbox_inches='tight')
            plt.close(fig1)
        
        # Cluster analysis
        fig2 = self.plot_cluster_analysis()
        if fig2:
            fig2.savefig(output_dir / "02_clustering.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # RDO analysis
        fig3 = self.plot_rdo_analysis()
        if fig3:
            fig3.savefig(output_dir / "03_rdo_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig3)
        
        # Encoding results
        fig4 = self.plot_encoding_results()
        if fig4:
            fig4.savefig(output_dir / "04_encoding_results.png", dpi=150, bbox_inches='tight')
            plt.close(fig4)
        
        print(f"\n📁 All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PCADC experiment database')
    parser.add_argument('db_path', type=Path, help='Path to SQLite database')
    parser.add_argument('--output', type=Path, default=Path('analysis_plots'), 
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    if not args.db_path.exists():
        print(f"❌ Database not found: {args.db_path}")
        return
    
    analyzer = PCDCAnalyzer(args.db_path)
    analyzer.load_data()
    analyzer.generate_summary_report()
    
    if args.show:
        # Show plots interactively
        analyzer.plot_overview()
        analyzer.plot_cluster_analysis()
        analyzer.plot_rdo_analysis()
        analyzer.plot_encoding_results()
        plt.show()
    else:
        # Save plots to files
        analyzer.save_all_plots(args.output)


if __name__ == "__main__":
    main()
