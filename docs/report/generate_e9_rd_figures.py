#!/usr/bin/env python3
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "results/exp_e9_multi_entropy_suite.db"
fig_dir = project_root / "docs/report/figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35
})

def generate_e9_rd_curves():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()

    B_list = [32, 16, 8, 4]
    
    # 4 distinct colors/markers for all 4 capacities
    colors = {24: '#1f77b4', 16: '#2ca02c', 8: '#ff7f0e', 4: '#d62728'}
    markers = {24: 'o', 16: 's', 8: '^', 4: 'v'}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, B in enumerate(B_list):
        ax = axes[idx]

        # 1. Baseline Structural Curve
        c.execute("""
            SELECT q_step, baseline_real_bpv, decoded_psnr_db 
            FROM benchmark_runs 
            WHERE child_b=? AND K=24 
            ORDER BY q_step
        """, (B,))
        b_rows = c.fetchall()
        
        b_bpvs = [r[1] for r in b_rows]
        b_psnrs = [r[2] for r in b_rows]
        
        ax.plot(b_bpvs, b_psnrs, 'k*--', linewidth=2.4, markersize=10, label='Línea Base Estructural', zorder=6)

        # 2. PCADC Net Curves for all K in {24, 16, 8, 4}
        for K in [24, 16, 8, 4]:
            c.execute("""
                SELECT q_step, pcadc_real_net_arith_bpv, decoded_psnr_db, real_delta_arith_pct
                FROM benchmark_runs 
                WHERE child_b=? AND K=? 
                ORDER BY q_step
            """, (B, K))
            rows = c.fetchall()
            if rows:
                qs, net_bpvs, psnrs, delta_pcts = zip(*rows)
                label_txt = f'PCADC K={K}'
                ax.plot(net_bpvs, psnrs, color=colors[K], marker=markers[K], linewidth=2.0, markersize=7.5, label=label_txt, zorder=7)

        # Labels and cosmetics
        ax.set_title(f'Escala de Sub-bloque $B_c={B}$ ($N_p=834$ Padres)', fontweight='bold', pad=10)
        ax.set_xlabel('Tasa Real de Bits por Vóxel (BPV)')
        ax.set_ylabel('PSNR Decodificado $Y$ (dB)')
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    out_file = fig_dir / "fig1_rd_curves_zoomed.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Generated clean E9 RD figure with all 4 K capacities: {out_file}")

if __name__ == '__main__':
    generate_e9_rd_curves()
