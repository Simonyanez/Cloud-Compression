"""
High-Visibility Rate-Distortion Visualization with Inset Zoom Windows & No-Overhead Ideal Transform Curve
========================================================================================================
Includes:
1. Baseline Curve (Structural GFT)
2. PCADC Net RD Curves (K=4, 8, 10, 12, 16) including signaling overhead
3. PCADC Ideal Transform Curve (No Overhead / Zero Side-Information limit) for the optimal K
4. Inset Zoom Windows at Q=24 and Q=32 operating points showing both Net and Ideal points relative to Baseline
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

project_root = Path(__file__).resolve().parent.parent.parent
db_path = project_root / "results/exp_e6_master_suite.db"
fig_dir = project_root / "docs/report/figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.35
})

def generate_zoomed_rd():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()

    B_list = [32, 16, 8, 4]
    Q_list = [24, 32, 48, 64]
    K_list = [4, 8, 10, 12, 16]
    k_colors = {4: '#4e79a7', 8: '#f28e2b', 10: '#e15759', 12: '#2b908f', 16: '#59a14f'}
    k_markers = {4: 'o', 8: 's', 10: '^', 12: 'D', 16: 'v'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, B in enumerate(B_list):
        ax = axes[idx]
        
        # 1. Baseline curve
        c.execute("""SELECT q_step, baseline_bpv, baseline_psnr_db 
                     FROM gating_results WHERE child_b=? AND K=4 
                     GROUP BY q_step ORDER BY q_step""", (B,))
        b_pts = c.fetchall()
        if b_pts:
            b_qs, b_bpv, b_psnr = zip(*b_pts)
            ax.plot(b_bpv, b_psnr, 'k*--', linewidth=2.2, markersize=9, label='Structural Baseline', zorder=7)

        # 2. PCADC curves for each K (with overhead)
        for K in K_list:
            c.execute("""SELECT g.q_step, g.net_bpv, g.psnr_db, g.delta_bpv
                         FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=?
                         ORDER BY g.q_step""", (B, K, B, K))
            pts = c.fetchall()
            if pts:
                qs, bpv, psnr, d_bpv = zip(*pts)
                ax.plot(bpv, psnr, color=k_colors[K], marker=k_markers[K], linewidth=1.4, markersize=5.5, label=f'PCADC K={K} (Net)')

        # 3. Ideal Transform Curve (No Overhead / Zero-OH Bound) for best K (K=12 or K=16)
        best_K = 16 if B == 32 else 12
        c.execute("""SELECT g.q_step, g.hier_transform_bpv, g.psnr_db
                     FROM gating_results g
                     INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                 FROM gating_results WHERE child_b=? AND K=? 
                                 GROUP BY child_b, K, q_step) b
                     ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                     WHERE g.child_b=? AND g.K=?
                     ORDER BY g.q_step""", (B, best_K, B, best_K))
        no_oh_pts = c.fetchall()
        if no_oh_pts:
            no_oh_qs, no_oh_bpv, no_oh_psnr = zip(*no_oh_pts)
            ax.plot(no_oh_bpv, no_oh_psnr, color='#9467bd', linestyle=':', marker='x', linewidth=2.0, markersize=7,
                    label=f'PCADC K={best_K} (No Overhead / Ideal)', zorder=6)

        ax.set_title(f'Multi-Scale RD Curve with Inset Zooms (Scale B = {B})', fontweight='bold', fontsize=13)
        ax.set_xlabel('Rate (BPV)', fontsize=11)
        ax.set_ylabel('Attribute PSNR (dB)', fontsize=11)
        ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right', fontsize=8.5)

        # ---------------------------------------------------------------------
        # Inset 1: Zoom on Q=24 Operating Cluster (High Bitrate / High PSNR)
        # ---------------------------------------------------------------------
        axins1 = inset_axes(ax, width="42%", height="31%", loc='center',
                            bbox_to_anchor=(0.04, 0.47, 0.9, 0.9), bbox_transform=ax.transAxes)
        
        c.execute("SELECT baseline_bpv, baseline_psnr_db FROM gating_results WHERE child_b=? AND q_step=24 LIMIT 1", (B,))
        q24_base = c.fetchone()
        if q24_base:
            axins1.scatter([q24_base[0]], [q24_base[1]], color='black', marker='*', s=130, zorder=8, label='Baseline')

        q24_bpvs = [q24_base[0]] if q24_base else []
        q24_psnrs = [q24_base[1]] if q24_base else []

        # PCADC Net points
        for K in K_list:
            c.execute("""SELECT g.net_bpv, g.psnr_db, g.hier_transform_bpv FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? AND q_step=24 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? AND g.q_step=24""", (B, K, B, K))
            r = c.fetchone()
            if r:
                axins1.scatter([r[0]], [r[1]], color=k_colors[K], marker=k_markers[K], s=55, zorder=7)
                q24_bpvs.append(r[0]); q24_psnrs.append(r[1])
                # Also include no-overhead point for best K
                if K == best_K:
                    axins1.scatter([r[2]], [r[1]], color='#9467bd', marker='x', s=80, linewidth=2.0, zorder=7)
                    q24_bpvs.append(r[2])

        if q24_bpvs:
            bpv_pad = (max(q24_bpvs) - min(q24_bpvs)) * 0.20 or 0.001
            psnr_pad = (max(q24_psnrs) - min(q24_psnrs)) * 0.35 or 0.015
            axins1.set_xlim(min(q24_bpvs) - bpv_pad, max(q24_bpvs) + bpv_pad)
            axins1.set_ylim(min(q24_psnrs) - psnr_pad, max(q24_psnrs) + psnr_pad)
            
            # Annotate overhead arrow from No-OH to Net for best_K
            c.execute("""SELECT g.hier_transform_bpv, g.net_bpv, g.psnr_db FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? AND q_step=24 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? AND g.q_step=24""", (B, best_K, B, best_K))
            arr_pts = c.fetchone()
            if arr_pts:
                axins1.annotate('', xy=(arr_pts[1], arr_pts[2]), xytext=(arr_pts[0], arr_pts[2]),
                                arrowprops=dict(arrowstyle="->", color='#9467bd', lw=1.2, ls='--'))
                axins1.text(arr_pts[0] + (arr_pts[1]-arr_pts[0])*0.5, arr_pts[2] + psnr_pad*0.25,
                            'OH', color='#9467bd', fontsize=7, fontweight='bold', ha='center')

            axins1.set_title(f'Zoom @ Q=24 (Net Δ={-(q24_base[0]-min(q24_bpvs))*1000:.1f} mBPV)', fontsize=8, fontweight='bold', pad=3)
            axins1.tick_params(axis='both', which='major', labelsize=7)
            mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

        # ---------------------------------------------------------------------
        # Inset 2: Zoom on Q=32 Operating Cluster (Medium Bitrate)
        # ---------------------------------------------------------------------
        axins2 = inset_axes(ax, width="40%", height="27%", loc='center',
                            bbox_to_anchor=(0.56, 0.08, 0.9, 0.9), bbox_transform=ax.transAxes)
        
        c.execute("SELECT baseline_bpv, baseline_psnr_db FROM gating_results WHERE child_b=? AND q_step=32 LIMIT 1", (B,))
        q32_base = c.fetchone()
        if q32_base:
            axins2.scatter([q32_base[0]], [q32_base[1]], color='black', marker='*', s=130, zorder=8)

        q32_bpvs = [q32_base[0]] if q32_base else []
        q32_psnrs = [q32_base[1]] if q32_base else []

        for K in K_list:
            c.execute("""SELECT g.net_bpv, g.psnr_db, g.hier_transform_bpv FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? AND q_step=32 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? AND g.q_step=32""", (B, K, B, K))
            r = c.fetchone()
            if r:
                axins2.scatter([r[0]], [r[1]], color=k_colors[K], marker=k_markers[K], s=55, zorder=7)
                q32_bpvs.append(r[0]); q32_psnrs.append(r[1])
                if K == best_K:
                    axins2.scatter([r[2]], [r[1]], color='#9467bd', marker='x', s=80, linewidth=2.0, zorder=7)
                    q32_bpvs.append(r[2])

        if q32_bpvs:
            bpv_pad = (max(q32_bpvs) - min(q32_bpvs)) * 0.20 or 0.001
            psnr_pad = (max(q32_psnrs) - min(q32_psnrs)) * 0.35 or 0.015
            axins2.set_xlim(min(q32_bpvs) - bpv_pad, max(q32_bpvs) + bpv_pad)
            axins2.set_ylim(min(q32_psnrs) - psnr_pad, max(q32_psnrs) + psnr_pad)
            
            c.execute("""SELECT g.hier_transform_bpv, g.net_bpv, g.psnr_db FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? AND q_step=32 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? AND g.q_step=32""", (B, best_K, B, best_K))
            arr_pts = c.fetchone()
            if arr_pts:
                axins2.annotate('', xy=(arr_pts[1], arr_pts[2]), xytext=(arr_pts[0], arr_pts[2]),
                                arrowprops=dict(arrowstyle="->", color='#9467bd', lw=1.2, ls='--'))

            axins2.set_title(f'Zoom @ Q=32 (Net Δ={-(q32_base[0]-min(q32_bpvs))*1000:.1f} mBPV)', fontsize=8, fontweight='bold', pad=3)
            axins2.tick_params(axis='both', which='major', labelsize=7)
            mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_rd_curves_zoomed.png', bbox_inches='tight')
    plt.close()
    conn.close()
    print("Generated fig1_rd_curves_zoomed.png with No-Overhead Ideal Transform Curve & Inset Zooms!")

if __name__ == '__main__':
    generate_zoomed_rd()
