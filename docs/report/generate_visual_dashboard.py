"""
High-Visibility Visual Figure Generator for PCADC (v3)
======================================================
Key visualization improvements:
1. fig1_rd_curves_zoomed.png:
   - Panel (a): RD Rate-Savings Curve (PSNR vs % Rate Savings over Baseline)
   - Panel (b): Inset Zoomed RD operating points around each Q step with error bars / spread
   - Panel (c): Direct milli-BPV (mBPV) Gain vs Quantization step Q with independent scale-appropriate dynamic y-axis
2. fig2_delta_bpv_heatmap.png:
   - Heatmap of percentage savings and milli-BPV savings
3. fig3_mode_distribution_bars.png:
   - Mode unfreezing
4. fig4_codebook_slopes_analysis.png:
   - 3D Orientations, SLP and SLW distributions
5. fig5_overhead_ratio_curves.png:
   - Overhead breakeven
6. fig6_correlation_phase_map.png:
   - Phase boundary
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def generate_visuals():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()

    B_list = [32, 16, 8, 4]
    Q_list = [24, 32, 48, 64]
    K_list = [4, 8, 10, 12, 16]
    k_colors = {4: '#4e79a7', 8: '#f28e2b', 10: '#e15759', 12: '#2b908f', 16: '#59a14f'}

    # =========================================================================
    # Figure 1: 3-Row Multi-Perspective Rate-Distortion & Relative Gain Visualizer
    # Row 1: Traditional RD (PSNR vs Rate) with Inset Zoom Box
    # Row 2: Relative Rate Savings (% Reduction in Bitrate vs Baseline) vs PSNR
    # Row 3: Absolute Bit Savings ΔBPV (mBPV = 10⁻³ BPV) vs Q
    # =========================================================================
    print("Generating Figure 1: Multi-Perspective Zoomed RD Visualizer...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 10.5))

    for idx, B in enumerate(B_list):
        ax_rd = axes[0, idx]
        ax_pct = axes[1, idx]
        ax_mbpv = axes[2, idx]

        # 1. Baseline
        c.execute("""SELECT q_step, baseline_bpv, baseline_psnr_db 
                     FROM gating_results WHERE child_b=? AND K=4 
                     GROUP BY q_step ORDER BY q_step""", (B,))
        b_pts = c.fetchall()
        if b_pts:
            b_qs, b_bpv, b_psnr = zip(*b_pts)
            ax_rd.plot(b_bpv, b_psnr, 'k*--', linewidth=2.0, markersize=8, label='Baseline', zorder=6)

        # 2. PCADC Curves
        for K in K_list:
            c.execute("""SELECT g.q_step, g.net_bpv, g.psnr_db, g.delta_bpv, g.delta_pct
                         FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=? 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=?
                         ORDER BY g.q_step""", (B, K, B, K))
            pts = c.fetchall()
            if pts:
                qs, bpv, psnr, d_bpv, d_pct = zip(*pts)
                
                # Row 1: Standard RD
                ax_rd.plot(bpv, psnr, 'o-', color=k_colors[K], linewidth=1.5, markersize=5, label=f'K={K}')
                
                # Row 2: Relative Rate Savings (%) vs PSNR (Positive = Gain)
                # Savings is -delta_pct so higher is better
                savings_pct = -np.array(d_pct)
                ax_pct.plot(psnr, savings_pct, 's-', color=k_colors[K], linewidth=1.8, markersize=6, label=f'K={K}')
                
                # Row 3: Absolute Savings (milli-BPV) vs Q (Negative = Net Bitrate Reduction)
                ax_mbpv.plot(qs, np.array(d_bpv)*1000.0, 'D-', color=k_colors[K], linewidth=1.8, markersize=6, label=f'K={K}')

        # Formatting Row 1: Standard RD
        ax_rd.set_title(f'RD Curve: B = {B}', fontweight='bold')
        ax_rd.set_xlabel('Rate (BPV)')
        if idx == 0:
            ax_rd.set_ylabel('Attribute PSNR (dB)')
            ax_rd.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')

        # Formatting Row 2: % Rate Savings (Higher is Better)
        ax_pct.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_pct.set_title(f'% Bitrate Savings: B = {B}', fontweight='bold')
        ax_pct.set_xlabel('Attribute PSNR (dB)')
        if idx == 0:
            ax_pct.set_ylabel('Bitrate Savings vs Baseline (%) [↑ Better]')
            ax_pct.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')

        # Formatting Row 3: Absolute mBPV Savings (Negative is Better)
        ax_mbpv.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_mbpv.set_title(f'Net Bit Reduction ΔBPV: B = {B}', fontweight='bold')
        ax_mbpv.set_xlabel('Quantization Parameter Q')
        ax_mbpv.set_xticks(Q_list)
        if idx == 0:
            ax_mbpv.set_ylabel('Δ BPV ($10^{-3}$ BPV) [↓ Better]')
            ax_mbpv.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_rd_curves_zoomed.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Delta BPV Heatmap Matrix
    # =========================================================================
    print("Generating Figure 2: Delta BPV Heatmap...")
    fig, axes = plt.subplots(1, len(K_list), figsize=(18, 3.8), sharey=True)
    
    for kidx, K in enumerate(K_list):
        ax = axes[kidx]
        matrix = np.zeros((len(B_list), len(Q_list)))
        for bidx, B in enumerate(B_list):
            for qidx, Q in enumerate(Q_list):
                c.execute("""SELECT delta_bpv FROM gating_results g
                             INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                         FROM gating_results WHERE child_b=? AND K=? AND q_step=? 
                                         GROUP BY child_b, K, q_step) b
                             ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                             WHERE g.child_b=? AND g.K=? AND g.q_step=?""", (B, K, Q, B, K, Q))
                row = c.fetchone()
                matrix[bidx, qidx] = row[0] * 1000.0 if row else 0.0

        im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=-4.0, vmax=0.1, aspect='auto')
        ax.set_title(f'Codebook K = {K}', fontweight='bold')
        ax.set_xticks(range(len(Q_list)))
        ax.set_xticklabels([f'Q={q}' for q in Q_list])
        if kidx == 0:
            ax.set_yticks(range(len(B_list)))
            ax.set_yticklabels([f'B={b}' for b in B_list])
            ax.set_ylabel('Block Dimension B')
        else:
            ax.set_yticks([])

        for i in range(len(B_list)):
            for j in range(len(Q_list)):
                val = matrix[i, j]
                color = 'white' if abs(val) > 2.0 else 'black'
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', color=color, fontsize=8, fontweight='bold')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\Delta$ BPV ($10^{-3}$ BPV, Negative = Net Gain)', rotation=270, labelpad=15)
    plt.subplots_adjust(right=0.90)
    plt.savefig(fig_dir / 'fig2_delta_bpv_heatmap.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Mode Diversity Histogram K=4 vs K=12
    # =========================================================================
    print("Generating Figure 3: Mode Diversity...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for pidx, (K_val, title) in enumerate([(4, '(a) Codebook K=4 (Mode Starvation)'), (12, '(b) Codebook K=12 (Rich Multi-mode Utilization)')]):
        ax = axes[pidx]
        c.execute("""SELECT g.run_id FROM gating_results g
                     INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                 FROM gating_results WHERE child_b=16 AND K=? AND q_step=24 
                                 GROUP BY child_b, K, q_step) b
                     ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                     WHERE g.child_b=16 AND g.K=? AND g.q_step=24""", (K_val, K_val))
        run_id = c.fetchone()[0]
        
        c.execute("SELECT mode_k, activation_pct FROM mode_activation_stats WHERE run_id=? ORDER BY mode_k", (run_id,))
        m_rows = c.fetchall()
        modes, pcts = zip(*m_rows)
        colors = ['#8c564b' if m==0 else '#1f77b4' for m in modes]
        bars = ax.bar(modes, pcts, color=colors, edgecolor='black', alpha=0.85, width=0.6)
        ax.set_title(f'{title} at B=16, Q=24', fontweight='bold')
        ax.set_xlabel('Mode Index k (0: Fallback, 1..K-1: Directional)')
        ax.set_ylabel('Selection Share (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(range(K_val))
        for bar in bars:
            h = bar.get_height()
            if h > 1.0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_mode_distribution_bars.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 4: Codebook Parameter Analysis
    # =========================================================================
    print("Generating Figure 4: Codebook Parameter Analysis...")
    fig = plt.figure(figsize=(18, 4.5))

    # (a) 3D Centroid Quivers on Unit Sphere (K=12)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax1.plot_wireframe(xs, ys, zs, color='gray', alpha=0.15, linewidth=0.5)

    c.execute("SELECT mode_k, centroid_x, centroid_y, centroid_z FROM cluster_centroids WHERE K=12 ORDER BY mode_k")
    c_pts = c.fetchall()
    colors_quiv = cm.tab20(np.linspace(0, 1, len(c_pts)))
    for idx, (mk, cx, cy, cz) in enumerate(c_pts):
        ax1.quiver(0, 0, 0, cx, cy, cz, color=colors_quiv[idx], linewidth=2.5, arrow_length_ratio=0.15)
        ax1.scatter([cx], [cy], [cz], color=colors_quiv[idx], s=40)
        ax1.text(cx*1.15, cy*1.15, cz*1.15, f'k{mk}', fontsize=8, fontweight='bold')

    ax1.set_title('(a) 3D Directional Centroids (K=12)', fontweight='bold')
    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # (b) Sink Fraction (SLP) Distribution
    ax2 = fig.add_subplot(1, 3, 2)
    slp_vals = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
    slp_by_b = {}
    for B in B_list:
        c.execute("SELECT slp, COUNT(*) FROM cluster_slopes WHERE child_b=? GROUP BY slp", (B,))
        counts = dict(c.fetchall())
        tot = sum(counts.values()) or 1
        slp_by_b[B] = [100.0 * counts.get(s, 0) / tot for s in slp_vals]

    x = np.arange(len(slp_vals))
    width = 0.2
    for bidx, B in enumerate(B_list):
        ax2.bar(x + (bidx - 1.5)*width, slp_by_b[B], width, label=f'B={B}', alpha=0.85)

    ax2.set_title('(b) Sink Fraction ($s_{\mathrm{lp}}$) Distribution by Scale B', fontweight='bold')
    ax2.set_xlabel('Sink Node Selection Fraction ($s_{\mathrm{lp}}$)')
    ax2.set_ylabel('Frequency (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s:.2f}' for s in slp_vals])
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    # (c) Self-Loop Weight (SLW) Distribution
    ax3 = fig.add_subplot(1, 3, 3)
    slw_vals = [0.25, 0.50, 0.80, 1.00, 1.40, 1.80]
    slw_by_b = {}
    for B in B_list:
        c.execute("SELECT slw, COUNT(*) FROM cluster_slopes WHERE child_b=? GROUP BY slw", (B,))
        counts = dict(c.fetchall())
        tot = sum(counts.values()) or 1
        slw_by_b[B] = [100.0 * counts.get(s, 0) / tot for s in slw_vals]

    x3 = np.arange(len(slw_vals))
    for bidx, B in enumerate(B_list):
        ax3.bar(x3 + (bidx - 1.5)*width, slw_by_b[B], width, label=f'B={B}', alpha=0.85)

    ax3.set_title('(c) Self-Loop Weight ($s_{\mathrm{lw}}$) Distribution by Scale B', fontweight='bold')
    ax3.set_xlabel('Self-Loop Diagonal Weight ($s_{\mathrm{lw}}$)')
    ax3.set_ylabel('Frequency (%)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels([f'{s:.2f}' for s in slw_vals])
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_codebook_slopes_analysis.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 5: Overhead Breakeven Curves
    # =========================================================================
    print("Generating Figure 5: Overhead Breakeven Curves...")
    fig, ax = plt.subplots(figsize=(8, 4.0))
    for Q in Q_list:
        ratios = []
        for B in B_list:
            c.execute("""SELECT g.effective_oh_bpv, g.baseline_bpv - g.hier_transform_bpv
                         FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn 
                                     FROM gating_results WHERE child_b=? AND K=12 AND q_step=? 
                                     GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=12 AND g.q_step=?""", (B, Q, B, Q))
            row = c.fetchone()
            ratios.append(row[0]/row[1] if (row and row[1]>1e-8) else np.nan)
        ax.plot([4, 8, 16, 32], ratios[::-1], 'o-', linewidth=2.0, markersize=7, label=f'Q = {Q} (K=12)')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=2.0, label='Breakeven Line (Ratio = 1.0)')
    ax.fill_between([4, 8, 16, 32], 1.0, 2.5, color='red', alpha=0.10)
    ax.fill_between([4, 8, 16, 32], 0.0, 1.0, color='green', alpha=0.08)
    ax.set_title('Signaling Overhead-to-Savings Ratio vs Block Scale B', fontweight='bold')
    ax.set_xlabel('Block Dimension B')
    ax.set_ylabel('OH : Savings Ratio')
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(0.2, 2.0)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_overhead_ratio_curves.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 6: Operational Phase Map
    # =========================================================================
    print("Generating Figure 6: Phase Map...")
    fig, ax = plt.subplots(figsize=(7, 4.0))
    corr_matrix = np.zeros((len(B_list), len(Q_list)))
    for bidx, B in enumerate(B_list):
        for qidx, Q in enumerate(Q_list):
            c.execute("SELECT g.run_id, g.delta_bpv FROM gating_results g WHERE g.child_b=? AND g.q_step=?", (B, Q))
            pairs = c.fetchall()
            xs, ys = [], []
            for rid, d in pairs:
                c.execute("SELECT SUM(activation_pct) FROM mode_activation_stats WHERE run_id=? AND mode_k>0", (rid,))
                gft_pct = c.fetchone()[0] or 0.0
                xs.append(gft_pct); ys.append(d)
            corr_matrix[bidx, qidx] = np.corrcoef(xs, ys)[0, 1] if len(xs) > 4 else 0.0

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')
    ax.set_title('Operational Phase Map: Pearson r(GFT Coverage%, ΔBPV)', fontweight='bold')
    ax.set_xticks(range(len(Q_list)))
    ax.set_xticklabels([f'Q={q}' for q in Q_list])
    ax.set_yticks(range(len(B_list)))
    ax.set_yticklabels([f'B={b}' for b in B_list])
    ax.set_xlabel('Quantization Step Q')
    ax.set_ylabel('Block Scale B')
    ax.contour(corr_matrix, levels=[0.0], colors='black', linestyles='--', linewidths=2.5)

    for i in range(len(B_list)):
        for j in range(len(Q_list)):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson r (Negative = Compaction, Positive = Overhead)', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig6_correlation_phase_map.png', bbox_inches='tight')
    plt.close()

    conn.close()
    print("Multi-perspective visual figures generated successfully!")

if __name__ == '__main__':
    generate_visuals()
