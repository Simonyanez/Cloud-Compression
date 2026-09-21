"""
Comprehensive Visual Figure Suite for PCADC Single-Column Audit Report
========================================================================
Figures generated:
1. fig1_rd_curves_zoomed.png: 4-Panel Multi-Scale RD curves with zoomed Inset Windows and No-Overhead bounds.
2. fig2_cluster_activation_per_scale.png: Faceted Subplots showing cluster activation P(M=k) per block size B in {32, 16, 8, 4} across K in {4, 8, 12, 16}.
3. fig3_tau_gate_sensitivity.png: Dual-panel analysis showing Bypass Rate % vs tau_gate and Net Gain $\Delta vs tau_gate across scales.
4. fig4_codebook_weighted_slopes.png: 3D Directional Centroid orientations (raw vs empirically activation-weighted) and SLP / SLW distributions.
5. fig5_overhead_ratio_curves.png: Overhead-to-Savings ratio vs B showing the y=1.0 breakeven line.
6. fig6_correlation_phase_map.png: Operational Phase Transition Map (Pearson r across B x Q).
7. fig7_algorithmic_sensitivity_matrix.png: Parameter Sensitivity Heatmap correlating internal controls (tau_gate, K, gamma_0, slp, slw) against BD-Rate, switching frequency, and OH:DeltaE.
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def generate_all_figures():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c = conn.cursor()

    B_list = [32, 16, 8, 4]
    Q_list = [24, 32, 48, 64]
    K_list = [4, 8, 10, 12, 16]
    k_colors = {4: '#4e79a7', 8: '#f28e2b', 10: '#e15759', 12: '#2b908f', 16: '#59a14f'}
    k_markers = {4: 'o', 8: 's', 10: '^', 12: 'D', 16: 'v'}

    # =========================================================================
    # FIG 1: RD Curves with Zoomed Insets and Ideal No-Overhead Bounds
    # =========================================================================
    print("Generating Figure 1: RD Curves with Inset Zooms & No-Overhead Bound...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, B in enumerate(B_list):
        ax = axes[idx]
        
        # Baseline
        c.execute("SELECT q_step, baseline_bpv, baseline_psnr_db FROM gating_results WHERE child_b=? AND K=4 GROUP BY q_step ORDER BY q_step", (B,))
        b_pts = c.fetchall()
        if b_pts:
            b_qs, b_bpv, b_psnr = zip(*b_pts)
            ax.plot(b_bpv, b_psnr, 'k*--', linewidth=2.0, markersize=9, label='Structural Baseline', zorder=7)

        # PCADC Net
        for K in K_list:
            c.execute("""SELECT g.q_step, g.net_bpv, g.psnr_db FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=? GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? ORDER BY g.q_step""", (B, K, B, K))
            pts = c.fetchall()
            if pts:
                qs, bpv, psnr = zip(*pts)
                ax.plot(bpv, psnr, color=k_colors[K], marker=k_markers[K], linewidth=1.4, markersize=5.5, label=f'PCADC K={K} (Net)')

        # Ideal No-Overhead
        best_K = 16 if B == 32 else 12
        c.execute("""SELECT g.q_step, g.hier_transform_bpv, g.psnr_db FROM gating_results g
                     INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=? GROUP BY child_b, K, q_step) b
                     ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                     WHERE g.child_b=? AND g.K=? ORDER BY g.q_step""", (B, best_K, B, best_K))
        no_oh_pts = c.fetchall()
        if no_oh_pts:
            no_oh_qs, no_oh_bpv, no_oh_psnr = zip(*no_oh_pts)
            ax.plot(no_oh_bpv, no_oh_psnr, color='#9467bd', linestyle=':', marker='x', linewidth=2.0, markersize=7,
                    label=f'PCADC K={best_K} (No Overhead Limit)', zorder=6)

        ax.set_title(f'Rate-Distortion Performance: Scale B = {B}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Bitrate (BPV)')
        ax.set_ylabel('Attribute PSNR (dB)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='lower right', fontsize=8.5)

        # Inset Zoom @ Q=24
        axins1 = inset_axes(ax, width="40%", height="30%", loc='center',
                            bbox_to_anchor=(0.04, 0.48, 0.9, 0.9), bbox_transform=ax.transAxes)
        c.execute("SELECT baseline_bpv, baseline_psnr_db FROM gating_results WHERE child_b=? AND q_step=24 LIMIT 1", (B,))
        q24_base = c.fetchone()
        if q24_base:
            axins1.scatter([q24_base[0]], [q24_base[1]], color='black', marker='*', s=120, zorder=8)

        q24_bpvs = [q24_base[0]] if q24_base else []
        q24_psnrs = [q24_base[1]] if q24_base else []
        for K in K_list:
            c.execute("""SELECT g.net_bpv, g.psnr_db, g.hier_transform_bpv FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=? AND q_step=24 GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=? AND g.q_step=24""", (B, K, B, K))
            r = c.fetchone()
            if r:
                axins1.scatter([r[0]], [r[1]], color=k_colors[K], marker=k_markers[K], s=55, zorder=7)
                q24_bpvs.append(r[0]); q24_psnrs.append(r[1])
                if K == best_K:
                    axins1.scatter([r[2]], [r[1]], color='#9467bd', marker='x', s=75, linewidth=2.0, zorder=7)
                    q24_bpvs.append(r[2])

        if q24_bpvs:
            bpv_pad = (max(q24_bpvs) - min(q24_bpvs)) * 0.20 or 0.001
            psnr_pad = (max(q24_psnrs) - min(q24_psnrs)) * 0.35 or 0.015
            axins1.set_xlim(min(q24_bpvs) - bpv_pad, max(q24_bpvs) + bpv_pad)
            axins1.set_ylim(min(q24_psnrs) - psnr_pad, max(q24_psnrs) + psnr_pad)
            axins1.set_title(f'Zoom @ Q=24 (Net Δ={-(q24_base[0]-min(q24_bpvs))*1000:.1f} mBPV)', fontsize=8, fontweight='bold', pad=3)
            axins1.tick_params(axis='both', which='major', labelsize=7)
            mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_rd_curves_zoomed.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 2: Cluster Activation Dynamics (Per Block Size Faceted Subplots)
    # =========================================================================
    print("Generating Figure 2: Cluster Activation Faceted Subplots per Block Size...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    for idx, B in enumerate(B_list):
        ax = axes[idx]
        
        # Plot distribution for K=12 at Q=24, comparing mode shares
        c.execute("""SELECT g.run_id FROM gating_results g
                     INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=12 AND q_step=24 GROUP BY child_b, K, q_step) b
                     ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                     WHERE g.child_b=? AND g.K=12 AND g.q_step=24""", (B, B))
        r12 = c.fetchone()
        
        c.execute("""SELECT g.run_id FROM gating_results g
                     INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=4 AND q_step=24 GROUP BY child_b, K, q_step) b
                     ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                     WHERE g.child_b=? AND g.K=4 AND g.q_step=24""", (B, B))
        r4 = c.fetchone()

        if r12 and r4:
            c.execute("SELECT mode_k, activation_pct FROM mode_activation_stats WHERE run_id=? ORDER BY mode_k", (r12[0],))
            m12 = dict(c.fetchall())
            c.execute("SELECT mode_k, activation_pct FROM mode_activation_stats WHERE run_id=? ORDER BY mode_k", (r4[0],))
            m4 = dict(c.fetchall())

            modes = np.arange(12)
            pcts_k12 = [m12.get(k, 0.0) for k in modes]
            pcts_k4 = [m4.get(k, 0.0) if k < 4 else 0.0 for k in modes]

            width = 0.38
            ax.bar(modes - width/2, pcts_k4, width, label='K=4 (Starved)', color='#e15759', alpha=0.85, edgecolor='black')
            ax.bar(modes + width/2, pcts_k12, width, label='K=12 (Rich Multi-Mode)', color='#2b908f', alpha=0.85, edgecolor='black')

            ax.set_title(f'Cluster Activation Distribution: Scale B = {B} (Q=24)', fontweight='bold')
            ax.set_xlabel('Codebook Mode Index k (0: Fallback, 1..11: Directional)')
            ax.set_ylabel('Selection Share (%)')
            ax.set_xticks(modes)
            ax.set_ylim(0, 100)
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

            # Annotate fallback bypass %
            ax.text(0, max(pcts_k12[0], pcts_k4[0]) + 3.0, f"k0={pcts_k12[0]:.1f}%", ha='center', fontsize=8, fontweight='bold', color='#2b908f')

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_cluster_activation_per_scale.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 3: tau_gate Sensitivity & Operational Gating Dynamics
    # =========================================================================
    print("Generating Figure 3: tau_gate Gating Sensitivity Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    tau_vals = [0.0, 50.0, 150.0, 300.0, 500.0, 800.0, 1200.0, 1600.0]

    for B in B_list:
        c.execute("""SELECT tau_gate, gating_ratio_pct, delta_bpv * 1000.0 
                     FROM gating_results WHERE child_b=? AND K=12 AND q_step=24 
                     ORDER BY tau_gate""", (B,))
        rows = c.fetchall()
        if rows:
            tg, bypass, d_mbpv = zip(*rows)
            ax1.plot(tg, bypass, 'o-', linewidth=2.0, markersize=6, label=f'Scale B={B}')
            ax2.plot(tg, d_mbpv, 's-', linewidth=2.0, markersize=6, label=f'Scale B={B}')

    ax1.set_title('(a) Fallback Bypass Rate (% Blocks Routed to Mode 0) vs $\\tau_{\\mathrm{gate}}$', fontweight='bold')
    ax1.set_xlabel('Parent Luminance Variance Threshold ($\\tau_{\\mathrm{gate}}$)')
    ax1.set_ylabel('Bypassed Parent Blocks (%)')
    ax1.set_xticks(tau_vals)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_title('(b) Net Coding Gain $\\Delta\\mathrm{BPV}$ vs $\\tau_{\\mathrm{gate}}$ (Q=24, K=12)', fontweight='bold')
    ax2.set_xlabel('Parent Luminance Variance Threshold ($\\tau_{\\mathrm{gate}}$)')
    ax2.set_ylabel('$\\Delta\\mathrm{BPV}$ ($10^{-3}$ BPV, Negative = Net Gain)')
    ax2.set_xticks(tau_vals)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    # Highlight bimodal optimal gating valley
    ax2.axvspan(50.0, 300.0, color='green', alpha=0.10, label='Optimal Texture Filtering Window')

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_tau_gate_sensitivity.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 4: Codebook Directional Centroids & Activation-Weighted Energy Compaction
    # =========================================================================
    print("Generating Figure 4: Codebook & Activation-Weighted Directional Slopes...")
    fig = plt.figure(figsize=(18, 5))

    # (a) 3D Centroid Quivers
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

    ax1.set_title('(a) Learned 3D Directional Centroids (K=12)', fontweight='bold')
    ax1.set_xlim([-1.2, 1.2]); ax1.set_ylim([-1.2, 1.2]); ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

    # (b) Empirical Activation-Weighted Energy Compaction per Mode
    ax2 = fig.add_subplot(1, 3, 2)
    c.execute("""SELECT cc.mode_k, SUM(mas.n_activations) as tot_act, AVG(mas.mean_rate_savings_pct) as avg_sav
                 FROM cluster_centroids cc
                 JOIN gating_results gr ON gr.K=cc.K
                 JOIN mode_activation_stats mas ON mas.run_id=gr.run_id AND mas.mode_k=cc.mode_k
                 WHERE cc.K=12
                 GROUP BY cc.mode_k ORDER BY cc.mode_k""")
    act_data = c.fetchall()
    if act_data:
        modes, acts, savs = zip(*act_data)
        norm_acts = np.array(acts) / max(acts)
        bars = ax2.bar(modes, savs, color=colors_quiv, edgecolor='black', alpha=0.85)
        ax2.set_title('(b) Directional Mode Compaction Efficiency (K=12)', fontweight='bold')
        ax2.set_xlabel('Directional Mode Index k')
        ax2.set_ylabel('Mean Rate Savings on Selected Blocks (%)')
        ax2.set_xticks(modes)
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{acts[idx]} blks', ha='center', va='bottom', fontsize=7, rotation=35)

    # (c) Sink Fraction & Self-Loop Parameterization
    ax3 = fig.add_subplot(1, 3, 3)
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
        ax3.bar(x + (bidx - 1.5)*width, slp_by_b[B], width, label=f'B={B}', alpha=0.85)

    ax3.set_title('(c) Sink Fraction ($s_{\mathrm{lp}}$) Parameter Selection', fontweight='bold')
    ax3.set_xlabel('Sink Fraction ($s_{\mathrm{lp}}$)')
    ax3.set_ylabel('Frequency (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{s:.2f}' for s in slp_vals])
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_codebook_weighted_slopes.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 5: Overhead-to-Savings Ratio Breakeven Curves
    # =========================================================================
    print("Generating Figure 5: Overhead Breakeven Curves...")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for Q in Q_list:
        ratios = []
        for B in B_list:
            c.execute("""SELECT g.effective_oh_bpv, g.baseline_bpv - g.hier_transform_bpv
                         FROM gating_results g
                         INNER JOIN (SELECT child_b, K, q_step, MIN(net_bpv) as mn FROM gating_results WHERE child_b=? AND K=12 AND q_step=? GROUP BY child_b, K, q_step) b
                         ON g.child_b=b.child_b AND g.K=b.K AND g.q_step=b.q_step AND g.net_bpv=b.mn
                         WHERE g.child_b=? AND g.K=12 AND g.q_step=?""", (B, Q, B, Q))
            row = c.fetchone()
            ratios.append(row[0]/row[1] if (row and row[1]>1e-8) else np.nan)
        ax.plot([4, 8, 16, 32], ratios[::-1], 'o-', linewidth=2.0, markersize=7, label=f'Q = {Q} (K=12)')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=2.0, label='Breakeven Frontier (OH : Savings = 1.0)')
    ax.fill_between([4, 8, 16, 32], 1.0, 2.5, color='red', alpha=0.10)
    ax.fill_between([4, 8, 16, 32], 0.0, 1.0, color='green', alpha=0.08)
    ax.set_title('Signaling Overhead-to-Savings Ratio vs Block Scale B', fontweight='bold')
    ax.set_xlabel('Block Dimension B')
    ax.set_ylabel('OH : Transform Savings Ratio')
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(0.2, 2.0)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_overhead_ratio_curves.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 6: Correlation Phase Transition Map
    # =========================================================================
    print("Generating Figure 6: Operational Phase Map...")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
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
    cbar.set_label('Pearson r (Negative = Compaction Domain, Positive = Overhead Domain)', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig6_correlation_phase_map.png', bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIG 7: Algorithmic Parameter Sensitivity Matrix
    # =========================================================================
    print("Generating Figure 7: Algorithmic Parameter Sensitivity Heatmap...")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # Controls: tau_gate, K, gamma_0 (or gamma_q), slp, slw
    # Metrics: delta_bpv, n_viterbi_switches, OH:savings ratio, active_edge_compaction
    c.execute("""
    SELECT gr.tau_gate, gr.K, gr.gamma_q, cs.slp, cs.slw,
           gr.delta_bpv, gr.n_viterbi_switches, 
           gr.effective_oh_bpv / NULLIF(gr.baseline_bpv - gr.hier_transform_bpv, 0) as oh_ratio,
           gr.active_edge_compaction_pct
    FROM gating_results gr
    JOIN cluster_slopes cs ON cs.run_id=gr.run_id
    WHERE gr.delta_bpv IS NOT NULL AND cs.mode_k > 0
    """)
    raw_sens = c.fetchall()
    
    sens_arr = np.array([[float(x) if (x is not None and not np.isnan(x) and not np.isinf(x)) else 0.0 for x in r] for r in raw_sens])
    
    controls = ['tau_gate', 'K', 'gamma_q', 'slp', 'slw']
    metrics = ['$\Delta', 'Viterbi Switches', 'OH : Savings Ratio', 'Compaction %']
    
    sens_matrix = np.zeros((len(controls), len(metrics)))
    for ci_idx in range(len(controls)):
        for mi_idx in range(len(metrics)):
            val_c = sens_arr[:, ci_idx]
            val_m = sens_arr[:, len(controls) + mi_idx]
            mask = np.isfinite(val_c) & np.isfinite(val_m)
            sens_matrix[ci_idx, mi_idx] = np.corrcoef(val_c[mask], val_m[mask])[0, 1]

    im_sens = ax.imshow(sens_matrix, cmap='PiYG', vmin=-0.8, vmax=0.8, aspect='auto')
    ax.set_title('Algorithmic Parameter Sensitivity Matrix', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_yticks(range(len(controls)))
    ax.set_yticklabels([r'$\tau_{\mathrm{gate}}$', '$K$', r'$\gamma(Q)$', r'$s_{\mathrm{lp}}$', r'$s_{\mathrm{lw}}$'], fontweight='bold', fontsize=11)

    for i in range(len(controls)):
        for j in range(len(metrics)):
            v = sens_matrix[i, j]
            col = 'white' if abs(v) > 0.45 else 'black'
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center', color=col, fontweight='bold', fontsize=9)

    cbar_sens = fig.colorbar(im_sens, ax=ax)
    cbar_sens.set_label('Pearson Correlation Coefficient $r$', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig7_algorithmic_sensitivity_matrix.png', bbox_inches='tight')
    plt.close()

    conn.close()
    print("All figures successfully generated in docs/report/figures/!")

if __name__ == '__main__':
    generate_all_figures()
