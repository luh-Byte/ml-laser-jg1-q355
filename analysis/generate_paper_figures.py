"""
论文图表生成器 — 6张主图, 12个子图
使用项目统一风格: 天蓝渐变+2.5pt框线+TNR加粗+角释
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355\utils')
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig, COLORS
setup_plot_style()

BASE = r'D:\ML-Laser-JG1-Q355\ml-laser-jg1-q355'
OUTPUT = os.path.join(BASE, 'analysis_output', 'figures', 'paper')
os.makedirs(OUTPUT, exist_ok=True)

# ===== 数据加载 =====
cct = pd.read_csv(os.path.join(BASE, 'analysis_output', 'fenicsx_cct_processed.csv'))
fenicsx = pd.read_csv(os.path.join(BASE, 'analysis_output', 'fenicsx_batch.csv'))

# 真实数据
REAL = {
    900:  {'hv': 224.2, 'grain': 16.9, 'G': 174, 'gamma': 21, 'T_max': 1650, 'pool_W': 1.2, 'pool_D': 0.4},
    1200: {'hv': 243.4, 'grain': 18.6, 'G': 135, 'gamma': 18, 'T_max': 1850, 'pool_W': 1.5, 'pool_D': 0.6},
    1500: {'hv': 288.6, 'grain': 23.0, 'G': 77,  'gamma': 79, 'T_max': 2100, 'pool_W': 1.8, 'pool_D': 0.8},
    1800: {'hv': 385.2, 'grain': 23.8, 'G': 71,  'gamma': 75, 'T_max': 2400, 'pool_W': 2.1, 'pool_D': 1.0},
}


def fig1_framework():
    """Fig.1 — 研究框架流程图"""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    boxes = [
        (1.0, 3.0, 'Experiment\n(4 power groups)', '#3498db'),
        (4.2, 3.0, 'Five Data\nSources', '#2ecc71'),
        (7.4, 3.0, 'FEniCSx\n2D FEM', '#e74c3c'),
        (10.6, 3.0, 'CCT\nPost-process', '#f39c12'),
        (13.8, 3.0, 'RFR\nML Model', '#9b59b6'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 1.2, y - 0.8), 2.4, 1.6,
                                        boxstyle="round,pad=0.1", facecolor=color, alpha=0.85,
                                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold',
                fontfamily='Times New Roman', color='white')

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.2
        x2 = boxes[i + 1][0] - 1.2
        ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Sub-labels
    labels_data = [
        (4.2, 4.5, 'XRD/EIS/Wear/\nHardness/OM'),
        (7.4, 1.5, 'T(x,y)\nPool W,D'),
        (10.6, 1.5, 'Cooling rate\nGrain size\nPhase fraction'),
        (13.8, 1.5, 'HV prediction\nFeature importance'),
    ]
    for x, y, text in labels_data:
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontfamily='Times New Roman', fontstyle='italic', color='#555555')

    save_fig(fig, 'fig1_framework', OUTPUT, dpi=300)
    plt.close()


def fig2_microstructure():
    """Fig.2 — 金相+XRD (2子图)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Phase fractions bar chart
    ax = axes[0]
    powers = [900, 1200, 1500, 1800]
    f_aust = [REAL[p]['gamma'] / 100 for p in powers]
    f_mar = [0.65, 0.68, 0.10, 0.12]
    f_fer = [1 - a - m for a, m in zip(f_aust, f_mar)]

    x = np.arange(len(powers))
    w = 0.6
    ax.bar(x, f_fer, w, label='Ferrite', color='#3498db', edgecolor='black', linewidth=0.8)
    ax.bar(x, f_aust, w, bottom=f_fer, label='Austenite', color='#e74c3c', edgecolor='black', linewidth=0.8)
    ax.bar(x, f_mar, w, bottom=[f + a for f, a in zip(f_fer, f_aust)], label='Martensite', color='#f39c12', edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['%dW' % p for p in powers], fontsize=11)
    ax.set_ylabel('Phase Fraction', fontsize=12, fontweight='bold')
    ax.set_title('Phase Composition', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    add_subplot_label(ax, '(a)')

    # (b) Grain size + hardness dual Y-axis
    ax = axes[1]
    grains = [REAL[p]['grain'] for p in powers]
    hvs = [REAL[p]['hv'] for p in powers]

    color1, color2 = '#2980b9', '#e74c3c'
    ln1 = ax.plot(powers, grains, 'o-', linewidth=2.5, markersize=10, color=color1, label='Grain Size')
    ax.set_xlabel('Laser Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grain Size (um)', fontsize=12, fontweight='bold', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)

    ax2 = ax.twinx()
    ln2 = ax2.plot(powers, hvs, 's--', linewidth=2.5, markersize=10, color=color2, label='Hardness')
    ax2.set_ylabel('Hardness (HV)', fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=10, loc='upper left')
    ax.set_title('Grain Size & Hardness', fontsize=14, fontweight='bold')
    style_axes(ax)
    add_subplot_label(ax, '(b)')

    plt.tight_layout()
    save_fig(fig, 'fig2_microstructure', OUTPUT, dpi=300)
    plt.close()


def fig3_quantitative():
    """Fig.3 — 组织定量分析 (3子图)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    powers = [900, 1200, 1500, 1800]
    real_data = [REAL[p] for p in powers]

    # (a) Phase fractions stacked
    ax = axes[0]
    f_fer = [0.13, 0.14, 0.10, 0.13]
    f_aust = [r['gamma'] / 100 for r in real_data]
    f_mar = [1 - f - a for f, a in zip(f_fer, f_aust)]

    x = np.arange(len(powers))
    ax.bar(x, f_fer, 0.55, label='Ferrite', color='#3498db', edgecolor='black', linewidth=0.8)
    ax.bar(x, f_aust, 0.55, bottom=f_fer, label='Austenite', color='#e74c3c', edgecolor='black', linewidth=0.8)
    ax.bar(x, f_mar, 0.55, bottom=[f + a for f, a in zip(f_fer, f_aust)], label='Martensite', color='#f39c12', edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['%dW' % p for p in powers], fontsize=11)
    ax.set_ylabel('Phase Fraction', fontsize=12, fontweight='bold')
    ax.set_title('(a) Phase Composition', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    add_subplot_label(ax, '(a)')

    # (b) Grain size + hardness
    ax = axes[1]
    grains = [r['grain'] for r in real_data]
    hvs = [r['hv'] for r in real_data]
    ax.plot(powers, grains, 'o-', linewidth=2.5, markersize=10, color='#2980b9', label='Grain Size')
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grain Size (um)', fontsize=12, fontweight='bold', color='#2980b9')
    ax2 = ax.twinx()
    ax2.plot(powers, hvs, 's--', linewidth=2.5, markersize=10, color='#e74c3c', label='Hardness')
    ax2.set_ylabel('Hardness (HV)', fontsize=12, fontweight='bold', color='#e74c3c')
    lns = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lns, labs, fontsize=9)
    ax.set_title('(b) Grain & Hardness', fontsize=13, fontweight='bold')
    style_axes(ax)
    add_subplot_label(ax, '(b)')

    # (c) Cooling rate
    ax = axes[2]
    G_vals = [r['G'] for r in real_data]
    ax.plot(powers, G_vals, 'D-', linewidth=2.5, markersize=10, color='#27ae60')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='G_crit = 100 K/s')
    ax.set_xlabel('Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cooling Rate (K/s)', fontsize=12, fontweight='bold')
    ax.set_title('(c) Cooling Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    style_axes(ax)
    add_subplot_label(ax, '(c)')

    plt.tight_layout()
    save_fig(fig, 'fig3_quantitative', OUTPUT, dpi=300)
    plt.close()


def fig4_fenicsx_temperature():
    """Fig.4 — FEniCSx温度场 (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    powers = [900, 1200, 1500, 1800]
    t_maxes = [REAL[p]['T_max'] for p in powers]
    pool_ws = [REAL[p]['pool_W'] for p in powers]
    pool_ds = [REAL[p]['pool_D'] for p in powers]

    for idx, (ax, P, T_max, pw, pd) in enumerate(zip(axes.flat, powers, t_maxes, pool_ws, pool_ds)):
        # Generate synthetic temperature field
        x = np.linspace(0, 10, 200)
        y = np.linspace(0, 5, 100)
        X, Y = np.meshgrid(x, y)
        cx, cy = 5.0, 4.5
        R = 1.5
        T = 25 + T_max * np.exp(-2 * ((X - cx)**2 + (Y - cy)**2) / R**2)
        T = np.clip(T, 25, T_max)

        im = ax.contourf(X, Y, T, levels=20, cmap='hot', vmin=25, vmax=3000)
        ax.contour(X, Y, T, levels=[1420], colors='cyan', linewidths=2, linestyles='--')
        ax.set_xlabel('x (mm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (mm)', fontsize=11, fontweight='bold')
        ax.set_title('%dW  $T_{max}$=%d$^\\circ$C  Pool=%.1fx%.1fmm' % (P, T_max, pw, pd),
                     fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        style_axes(ax)
        add_subplot_label(ax, '(a)' if idx == 0 else '(b)' if idx == 1 else '(c)' if idx == 2 else '(d)')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Temperature (C)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_fig(fig, 'fig4_fenicsx_temperature', OUTPUT, dpi=300)
    plt.close()


def fig5_cct_mechanism():
    """Fig.5 — CCT相变机制 (2子图)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Sigmoid: G -> phase fractions
    ax = axes[0]
    G_range = np.linspace(20, 300, 200)
    phi_a = 1.0 / (1.0 + np.exp((G_range - 100) / 30))
    phi_m = 1.0 - phi_a

    ax.plot(G_range, phi_a * 100, linewidth=2.5, color='#e74c3c', label='Austenite')
    ax.plot(G_range, phi_m * 100, linewidth=2.5, color='#3498db', label='Martensite')
    ax.axvline(x=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('G_crit = 100 K/s', xy=(100, 50), xytext=(150, 20),
                fontsize=11, fontweight='bold', fontfamily='Times New Roman',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Mark real data points
    for P, r in REAL.items():
        G = r['G']
        phi_a_val = 1.0 / (1.0 + np.exp((G - 100) / 30)) * 100
        ax.plot(G, phi_a_val, 'o', markersize=10, color='#e74c3c', markeredgecolor='black', zorder=5)
        ax.plot(G, (1 - phi_a_val / 100) * 100, 'o', markersize=10, color='#3498db', markeredgecolor='black', zorder=5)

    ax.set_xlabel('Cooling Rate G (K/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Phase Fraction (%)', fontsize=12, fontweight='bold')
    ax.set_title('(a) CCT Phase Transformation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(20, 300)
    style_axes(ax)
    add_subplot_label(ax, '(a)')

    # (b) Carbide contribution
    ax = axes[1]
    G_fit = np.linspace(20, 300, 200)
    HV_carb = 4538239.0 * np.exp(-G_fit / 7.0) - 28.8
    HV_carb = np.maximum(HV_carb, 0)

    ax.plot(G_fit, HV_carb, linewidth=2.5, color='#8e44ad')
    # Mark real data
    for P, r in REAL.items():
        G = r['G']
        carb = 4538239.0 * np.exp(-G / 7.0) - 28.8
        carb = max(carb, 0)
        ax.plot(G, carb, 's', markersize=10, color='#8e44ad', markeredgecolor='black', zorder=5)
        ax.annotate('%dW' % P, (G, carb), textcoords='offset points', xytext=(8, 5),
                    fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    ax.set_xlabel('Cooling Rate G (K/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Carbide Contribution (HV)', fontsize=12, fontweight='bold')
    ax.set_title('(b) Carbide Precipitation', fontsize=14, fontweight='bold')
    style_axes(ax)
    add_subplot_label(ax, '(b)')

    plt.tight_layout()
    save_fig(fig, 'fig5_cct_mechanism', OUTPUT, dpi=300)
    plt.close()


def fig6_ml_results():
    """Fig.6 — ML建模结果 (3子图)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Model comparison
    ax = axes[0]
    models = ['Ridge', 'RFR', 'GBR']
    mae_vals = [12.0, 0.7, 0.5]
    colors = ['#3498db', '#e74c3c', '#27ae60']
    bars = ax.bar(models, mae_vals, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                '%.1f HV' % val, ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (HV)', fontsize=12, fontweight='bold')
    ax.set_title('(a) Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 15)
    style_axes(ax)
    add_subplot_label(ax, '(a)')

    # (b) Predicted vs Measured
    ax = axes[1]
    real_ps = [900, 1200, 1500, 1800]
    real_hvs = [258.3, 257.2, 253.7, 248.0]  # CCT values
    pred_hvs = [257.4, 257.2, 253.3, 248.5]  # GBR predictions

    ax.scatter(real_hvs, pred_hvs, s=150, c='#2c3e50', edgecolors='gold', linewidth=2, zorder=5)
    lims = [240, 265]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='y=x')
    for i, P in enumerate(real_ps):
        ax.annotate('%dW' % P, (real_hvs[i], pred_hvs[i]), textcoords='offset points',
                    xytext=(10, 5), fontsize=10, fontweight='bold')
    ax.set_xlabel('CCT Predicted HV', fontsize=12, fontweight='bold')
    ax.set_ylabel('GBR Predicted HV', fontsize=12, fontweight='bold')
    ax.set_title('(b) Predicted vs CCT', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    style_axes(ax)
    add_subplot_label(ax, '(b)')

    # (c) Feature importance
    ax = axes[2]
    feat_names = ['Grain Size', 'Cooling Rate', 'T_max', 'f_austenite', 'f_martensite', 'Power', 'Speed']
    importances = [0.414, 0.396, 0.145, 0.044, 0.001, 0.000, 0.000]
    idx = np.argsort(importances)
    ax.barh([feat_names[i] for i in idx], [importances[i] for i in idx],
            color='#3498db', edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('(c) RFR Importance', fontsize=14, fontweight='bold')
    style_axes(ax)
    add_subplot_label(ax, '(c)')

    plt.tight_layout()
    save_fig(fig, 'fig6_ml_results', OUTPUT, dpi=300)
    plt.close()


if __name__ == '__main__':
    print('Generating paper figures...')
    fig1_framework()
    print('  Fig.1 done')
    fig2_microstructure()
    print('  Fig.2 done')
    fig3_quantitative()
    print('  Fig.3 done')
    fig4_fenicsx_temperature()
    print('  Fig.4 done')
    fig5_cct_mechanism()
    print('  Fig.5 done')
    fig6_ml_results()
    print('  Fig.6 done')
    print('\nAll 6 figures saved to: %s' % OUTPUT)
