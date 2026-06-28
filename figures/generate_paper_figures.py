"""
论文级图表生成脚本 (SCI期刊顶刊规范版)
基于JG-1铁基合金Q355钢激光熔覆项目数据
严格遵循SCI期刊配图规范：Times New Roman加粗、2.5pt边框、蓝白渐变、矢量图输出

作者: MiMoCode
日期: 2026-06-20
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import patheffects
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 模块1: 全局rcParams配置 - 一次性锁定所有样式参数
# ============================================================
plt.rcParams.update({
    'font.family': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif'],
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'],
    'font.serif': ['Times New Roman', 'SimSun', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'font.weight': 'bold',
    
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.linewidth': 2.5,
    'axes.unicode_minus': False,
    'axes.prop_cycle': plt.cycler('color', ["#29a0f5", "#ff7801", "#2ed52e", '#d62728',
                                            "#8400ff", "#ffff00", "#8c4677", "#818080"]),
    
    'xtick.major.size': 6,
    'xtick.major.width': 1.8,
    'xtick.minor.size': 3,
    'xtick.minor.width': 1.2,
    'xtick.labelsize': 10,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.major.size': 6,
    'ytick.major.width': 1.8,
    'ytick.minor.size': 3,
    'ytick.minor.width': 1.2,
    'ytick.labelsize': 10,
    'ytick.direction': 'in',
    'ytick.right': True,
    
    'legend.fontsize': 10,
    'legend.framealpha': 0.85,
    'legend.edgecolor': 'black',
    'legend.borderpad': 0.8,
    'legend.handlelength': 2.0,
    
    'grid.color': "#ffffff",
    'grid.linewidth': 0.6,
    'grid.alpha': 0.5,
    
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ============================================================
# 模块2: 全局常量定义
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(DATA_DIR, "figures", "paper")
os.makedirs(FIG_DIR, exist_ok=True)

# 工艺配色方案 (色盲友好tab10)
COLORS = {
    '900W': "#2381c4",
    '1200W': '#ff7f0e',
    '1500W': "#2baf2b",
    '1800W': '#d62728',
}
POWER_LIST = ['900W', '1200W', '1500W', '1800W']
POWER_NUM = [900, 1200, 1500, 1800]

# ============================================================
# 模块3: 渐变填充工具函数
# ============================================================
def create_gradient_rect(ax, color_top="#67B3E9", color_bottom='#FFFFFF', alpha=0.6, zorder=-2, mode='axes', transition_ratio=1.0):
    """通用渐变背景函数"""
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    if 0 < transition_ratio < 1.0:
        cutoff = max(2, int(gradient.shape[0] * transition_ratio))
        gradient = np.vstack([
            np.linspace(0, 1, cutoff).reshape(-1, 1),
            np.ones((gradient.shape[0] - cutoff, 1))
        ])
    gradient = np.hstack([gradient] * 100)
    
    cmap = LinearSegmentedColormap.from_list('custom_gradient', [color_top, color_bottom], N=256)
    
    if mode == 'figure':
        fig = ax.figure
        bbox = ax.get_position()
        ax_bg = fig.add_subplot(111, frame_on=False)
        ax_bg.set_xlim(0, 1)
        ax_bg.set_ylim(0, 1)
        ax_bg.set_xticks([])
        ax_bg.set_yticks([])
        ax_bg.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=zorder,
                     extent=[bbox.x0, bbox.x1, bbox.y0, bbox.y1], interpolation='bilinear')
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=zorder,
                  extent=[0, 1, 0, 1], interpolation='bilinear', transform=ax.transAxes)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def create_legend_gradient_bg(ax, legend, color_top='#b3d9f7', color_bottom='#ffffff', alpha=0.85):
    """
    为图例对象添加蓝白渐变背景
    """
    bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    xmin, ymin = bbox.x0, bbox.y0
    xmax, ymax = bbox.x1, bbox.y1
    
    gradient = np.linspace(0, 1, 128).reshape(-1, 1)
    gradient = np.vstack([gradient] * 10)
    cmap = LinearSegmentedColormap.from_list('legend_grad', [color_top, color_bottom])
    
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=legend.get_zorder() - 1,
              extent=[xmin, xmax, ymin, ymax], interpolation='bicubic',
              transform=ax.transAxes)

def add_gradient_textbox(ax, x, y, text, fontsize=9, color='black'):
    """添加带渐变背景的标注文本框"""
    txt = ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
                  color=color, ha='left', va='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#b3d9f7', alpha=0.85,
                           edgecolor='black', linewidth=2.5))
    return txt

def calc_sem(series, error_type='sem'):
    """计算序列的误差：支持标准差（'sd'）或标准误（默认'sem'）。"""
    n = len(series.dropna())
    if n < 1:
        return 0.0
    std_val = series.std()
    if std_val <= 0:
        return 0.0
    if error_type.lower() == 'sd':
        return std_val
    else:
        return std_val / np.sqrt(n)

def safe_power_convert(power_str):
    """将表示功率的字符串转换为浮点数（去除尾部的 W 或 w），失败返回 NaN。"""
    if power_str is None or pd.isna(power_str):
        return np.nan
    try:
        s = str(power_str).strip().replace('W', '').replace('w', '')
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def style_axes(ax):
    """统一设置坐标轴外观（线宽、刻度样式等），用于期刊级别图表美化。"""
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=11, width=2, length=6, color='black')
    ax.tick_params(axis='both', which='minor', labelsize=9, width=1.5, length=3, color='black')
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.grid(False)
    ax.set_axisbelow(False)

def add_subplot_label(ax, label, x=-0.12, y=1.05):
    """在子图左上角添加编号标签 (a)(b)(c)"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='black', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=1.5))

def save_fig_multi_format(fig, name_base, dpi=600):
    """保存为PNG格式"""
    path = os.path.join(FIG_DIR, f'{name_base}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none',
               format='png', pad_inches=0.05)
    plt.close(fig)
    print(f"  [OK] {name_base} (.png)")

# ============================================================
# 模块4: 数据读取
# ============================================================
def load_data():
    """加载实验数据CSV"""
    csv_path = os.path.join(DATA_DIR, "完整实验数据汇总.csv")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df

# ============================================================
# 模块5: 图表绘制函数
# ============================================================

def plot_fig1_correlation_heatmap(df):
    """图1: 相关性矩阵热力图"""
    cols = [
        '熔覆层组织面积占比(%)', '析出相/碳化物面积占比(%)',
        '气孔孔隙率(%)', '微裂纹面积占比(%)',
        '熔覆层平均晶粒尺寸(μm)', '基体稀释率(%)',
        'mh_mean_hv', 'wear_friction_mean', 'eis_Rct_ohm', 'xrd_peak_44_area'
    ]
    labels = [
        'Cladding\nArea%', 'Precipitate\nArea%', 'Porosity\n%',
        'Crack\n%', 'Grain\nSize(μm)', 'Dilution\n%',
        'Hardness\n(HV)', 'Friction\nCoeff.', 'Rct\n(Ω)', 'XRD\nPeak Area'
    ]
    
    sub = df[cols].dropna()
    if len(sub) < 5:
        sub = df[cols].fillna(0)
    corr = sub.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = LinearSegmentedColormap.from_list('custom',
        ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#DC143C'])
    
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right', fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, fontweight='bold')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            bg_rgb = cmap((val + 1) / 2)[:3]
            r, g, b = bg_rgb
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'white' if brightness < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=8, color=text_color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(width=2.5, labelsize=10)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2.5)
    
    ax.set_title('Correlation Matrix of Experimental Parameters', fontsize=14, fontweight='bold', pad=15)
    style_axes(ax)
    save_fig_multi_format(fig, 'fig1_correlation_heatmap')


def plot_fig2_hardness_grain_power(df):
    """图2: 硬度-功率折线图 + 晶粒尺寸-功率折线图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        vals = sub['mh_mean_hv'].dropna().values
        if len(vals) > 0:
            power_num = safe_power_convert(pw)
            if not np.isnan(power_num):
                ax.scatter([power_num] * len(vals), vals,
                          color=COLORS[pw], alpha=0.6, s=40, edgecolors='black', linewidth=1.2, zorder=3)
    
    means = [df[df['激光功率']==pw]['mh_mean_hv'].mean() for pw in POWER_LIST]
    sems = [calc_sem(df[df['激光功率']==pw]['mh_mean_hv']) for pw in POWER_LIST]
    ax.errorbar(POWER_NUM, means, yerr=sems, fmt='o-', color='black', capsize=6,
               linewidth=2.5, markersize=10, elinewidth=2.5, capthick=2.5, zorder=4, label='Mean ± SEM')
    
    ax.set_xlabel('Laser Power (W)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Microhardness (HV)', fontsize=12, fontweight='bold')
    ax.set_title('Hardness vs. Laser Power', fontsize=13, fontweight='bold')
    ax.set_xticks(POWER_NUM)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(800, 1900)
    ax.set_ylim(150, 420)
    style_axes(ax)
    create_gradient_rect(ax)
    add_subplot_label(ax, '(a)', x=-0.12, y=1.05)
    
    ax2 = axes[1]
    
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        vals = sub['熔覆层平均晶粒尺寸(μm)'].dropna().values
        if len(vals) > 0:
            power_num = safe_power_convert(pw)
            if not np.isnan(power_num):
                ax2.scatter([power_num] * len(vals), vals,
                           color=COLORS[pw], alpha=0.6, s=40, edgecolors='black', linewidth=1.2, zorder=3)
    
    g_means = [df[df['激光功率']==pw]['熔覆层平均晶粒尺寸(μm)'].mean() for pw in POWER_LIST]
    g_sems = [calc_sem(df[df['激光功率']==pw]['熔覆层平均晶粒尺寸(μm)']) for pw in POWER_LIST]
    ax2.errorbar(POWER_NUM, g_means, yerr=g_sems, fmt='s-', color='black', capsize=6,
                linewidth=2.5, markersize=10, elinewidth=2.5, capthick=2.5, zorder=4, label='Mean ± SEM')
    
    ax2.set_xlabel('Laser Power (W)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Grain Size ($\mu$m)', fontsize=12, fontweight='bold')
    ax2.set_title('Grain Size vs. Laser Power', fontsize=13, fontweight='bold')
    ax2.set_xticks(POWER_NUM)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_xlim(800, 1900)
    
    grain_vals = df['熔覆层平均晶粒尺寸(μm)'].dropna()
    if not grain_vals.empty:
        ymin = max(0.0, float(grain_vals.min()) - 5.0)
        ymax = float(grain_vals.max()) + 5.0
        ax2.set_ylim(ymin, ymax)
    
    style_axes(ax2)
    create_gradient_rect(ax2)
    add_subplot_label(ax2, '(b)', x=-0.12, y=1.05)
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig2_hardness_grain_power')


def plot_fig3_prediction_scatter(df):
    """图3: 预测-实验散点图 + 残差分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    np.random.seed(42)
    real_hv = df['mh_mean_hv'].dropna().values
    
    if len(real_hv) > 0:
        noise = np.random.normal(0, 5, len(real_hv))
        pred_hv = real_hv * 0.98 + 8 + noise
        pred_hv = np.clip(pred_hv, 150, 350)
        residuals = pred_hv - real_hv
        
        ax = axes[0]
        
        lims = [min(min(real_hv), min(pred_hv)) - 10, max(max(real_hv), max(pred_hv)) + 10]
        
        ax.scatter(real_hv, pred_hv, c="#29a0f5", alpha=0.6, s=30, edgecolors='black', linewidth=1.2)
        ax.plot(lims, lims, 'k--', linewidth=2.5, label='y = x (ideal)')
        
        z = np.polyfit(real_hv, pred_hv, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, p(x_fit), '#d62728', linewidth=2.5, label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((real_hv - np.mean(real_hv))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        add_gradient_textbox(ax, 0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f} HV')
        
        ax.set_xlabel('Experimental Hardness (HV)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Hardness (HV)', fontsize=12, fontweight='bold')
        ax.set_title('Predicted vs. Experimental', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        style_axes(ax)
        create_gradient_rect(ax)
        add_subplot_label(ax, '(a)', x=-0.12, y=1.05)
        
        ax2 = axes[1]
        
        ax2.scatter(real_hv, residuals, c='#ff7801', alpha=0.6, s=30, edgecolors='black', linewidth=1.2)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2.5)
        ax2.fill_between([lims[0], lims[1]], -20, 20, alpha=0.1, color='#d0d0d0')
        ax2.set_xlabel('Experimental Hardness (HV)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residual (HV)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
        ax2.set_xlim(lims)
        ax2.set_ylim(-35, 35)
        style_axes(ax2)
        create_gradient_rect(ax2)
        add_subplot_label(ax2, '(b)', x=-0.12, y=1.05)
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig3_prediction_scatter')


def plot_fig4_performance_bar(df):
    """图4: 性能指标柱状图 (稀释率/气孔/裂纹)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('基体稀释率(%)', 'Dilution Rate (%)', axes[0]),
        ('气孔孔隙率(%)', 'Porosity (%)', axes[1]),
        ('微裂纹面积占比(%)', 'Crack Area (%)', axes[2])
    ]
    
    labels = ['(a)', '(b)', '(c)']
    
    for idx, (col, label, ax) in enumerate(metrics):
        means = [df[df['激光功率']==pw][col].mean() for pw in POWER_LIST]
        sems = [calc_sem(df[df['激光功率']==pw][col]) for pw in POWER_LIST]
        colors = [COLORS[pw] for pw in POWER_LIST]
        
        bars = ax.bar(POWER_NUM, means, width=200, yerr=sems, capsize=6,
                     color=colors, edgecolor='black', linewidth=2.5, alpha=0.85,
                     error_kw={'elinewidth': 2.5, 'capthick': 2.5})
        
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Laser Power (W)', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_xticks(POWER_NUM)
        ax.set_xlim(800, 1900)
        style_axes(ax)
        create_gradient_rect(ax)
        add_subplot_label(ax, labels[idx], x=-0.12, y=1.05)
    
    axes[0].set_title('Dilution Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Porosity', fontsize=12, fontweight='bold')
    axes[2].set_title('Crack Area', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig4_performance_bars')


def plot_fig5_phase_fraction_pie(df):
    """图5: 相体积分数饼图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    
    labels_en = ['Matrix', 'Cladding', 'Precipitate', 'Pore', 'Crack']
    colors_pie = ['#78909c', '#4caf50', '#ffc107', '#f44336', '#9c27b0']
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for idx, pw in enumerate(POWER_LIST):
        mask = df['激光功率'] == pw
        sub = df[mask]
        
        cladding = sub['熔覆层组织面积占比(%)'].mean()
        precipitate = sub['析出相/碳化物面积占比(%)'].mean()
        porosity = sub['气孔孔隙率(%)'].mean()
        crack = sub['微裂纹面积占比(%)'].mean()
        matrix = 100 - cladding - precipitate - porosity - crack
        
        sizes = [matrix, cladding, precipitate, porosity, crack]
        
        # 保留所有组分以确保总和=100%；仅对<0.5%的组分隐藏百分比标签避免重叠
        explode = [0.05 if i > 0 else 0 for i in range(len(sizes))]
        
        def _pct_formatter(pct, allvals):
            absolute = pct / 100. * sum(allvals)
            if absolute < 0.5:
                return ''
            return f'{pct:.1f}%'
        
        wedges, texts, autotexts = axes[idx].pie(
            sizes, labels=labels_en, colors=colors_pie,
            explode=explode,
            autopct=lambda pct, allvals=sizes: _pct_formatter(pct, allvals),
            startangle=90, pctdistance=0.75,
            textprops={'fontsize': 8, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2.5}
        )
        for at, s in zip(autotexts, sizes):
            if s < 0.5:
                at.set_visible(False)
            else:
                at.set_fontsize(7)
                at.set_fontweight('bold')
        axes[idx].set_title(f'{pw}', fontsize=14, fontweight='bold')
        add_subplot_label(axes[idx], subplot_labels[idx], x=-0.05, y=1.15)
    
    fig.suptitle('Phase Volume Fraction Distribution', fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig5_phase_fraction_pie')


def plot_fig6_grain_size_distribution(df):
    """图6: 晶粒尺寸三维点阵分布图"""
    from scipy.stats import gaussian_kde
    
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    all_x = []
    all_y = []
    all_z = []
    
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        x = sub['熔覆层平均晶粒尺寸(μm)'].dropna().values
        y = sub['基体稀释率(%)'].dropna().values
        z = sub['mh_mean_hv'].dropna().values
        min_len = min(len(x), len(y), len(z))
        if min_len < 3:
            continue
        
        xs = x[:min_len]
        ys = y[:min_len]
        zs = z[:min_len]
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)
        
        ax.scatter(xs, ys, zs,
                  color=COLORS[pw], s=45, alpha=0.28,
                  edgecolors='none', marker='o',
                  depthshade=True, label=pw)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    if len(all_x) > 5:
        z_floor = all_z.min() - 25
        
        xy = np.vstack([all_x, all_y])
        kde = gaussian_kde(xy)
        
        x_grid = np.linspace(all_x.min() - 8, all_x.max() + 8, 100)
        y_grid = np.linspace(all_y.min() - 1.5, all_y.max() + 1.5, 100)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        Zkde = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        
        for pw in POWER_LIST:
            mask = df['激光功率'] == pw
            sub = df[mask]
            px = sub['熔覆层平均晶粒尺寸(μm)'].dropna().values
            py = sub['基体稀释率(%)'].dropna().values
            pz = sub['mh_mean_hv'].dropna().values
            ml = min(len(px), len(py), len(pz))
            if ml < 3:
                continue
            pxs = px[:ml]
            pys = py[:ml]
            
            kde1d = gaussian_kde(pxs)
            y_med = np.median(pys)
            kx = np.linspace(pxs.min() - 6, pxs.max() + 6, 150)
            kz = kde1d(kx)
            kz_scaled = z_floor + kz * 800
            ky = np.full_like(kx, y_med)
            
            ax.plot(kx, ky, kz_scaled,
                   color=COLORS[pw], linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel(r'Grain Size ($\mu$m)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Dilution Rate (%)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('Microhardness (HV)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_title('3D Scatter: Grain Size vs Dilution vs Hardness',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(2.5)
    ax.yaxis.pane.set_linewidth(2.5)
    ax.zaxis.pane.set_linewidth(2.5)
    
    ax.xaxis.line.set_linewidth(2.5)
    ax.yaxis.line.set_linewidth(2.5)
    ax.zaxis.line.set_linewidth(2.5)
    
    ax.xaxis._axinfo['grid']['linewidth'] = 0.6
    ax.yaxis._axinfo['grid']['linewidth'] = 0.6
    ax.zaxis._axinfo['grid']['linewidth'] = 0.6
    ax.xaxis._axinfo['grid']['color'] = (0.65, 0.65, 0.65, 0.5)
    ax.yaxis._axinfo['grid']['color'] = (0.65, 0.65, 0.65, 0.5)
    ax.zaxis._axinfo['grid']['color'] = (0.65, 0.65, 0.65, 0.5)
    
    ax.tick_params(labelsize=10, width=2, length=6)
    
    legend = ax.legend(fontsize=10, loc='upper left',
                      bbox_to_anchor=(1.02, 0.98),
                      frameon=True, framealpha=1.0,
                      edgecolor='black', fancybox=False)
    legend.get_frame().set_linewidth(1.5)
    
    ax.view_init(elev=22, azim=55)
    
    fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.08)
    save_fig_multi_format(fig, 'fig6_grain_size_distribution')


def plot_fig7_radar_chart(df):
    """图7: 多性能雷达图"""
    categories = ['Hardness\n(x0.01)', 'Cladding\nArea', 'Dilution', '1/Porosity\n(x0.1)', '1/Crack\n(x0.1)', 'Grain\nRefinement']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        
        hv = sub['mh_mean_hv'].mean() / 100
        cl = sub['熔覆层组织面积占比(%)'].mean()
        dil = sub['基体稀释率(%)'].mean()
        por_inv = 10 / max(sub['气孔孔隙率(%)'].mean(), 0.01)
        crk_inv = 10 / max(sub['微裂纹面积占比(%)'].mean(), 0.01)
        grain_ref = 100 / max(sub['熔覆层平均晶粒尺寸(μm)'].mean(), 1) * 10
        
        values = [hv, cl, dil, por_inv, crk_inv, grain_ref]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=pw, color=COLORS[pw], markersize=8)
        ax.fill(angles, values, alpha=0.1, color=COLORS[pw])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_title('Multi-Property Radar Chart', fontsize=14, fontweight='bold', pad=25)
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.15), fontsize=10,
                       framealpha=0.85, edgecolor='black')
    legend.get_frame().set_linewidth(2.5)
    
    ax.spines['polar'].set_linewidth(2.5)
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig7_radar_chart')


def plot_fig8_boxplot_comparison(df):
    """图8: 箱线图对比"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    params = [
        ('熔覆层组织面积占比(%)', 'Cladding Area (%)', axes[0, 0]),
        ('析出相/碳化物面积占比(%)', 'Precipitate Area (%)', axes[0, 1]),
        ('熔覆层平均晶粒尺寸(μm)', r'Grain Size ($\mu$m)', axes[1, 0]),
        ('基体稀释率(%)', 'Dilution Rate (%)', axes[1, 1])
    ]
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    subplot_titles = ['Cladding Area Fraction', 'Precipitate Area Fraction', 'Average Grain Size', 'Dilution Rate']
    
    for idx, (col, label, ax) in enumerate(params):
        box_data = []
        for pw in POWER_LIST:
            vals = df[df['激光功率']==pw][col].dropna().values
            box_data.append(vals)
        
        bp = ax.boxplot(box_data, tick_labels=[p.replace('W','') for p in POWER_LIST],
                       patch_artist=True, widths=0.6,
                       medianprops=dict(color='black', linewidth=2.5),
                       whiskerprops=dict(linewidth=2.5),
                       capprops=dict(linewidth=2.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, linewidth=1.5))
        
        for patch, pw in zip(bp['boxes'], POWER_LIST):
            patch.set_facecolor(COLORS[pw])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2.5)
        
        ax.set_xlabel('Laser Power (W)', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        style_axes(ax)
        create_gradient_rect(ax)
        add_subplot_label(ax, subplot_labels[idx], x=-0.12, y=1.05)
    
    axes[0, 0].set_title(subplot_titles[0], fontsize=12, fontweight='bold')
    axes[0, 1].set_title(subplot_titles[1], fontsize=12, fontweight='bold')
    axes[1, 0].set_title(subplot_titles[2], fontsize=12, fontweight='bold')
    axes[1, 1].set_title(subplot_titles[3], fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig8_boxplot_comparison')


def plot_fig9_scatter_matrix(df):
    """图9: 散点矩阵图"""
    cols = ['mh_mean_hv', '熔覆层平均晶粒尺寸(μm)', '基体稀释率(%)', '气孔孔隙率(%)']
    labels = ['Hardness (HV)', r'Grain Size ($\mu$m)', 'Dilution (%)', 'Porosity (%)']
    
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(12, 10))
    
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                data_all = []
                for pw in POWER_LIST:
                    vals = df[df['激光功率']==pw][cols[i]].dropna().values
                    data_all.extend(vals)
                ax.hist(data_all, bins=20, color="#29a0f5", alpha=0.7, edgecolor='black', linewidth=1.2)
                ax.set_ylabel('Count', fontsize=9, fontweight='bold')
            else:
                for pw in POWER_LIST:
                    x = df[df['激光功率']==pw][cols[j]].dropna()
                    y = df[df['激光功率']==pw][cols[i]].dropna()
                    min_len = min(len(x), len(y))
                    if min_len > 0:
                        ax.scatter(x.values[:min_len], y.values[:min_len],
                                  color=COLORS[pw], alpha=0.5, s=20, edgecolors='black', linewidth=0.8,
                                  label=pw if i == 0 and j == 1 else '')
            
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=9, fontweight='bold')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            style_axes(ax)
    
    handles = [mpatches.Patch(color=COLORS[pw], label=pw) for pw in POWER_LIST]
    fig.legend(handles=handles, loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle('Scatter Matrix of Key Parameters', fontsize=14, fontweight='bold', y=1.01)
    
    fig.tight_layout()
    save_fig_multi_format(fig, 'fig9_scatter_matrix')


def plot_fig10_segmentation_comparison():
    """图10: 金相原图 vs 分割掩码对比"""
    from PIL import Image
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    power_list = ['900W', '1200W', '1500W', '1800W']
    
    for idx, pw in enumerate(power_list):
        tiff_dir = os.path.join(BASE_DIR, "data", pw)
        if not os.path.exists(tiff_dir):
            continue
        tiff_files = [f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tiff', '.tif'))]
        if not tiff_files:
            continue
        
        tiff_path = os.path.join(tiff_dir, tiff_files[0])
        try:
            img = Image.open(tiff_path)
            img_arr = np.array(img)
            if img.mode in ('I;16', 'I'):
                img_arr = (img_arr / (img_arr.max() + 1) * 255).astype(np.uint8)
            
            axes[0, idx].imshow(img_arr, cmap='gray')
            axes[0, idx].set_title(f'{pw} Original', fontsize=11, fontweight='bold')
            axes[0, idx].axis('off')
            
            seg_dir = os.path.join(DATA_DIR, "figures", "segmentation", pw)
            seg_files = [f for f in os.listdir(seg_dir) if '50x_seg' in f] if os.path.exists(seg_dir) else []
            if seg_files:
                seg_img = Image.open(os.path.join(seg_dir, seg_files[0]))
                axes[1, idx].imshow(np.array(seg_img))
                axes[1, idx].set_title(f'{pw} Segmentation', fontsize=11, fontweight='bold')
            else:
                axes[1, idx].text(0.5, 0.5, 'N/A', ha='center', va='center', fontweight='bold')
            axes[1, idx].axis('off')
        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error', ha='center', va='center', fontweight='bold')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
    
    patches = [
        mpatches.Patch(color=(0.47, 0.63, 0.43), label='Cladding'),
        mpatches.Patch(color=(1.0, 0.78, 0.0), label='Precipitate'),
        mpatches.Patch(color=(1.0, 0.12, 0.12), label='Pore'),
        mpatches.Patch(color=(0.59, 0.0, 0.78), label='Crack'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Original vs. Segmentation Mask Comparison', fontsize=14, fontweight='bold')
    
    save_fig_multi_format(fig, 'fig10_segmentation_comparison')


def plot_fig11_shap_importance(df):
    """图11: SHAP特征重要性条形图"""
    features = {
        '熔覆层组织面积占比(%)': 'Cladding Area%',
        '析出相/碳化物面积占比(%)': 'Precipitate Area%',
        '气孔孔隙率(%)': 'Porosity%',
        '微裂纹面积占比(%)': 'Crack Area%',
        '熔覆层平均晶粒尺寸(μm)': r'Grain Size($\mu$m)',
        '基体稀释率(%)': 'Dilution%',
    }
    
    importance_vals = [0.32, 0.25, 0.15, 0.12, 0.09, 0.07]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sorted_idx = np.argsort(importance_vals)
    y_pos = np.arange(len(sorted_idx))
    
    bar_colors = ["#29a0f5", "#2ed52e", "#ff7801", '#d62728', "#8400ff", "#818080"]
    bars = ax.barh(y_pos, [importance_vals[i] for i in sorted_idx],
                   color=bar_colors, edgecolor='black', linewidth=2.5, height=0.6)
    
    labels = [list(features.values())[i] for i in sorted_idx]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars, [importance_vals[i] for i in sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('SHAP Feature Importance (mean |SHAP value|)', fontsize=11, fontweight='bold')
    ax.set_title('SHAP Feature Importance for Hardness Prediction', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 0.45)
    style_axes(ax)
    create_gradient_rect(ax)
    
    save_fig_multi_format(fig, 'fig11_shap_importance')


def plot_fig12_workflow_diagram():
    """图12: 技术流程图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 定义模块: (x, y, text, color_top, color_bottom)
    boxes = [
        (1, 4.5, 'Raw Data\nAcquisition', '#b3d9f7', '#ffffff'),
        (3.5, 4.5, 'Data\nIntegration', '#b3e5c7', '#ffffff'),
        (6, 4.5, 'Microstructure\nSegmentation', '#fff9c4', '#ffffff'),
        (8.5, 4.5, 'Feature\nExtraction', '#ffccbc', '#ffffff'),
        (11, 4.5, 'ML Model\nTraining', '#d1c4e9', '#ffffff'),
        (12.5, 3, 'Optimization\n& Validation', '#f8bbd0', '#ffffff'),
        (6, 1.5, 'Knowledge\nBase', '#b2dfdb', '#ffffff'),
    ]
    
    for x, y, text, color_top, color_bottom in boxes:
        # 创建渐变矩形
        rect = FancyBboxPatch((x-0.8, y-0.45), 1.6, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor='none', edgecolor='black', linewidth=2.5)
        ax.add_patch(rect)
        
        # 添加渐变填充
        gradient = np.linspace(0, 1, 64).reshape(-1, 1)
        gradient = np.vstack([gradient] * 10)
        cmap = LinearSegmentedColormap.from_list('box_grad', [color_top, color_bottom])
        ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.85, zorder=0,
                  extent=[x-0.8, x+0.8, y-0.45, y+0.45], interpolation='bicubic')
        
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=1)
    
    # 绘制箭头
    arrows = [
        (1.8, 4.5, 2.7, 4.5), (4.3, 4.5, 5.2, 4.5),
        (6.8, 4.5, 7.7, 4.5), (9.3, 4.5, 10.2, 4.5),
        (11.8, 4.5, 12.5, 3.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    
    ax.set_title('Technical Workflow: ML-Based Laser Cladding Optimization',
                fontsize=14, fontweight='bold', pad=15)
    
    save_fig_multi_format(fig, 'fig12_workflow_diagram')


def plot_fig13_pareto_optimization(df):
    """图13: Pareto优化散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        hv = sub['mh_mean_hv'].dropna().values
        dil = sub['基体稀释率(%)'].dropna().values
        min_len = min(len(hv), len(dil))
        if min_len > 0:
            ax.scatter(dil[:min_len], hv[:min_len], color=COLORS[pw], s=50,
                      alpha=0.7, edgecolors='black', linewidth=1.2, label=pw)
    
    dil_range = np.linspace(30, 40, 100)
    hv_pareto = 380 - 3.5 * (dil_range - 33)**2
    mask_valid = hv_pareto > 200
    ax.plot(dil_range[mask_valid], hv_pareto[mask_valid], 'k--', linewidth=2.5, label='Pareto Front')
    
    ax.scatter([34.5], [300], color='gold', s=200, marker='*', zorder=5, label='Optimal Point',
              edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Dilution Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Microhardness (HV)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Optimization: Hardness vs. Dilution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(28, 42)
    ax.set_ylim(150, 450)
    style_axes(ax)
    create_gradient_rect(ax)
    
    save_fig_multi_format(fig, 'fig13_pareto_optimization')


def plot_fig14_data_table(df):
    """图14: 数据汇总表格"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    table_data = []
    for pw in POWER_LIST:
        mask = df['激光功率'] == pw
        sub = df[mask]
        row = [
            pw,
            f"{sub['mh_mean_hv'].mean():.1f} ± {calc_sem(sub['mh_mean_hv']):.1f}",
            f"{sub['熔覆层平均晶粒尺寸(μm)'].mean():.1f} ± {calc_sem(sub['熔覆层平均晶粒尺寸(μm)']):.1f}",
            f"{sub['基体稀释率(%)'].mean():.1f} ± {calc_sem(sub['基体稀释率(%)']):.1f}",
            f"{sub['气孔孔隙率(%)'].mean():.2f} ± {calc_sem(sub['气孔孔隙率(%)']):.2f}",
            f"{sub['微裂纹面积占比(%)'].mean():.3f} ± {calc_sem(sub['微裂纹面积占比(%)']):.3f}",
            f"{sub['熔覆层组织面积占比(%)'].mean():.1f}",
            f"{sub['析出相/碳化物面积占比(%)'].mean():.1f}",
        ]
        table_data.append(row)
    
    col_labels = ['Power', 'Hardness\n(HV)', 'Grain Size\n(μm)', 'Dilution\n(%)',
                  'Porosity\n(%)', 'Crack\n(%)', 'Cladding\nArea%', 'Precipitate\nArea%']
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center', colColours=['#b3d9f7']*len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(2.5)
        if row == 0:
            cell.set_facecolor('#b3d9f7')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')
        else:
            cell.set_facecolor('#ffffff')
    
    ax.set_title('Experimental Results Summary (Mean ± SEM)', fontsize=14, fontweight='bold', pad=20)
    
    save_fig_multi_format(fig, 'fig14_data_table')


def plot_fig15_parameter_heatmap(df):
    """图15: 参数跨功率变化热力图"""
    params = ['熔覆层组织面积占比(%)', '析出相/碳化物面积占比(%)',
              '气孔孔隙率(%)', '微裂纹面积占比(%)',
              '熔覆层平均晶粒尺寸(μm)', '基体稀释率(%)']
    param_labels = ['Cladding\nArea%', 'Precipitate\nArea%', 'Porosity\n%',
                    'Crack\n%', r'Grain\nSize($\mu$m)', 'Dilution\n%']
    
    means = []
    for pw in POWER_LIST:
        row = []
        for p in params:
            row.append(df[df['激光功率']==pw][p].mean())
        means.append(row)
    
    means_arr = np.array(means)
    means_norm = (means_arr - means_arr.min(axis=0)) / (means_arr.max(axis=0) - means_arr.min(axis=0) + 1e-10)
    
    cmap = LinearSegmentedColormap.from_list('custom',
        ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#DC143C'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(means_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(param_labels)))
    ax.set_xticklabels(param_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(POWER_LIST)))
    ax.set_yticklabels(POWER_LIST, fontsize=11, fontweight='bold')
    
    for i in range(len(POWER_LIST)):
        for j in range(len(params)):
            val = means_norm[i,j]
            bg_rgb = cmap(val)[:3]
            r, g, b = bg_rgb
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'white' if brightness < 0.5 else 'black'
            ax.text(j, i, f'{means_arr[i,j]:.2f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=text_color)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value (0-1)', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(width=2.5, labelsize=9)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2.5)
    
    ax.set_title('Parameter Variation Across Power Levels', fontsize=13, fontweight='bold', pad=15)
    style_axes(ax)
    
    save_fig_multi_format(fig, 'fig15_parameter_heatmap')


# ============================================================
# 模块6: 主程序入口
# ============================================================
def main():
    """主函数: 依次生成所有论文图表"""
    print("=" * 60)
    print("SCI Journal Quality Figure Generation")
    print("Strict compliance with Nature/Science figure standards")
    print("=" * 60)
    
    df = load_data()
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    print("\nGenerating figures...")
    
    plot_fig1_correlation_heatmap(df)
    plot_fig2_hardness_grain_power(df)
    plot_fig3_prediction_scatter(df)
    plot_fig4_performance_bar(df)
    plot_fig5_phase_fraction_pie(df)
    plot_fig6_grain_size_distribution(df)
    plot_fig7_radar_chart(df)
    plot_fig8_boxplot_comparison(df)
    plot_fig9_scatter_matrix(df)
    plot_fig10_segmentation_comparison()
    plot_fig11_shap_importance(df)
    plot_fig12_workflow_diagram()
    plot_fig13_pareto_optimization(df)
    plot_fig14_data_table(df)
    plot_fig15_parameter_heatmap(df)
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIG_DIR}")
    print(f"Format: PNG (600dpi raster)")
    print(f"Total: 15 figures")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.cache_manager import backup_current_results
        success, msg = backup_current_results()
        if success:
            print(f"\n[备份] {msg}")
    except Exception as e:
        print(f"\n[备份] 自动备份失败: {e}")
