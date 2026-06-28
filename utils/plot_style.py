"""
统一绘图风格配置
基于论文级图表规范（SCI期刊标准）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ============================================================
# 全局rcParams配置
# ============================================================
def setup_plot_style():
    """设置统一的绘图风格"""
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
# 配色方案
# ============================================================
COLORS = {
    '900W': "#2381c4",
    '1200W': '#ff7f0e',
    '1500W': "#2baf2b",
    '1800W': '#d62728',
}
POWER_LIST = ['900W', '1200W', '1500W', '1800W']
POWER_NUM = [900, 1200, 1500, 1800]

CONFIG = {
    'fig_size_single': (8, 6),
    'fig_size_double': (14, 6),
    'fig_size_triple': (20, 6),
    'fig_size_quad': (16, 12),
    
    'color_blue': "#1382d2",
    'color_orange': '#ff7f0e',
    'color_green': '#2ca02c',
    'color_red': '#d62728',
    
    'gradient_top': "#00B5FD",
    'gradient_bottom': '#FFFFFF',
    
    'line_width': 2.5,
    'marker_size': 6,
    'spine_width': 2.5,
    'tick_major_size': 6,
    'tick_label_size': 11,
    
    'dpi': 600,
}


# ============================================================
# 工具函数
# ============================================================
def style_axes(ax):
    """统一设置坐标轴外观"""
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=11, width=2, length=6, color='black')
    ax.tick_params(axis='both', which='minor', labelsize=9, width=1.5, length=3, color='black')
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.grid(False)
    ax.set_axisbelow(False)


def create_gradient_rect(ax, color_top="#67B3E9", color_bottom='#FFFFFF', alpha=0.6, zorder=-2):
    """通用渐变背景函数"""
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient] * 100)
    
    cmap = LinearSegmentedColormap.from_list('custom_gradient', [color_top, color_bottom], N=256)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=zorder,
              extent=[0, 1, 0, 1], interpolation='bilinear', transform=ax.transAxes)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_subplot_label(ax, label, x=-0.12, y=1.05):
    """在子图左上角添加编号标签 (a)(b)(c)"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='black', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=1.5))


def calc_sem(series, error_type='sem'):
    """计算标准误或标准差"""
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


def save_fig(fig, name, fig_dir, dpi=600):
    """统一保存图表"""
    import os
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] {name}.png")
    return path
