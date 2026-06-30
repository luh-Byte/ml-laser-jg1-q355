"""
统一绘图风格配置
基于论文级图表规范（SCI期刊标准）

规范:
  - 字体: Times New Roman (全局), 加粗
  - 图框宽度: 2.5pt 统一
  - 渐变背景: 天蓝(#67B3E9)到白色, 从上到下
  - 子图角释: (a)(b)(c) 左上角与框上边界齐平
  - 图例: 右上角, 不出图框
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
    """设置统一的绘图风格: Times New Roman, 加粗, 2.5pt框线"""
    plt.rcParams.update({
        # ---- 字体: Times New Roman 全局, 加粗 ----
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimSun', 'DejaVu Serif', 'serif'],
        'font.sans-serif': ['Times New Roman', 'Arial', 'DejaVu Sans', 'sans-serif'],
        'mathtext.fontset': 'stix',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'font.size': 11,
        'font.weight': 'bold',

        # ---- 坐标轴: 2.5pt 框线 ----
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.linewidth': 2.5,
        'axes.unicode_minus': False,
        'axes.prop_cycle': plt.cycler('color', [
            "#29a0f5", "#ff7801", "#2ed52e", '#d62728',
            "#8400ff", "#ffff00", "#8c4677", "#818080"
        ]),

        # ---- 刻度: 朝内, 右上均有 ----
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

        # ---- 图例 ----
        'legend.fontsize': 10,
        'legend.framealpha': 0.85,
        'legend.edgecolor': 'black',
        'legend.borderpad': 0.8,
        'legend.handlelength': 2.0,
        'legend.loc': 'upper right',

        # ---- 网格 ----
        'grid.color': "#ffffff",
        'grid.linewidth': 0.6,
        'grid.alpha': 0.5,

        # ---- 输出 ----
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
def style_axes(ax, gradient=True):
    """
    统一设置坐标轴外观
    - 框线: 2.5pt 黑色
    - 刻度: 朝内, 两侧均有
    - 渐变背景: 天蓝到白 (由gradient参数控制)
    """
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=11, width=2, length=6, color='black')
    ax.tick_params(axis='both', which='minor', labelsize=9, width=1.5, length=3, color='black')
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.grid(False)
    ax.set_axisbelow(False)
    if gradient:
        create_gradient_rect(ax)


def create_gradient_rect(ax, color_top="#67B3E9", color_bottom='#FFFFFF', alpha=0.6, zorder=-2):
    """
    渐变背景: 天蓝(#67B3E9) → 白色, 从上到下
    自动在data坐标范围内绘制, 不干扰数据
    """
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient] * 100)

    cmap = LinearSegmentedColormap.from_list('custom_gradient', [color_top, color_bottom], N=256)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=alpha, zorder=zorder,
              extent=[0, 1, 0, 1], interpolation='bilinear', transform=ax.transAxes)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_subplot_label(ax, label, x=-0.08, y=1.05):
    """
    子图角释: (a)(b)(c) 放在左上角, 与框上边界齐平
    x=-0.08 在框线左侧外, y=1.05 紧贴上边框
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', color='black', ha='left', va='top')


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
