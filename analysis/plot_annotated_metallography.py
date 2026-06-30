"""
生成带组织标注的金相对比图
每行一个功率: 原图 | 分割图 | 组织标注
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, TextArea
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig

OM_DIR = os.path.join(BASE_DIR, "data", "OM")
SEG_DIR = os.path.join(BASE_DIR, "analysis_output", "figures", "segmentation")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

setup_plot_style()

# 组织信息定义
PHASE_INFO = {
    "900W": {
        "xrd_gamma": 21,
        "main_phase": "Martensite/Ferrite",
        "cladding": "alpha-Fe (BCC) + residual gamma-Fe",
        "precipitate": "Cr23C6 carbides (fine dispersion)",
        "substrate": "Ferrite + Pearlite (Q355)",
        "grain": "16.9 um",
        "hardness": "224 HV",
    },
    "1200W": {
        "xrd_gamma": 18,
        "main_phase": "Martensite/Ferrite",
        "cladding": "alpha-Fe (BCC) + trace gamma-Fe",
        "precipitate": "Cr23C6 carbides",
        "substrate": "Ferrite + Pearlite (Q355)",
        "grain": "18.6 um",
        "hardness": "243 HV",
    },
    "1500W": {
        "xrd_gamma": 79,
        "main_phase": "Austenite",
        "cladding": "gamma-Fe (FCC) dendrites",
        "precipitate": "Cr23C6 + Cr7C3 carbides",
        "substrate": "Ferrite + Pearlite (Q355)",
        "grain": "23.0 um",
        "hardness": "289 HV",
    },
    "1800W": {
        "xrd_gamma": 75,
        "main_phase": "Austenite",
        "cladding": "gamma-Fe (FCC) coarse dendrites",
        "precipitate": "Cr23C6 + Cr7C3 (abundant)",
        "substrate": "Ferrite + Pearlite (Q355)",
        "grain": "23.8 um",
        "hardness": "385 HV",
    },
}

# 分割颜色 (BGR for OpenCV)
SEG_COLORS_BGR = {
    "Substrate": (120, 120, 120),
    "Cladding": (110, 160, 100),
    "Carbide": (0, 200, 255),
    "Pore": (30, 30, 255),
    "Crack": (200, 0, 150),
}

# 分割颜色 (RGB for matplotlib)
SEG_COLORS_RGB = {
    "Substrate (Q355)": (120, 120, 120),
    "Cladding Layer": (100, 160, 110),
    "Carbide Precipitate": (255, 200, 0),
    "Porosity": (255, 30, 30),
    "Crack": (150, 0, 200),
}


def load_tiff(path):
    """Load TIFF image and convert to RGB uint8"""
    from PIL import Image
    pil = Image.open(path)
    arr = np.array(pil)
    if pil.mode == "I;16":
        arr = (arr / 256).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif pil.mode == "I":
        arr = (arr / (arr.max() + 1) * 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif pil.mode == "L":
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif pil.mode == "RGB":
        pass
    elif pil.mode == "RGBA":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    else:
        pil = pil.convert("RGB")
        arr = np.array(pil)
    return arr


def draw_phase_overlay(seg_img, phase_info):
    """在分割图上绘制组织类型标注"""
    h, w = seg_img.shape[:2]
    overlay = seg_img.copy()

    # 半透明覆盖层
    alpha = 0.35
    overlay_bg = np.zeros_like(seg_img, dtype=np.uint8)

    # 匹配各区域
    for name, rgb in SEG_COLORS_RGB.items():
        mask = np.all(np.abs(seg_img.astype(int) - np.array(rgb)) < 30, axis=2)
        if np.sum(mask) > 1000:
            overlay_bg[mask] = rgb

    result = cv2.addWeighted(seg_img, 1 - alpha, overlay_bg, alpha, 0)
    return result


def add_annotation(ax, x, y, text, color="white", fontsize=9):
    """在图上添加文字标注"""
    ax.text(
        x, y, text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor=color, linewidth=1.5),
        ha="center",
        va="center",
    )


def plot_annotated_grid():
    """生成4x3的金相组织标注对比图"""
    fig = plt.figure(figsize=(24, 28))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.25, wspace=0.15)

    powers = ["900W", "1200W", "1500W", "1800W"]

    for row, power in enumerate(powers):
        info = PHASE_INFO[power]

        # 选择1000x放大图像
        om_name = f"{power.replace('W', '')}-1000x(1).tif"
        seg_name = f"{power.replace('W', '')}-1000x(1)_seg.png"

        om_path = os.path.join(OM_DIR, f"laser{power}", om_name)
        seg_path = os.path.join(SEG_DIR, power, seg_name)

        if not os.path.exists(om_path) or not os.path.exists(seg_path):
            continue

        # 加载图像
        om_img = load_tiff(om_path)
        seg_img = cv2.imread(seg_path)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        # === 列1: 原图 ===
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(om_img)
        ax1.set_title(f"{power} - Original (1000x)", fontsize=14, fontweight="bold",
                      fontfamily="Times New Roman")
        ax1.axis("off")
        style_axes(ax1, gradient=False)
        for spine in ax1.spines.values():
            spine.set_linewidth(2.5)

        # === 列2: 分割图 + 组织标注 ===
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(seg_img)
        ax2.set_title(f"{power} - Segmentation", fontsize=14, fontweight="bold",
                      fontfamily="Times New Roman")
        ax2.axis("off")
        style_axes(ax2, gradient=False)
        for spine in ax2.spines.values():
            spine.set_linewidth(2.5)

        # 在分割图上标注各区域 — 手动偏移避免重叠
        h, w = seg_img.shape[:2]
        manual_offsets = {
            "900W":  {"Substrate (Q355)": (0, 0), "Cladding Layer": (0, 0)},
            "1200W": {"Substrate (Q355)": (0, 0), "Cladding Layer": (0, 0)},
            "1500W": {"Substrate (Q355)": (-300, -150), "Cladding Layer": (300, 150)},
            "1800W": {"Substrate (Q355)": (-300, -150), "Cladding Layer": (300, 150)},
        }
        offsets = manual_offsets.get(power, {})
        used_pos = []  # track placed annotations to avoid collision
        for name, rgb in SEG_COLORS_RGB.items():
            mask = np.all(np.abs(seg_img.astype(int) - np.array(rgb)) < 30, axis=2)
            if np.sum(mask) > 5000:
                ys, xs = np.where(mask)
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                # apply manual offset
                dx, dy = offsets.get(name, (0, 0))
                cx, cy = cx + dx, cy + dy
                # keep within image bounds
                cx = max(120, min(cx, w - 120))
                cy = max(40, min(cy, h - 40))
                # check distance to already placed labels
                too_close = any(abs(cx - px) < 250 and abs(cy - py) < 50 for px, py in used_pos)
                if too_close:
                    cy = min(cy + 80, h - 40)
                used_pos.append((cx, cy))
                short_name = name.split(" (")[0] if "(" in name else name
                add_annotation(ax2, cx, cy, short_name, color="white", fontsize=10)

        # === 列3: 组织信息面板 ===
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis("off")

        # 绘制信息卡片
        card_x, card_y = 0.05, 0.92
        line_h = 0.09

        # 标题
        ax3.text(card_x, card_y, f"{power} Microstructure",
                 transform=ax3.transAxes, fontsize=16, fontweight="bold",
                 color="#2c3e50", va="top")
        ax3.axhline(y=card_y - 0.06, xmin=0.05, xmax=0.95, color="#3498db", linewidth=2)

        # 信息条目
        entries = [
            ("Main Phase:", info["main_phase"], "#e74c3c" if "Austenite" in info["main_phase"] else "#2980b9"),
            (f"gamma-Fe (XRD):", f"{info['xrd_gamma']}%", "#e74c3c" if info["xrd_gamma"] > 50 else "#2980b9"),
            ("Cladding:", info["cladding"], "#2c3e50"),
            ("Precipitate:", info["precipitate"], "#2c3e50"),
            ("Substrate:", info["substrate"], "#2c3e50"),
            ("Grain Size:", info["grain"], "#27ae60"),
            ("Hardness:", info["hardness"], "#e67e22"),
        ]

        for i, (label, value, color) in enumerate(entries):
            y = card_y - 0.12 - i * line_h
            ax3.text(card_x + 0.02, y, label,
                     transform=ax3.transAxes, fontsize=12, fontweight="bold",
                     color="#34495e", va="top")
            ax3.text(card_x + 0.38, y, value,
                     transform=ax3.transAxes, fontsize=11,
                     color=color, va="top")

        # 功率标签
        ax3.add_patch(mpatches.FancyBboxPatch(
            (0.75, 0.88), 0.22, 0.08,
            boxstyle="round,pad=0.02",
            facecolor="#3498db", edgecolor="#2980b9", linewidth=2,
            transform=ax3.transAxes, clip_on=False
        ))
        ax3.text(0.865, 0.92, power,
                 transform=ax3.transAxes, fontsize=14, fontweight="bold",
                 color="white", ha="center", va="center")

    # 全局标题
    fig.suptitle(
        "JG-1 / Q355 Laser Cladding: Microstructure Identification\n"
        "(XRD Phase Analysis + Image Segmentation)",
        fontsize=18, fontweight="bold", y=0.98, fontfamily="Times New Roman"
    )

    # 图例
    legend_patches = [
        mpatches.Patch(color=np.array(c) / 255, label=n)
        for n, c in SEG_COLORS_RGB.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.01),
        prop={"family": "Times New Roman", "weight": "bold"},
    )

    save_fig(fig, "annotated_metallography", OUTPUT_DIR, dpi=300)
    plt.close()
    print(f"Figure saved: {os.path.join(OUTPUT_DIR, 'annotated_metallography.png')}")


def plot_phase_evolution():
    """生成组织随功率演变的趋势图 — 新样式: 渐变+角释+TNR"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    powers = [900, 1200, 1500, 1800]
    gamma_pct = [21, 18, 79, 75]
    hardness = [224, 243, 289, 385]
    grain_size = [16.9, 18.6, 23.0, 23.8]
    subplot_labels = ["(a)", "(b)", "(c)"]

    # 图1: gamma-Fe含量
    ax = axes[0]
    colors = ["#2980b9" if g < 50 else "#e74c3c" for g in gamma_pct]
    bars = ax.bar(range(len(powers)), gamma_pct, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(y=50, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(range(len(powers)))
    ax.set_xticklabels([f"{p}W" for p in powers], fontsize=12, fontfamily="Times New Roman")
    ax.set_ylabel("gamma-Fe Content (%)", fontsize=12, fontweight="bold", fontfamily="Times New Roman")
    ax.set_title("XRD Phase Ratio", fontsize=14, fontweight="bold", fontfamily="Times New Roman")
    for bar, val in zip(bars, gamma_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val}%", ha="center", fontsize=11, fontweight="bold", fontfamily="Times New Roman")
    ax.text(0.5, 52, "Austenite dominant", fontsize=9, color="#e74c3c", alpha=0.7, fontfamily="Times New Roman")
    ax.text(0.5, 46, "Martensite dominant", fontsize=9, color="#2980b9", alpha=0.7, fontfamily="Times New Roman")
    style_axes(ax)
    add_subplot_label(ax, subplot_labels[0])

    # 图2: 硬度
    ax = axes[1]
    ax.plot(powers, hardness, "o-", linewidth=2.5, markersize=12, color="#e67e22", markeredgecolor="black")
    for p, h in zip(powers, hardness):
        ax.annotate(f"{h} HV", (p, h), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=11, fontweight="bold",
                    color="#e67e22", fontfamily="Times New Roman")
    ax.set_xlabel("Laser Power (W)", fontsize=12, fontweight="bold", fontfamily="Times New Roman")
    ax.set_ylabel("Microhardness (HV)", fontsize=12, fontweight="bold", fontfamily="Times New Roman")
    ax.set_title("Hardness vs Power", fontsize=14, fontweight="bold", fontfamily="Times New Roman")
    ax.set_ylim(180, 420)
    style_axes(ax)
    add_subplot_label(ax, subplot_labels[1])

    # 图3: 晶粒尺寸
    ax = axes[2]
    ax.plot(powers, grain_size, "s-", linewidth=2.5, markersize=12, color="#27ae60", markeredgecolor="black")
    for p, g in zip(powers, grain_size):
        ax.annotate(f"{g} um", (p, g), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=11, fontweight="bold",
                    color="#27ae60", fontfamily="Times New Roman")
    ax.set_xlabel("Laser Power (W)", fontsize=12, fontweight="bold", fontfamily="Times New Roman")
    ax.set_ylabel("Grain Size (um)", fontsize=12, fontweight="bold", fontfamily="Times New Roman")
    ax.set_title("Grain Size vs Power", fontsize=14, fontweight="bold", fontfamily="Times New Roman")
    style_axes(ax)
    add_subplot_label(ax, subplot_labels[2])

    plt.tight_layout()
    save_fig(fig, "phase_evolution_trend", OUTPUT_DIR, dpi=300)
    plt.close()
    print(f"Figure saved: {os.path.join(OUTPUT_DIR, 'phase_evolution_trend.png')}")


if __name__ == "__main__":
    plot_annotated_grid()
    plot_phase_evolution()
    print("Done.")
