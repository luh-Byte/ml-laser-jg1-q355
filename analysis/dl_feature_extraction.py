"""
深度学习特征提取 — ResNet50 从金相图中提取深层特征
对比传统分割特征 vs 深度学习特征的区分能力

方法:
  1. 加载预训练ResNet50 (ImageNet), 去掉最后的全连接层
  2. 对每张金相图提取2048维特征向量
  3. PCA降到2-3维, 可视化各功率组的分布
  4. 与传统分割特征对比: 哪种特征能更好地区分900W/1200W/1500W/1800W
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig

OM_DIR = os.path.join(BASE_DIR, "data", "OM")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

setup_plot_style()

# 图像预处理 (ImageNet标准)
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

POWER_MAP = {
    "laser900W": 900,
    "laser1200W": 1200,
    "laser1500W": 1500,
    "laser1800W": 1800,
}


def load_resnet50():
    """加载预训练ResNet50, 返回特征提取器 (去掉最后的fc层)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    # 去掉最后的全连接层, 保留avgpool输出 (2048维)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    print(f"ResNet50 loaded (ImageNet pretrained), feature dim=2048")
    return feature_extractor


def extract_features(feature_extractor, image_dir):
    """从所有金相图中提取ResNet50特征"""
    all_features = []
    all_labels = []
    all_names = []
    all_paths = []

    for folder_name, power in POWER_MAP.items():
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        tiff_files = sorted([f for f in os.listdir(folder_path)
                             if f.lower().endswith((".tif", ".tiff"))])

        for fname in tiff_files:
            fpath = os.path.join(folder_path, fname)
            try:
                pil_img = Image.open(fpath)
                # 转RGB
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                tensor = PREPROCESS(pil_img).unsqueeze(0)

                with torch.no_grad():
                    feat = feature_extractor(tensor)
                    feat = feat.squeeze().numpy()  # (2048,)

                all_features.append(feat)
                all_labels.append(power)
                all_names.append(fname)
                all_paths.append(fpath)
                print(f"  {power}W / {fname}: OK (feat={feat.shape})")
            except Exception as e:
                print(f"  {power}W / {fname}: FAILED ({e})")

    X = np.array(all_features)
    y = np.array(all_labels)
    return X, y, all_names


def load_traditional_features():
    """加载传统分割特征 (面积占比 + 晶粒尺寸)"""
    csv = os.path.join(BASE_DIR, "analysis_output", "金相定量表征数据汇总.csv")
    if not os.path.exists(csv):
        return None, None, None

    df = pd.read_csv(csv, encoding="utf-8-sig")
    df["power_w"] = df.apply(lambda row: float(str(row["激光功率"]).replace("W", "")), axis=1)

    trad_cols = ["熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
                 "气孔缺陷面积占比(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)"]

    X_trad = df[trad_cols].values
    y_trad = df["power_w"].values.astype(int)
    names_trad = df["图像名称"].tolist()
    return X_trad, y_trad, names_trad, trad_cols


def analyze_discrimination(X, y, method_name, n_components=2):
    """分析特征的区分能力"""
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  Total explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    # Per-group center distance
    powers = sorted(set(y))
    centers = {}
    for pw in powers:
        mask = y == pw
        centers[pw] = X_pca[mask].mean(axis=0)

    # Inter-group distances
    print(f"\n  Inter-group distances (PCA space):")
    for i, p1 in enumerate(powers):
        for p2 in powers[i+1:]:
            dist = np.linalg.norm(centers[p1] - centers[p2])
            print(f"    {p1}W <-> {p2}W: {dist:.3f}")

    # Intra-group spread
    print(f"\n  Intra-group spread (avg distance to center):")
    for pw in powers:
        mask = y == pw
        group_center = X_pca[mask].mean(axis=0)
        spreads = [np.linalg.norm(X_pca[j] - group_center) for j in range(len(y)) if y[j] == pw]
        print(f"    {pw}W: {np.mean(spreads):.3f} +/- {np.std(spreads):.3f}")

    return X_pca, pca


def plot_comparison(dl_pca, trad_pca, y_dl, y_trad, pca_dl, pca_trad):
    """对比深度学习特征 vs 传统特征"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {900: "#2980b9", 1200: "#e67e22", 1500: "#27ae60", 1800: "#c0392b"}
    markers = {900: "o", 1200: "s", 1500: "^", 1800: "D"}

    # 图1: 传统分割特征
    ax = axes[0]
    for pw in sorted(set(y_trad)):
        mask = y_trad == pw
        ax.scatter(trad_pca[mask, 0], trad_pca[mask, 1],
                   c=colors[pw], marker=markers[pw], s=100,
                   edgecolors="black", linewidth=0.8, label=f"{pw}W", zorder=5)
    ax.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold")
    ax.set_title("Traditional Segmentation Features", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", frameon=True)
    var1 = pca_trad.explained_variance_ratio_[0] * 100
    var2 = pca_trad.explained_variance_ratio_[1] * 100
    ax.text(0.02, 0.02, f"PC1={var1:.1f}%  PC2={var2:.1f}%",
            transform=ax.transAxes, fontsize=10, va="bottom")
    style_axes(ax)
    add_subplot_label(ax, "(a)")

    # 图2: 深度学习特征
    ax = axes[1]
    for pw in sorted(set(y_dl)):
        mask = y_dl == pw
        ax.scatter(dl_pca[mask, 0], dl_pca[mask, 1],
                   c=colors[pw], marker=markers[pw], s=100,
                   edgecolors="black", linewidth=0.8, label=f"{pw}W", zorder=5)
    ax.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold")
    ax.set_title("ResNet50 Deep Features", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", frameon=True)
    var1 = pca_dl.explained_variance_ratio_[0] * 100
    var2 = pca_dl.explained_variance_ratio_[1] * 100
    ax.text(0.02, 0.02, f"PC1={var1:.1f}%  PC2={var2:.1f}%",
            transform=ax.transAxes, fontsize=10, va="bottom")
    style_axes(ax)
    add_subplot_label(ax, "(b)")

    fig.suptitle("Feature Comparison: Traditional Segmentation vs Deep Learning",
                 fontsize=16, fontweight="bold", fontfamily="Times New Roman")

    plt.tight_layout()
    save_fig(fig, "dl_vs_traditional_features", OUTPUT_DIR, dpi=300)
    plt.close()
    print(f"\nFigure saved: {os.path.join(OUTPUT_DIR, 'dl_vs_traditional_features.png')}")


def main():
    print("=" * 60)
    print("Deep Learning Feature Extraction for Metallographic Images")
    print("=" * 60)

    # 1. Load ResNet50
    feature_extractor = load_resnet50()

    # 2. Extract features
    print("\nExtracting features from metallographic images...")
    X_dl, y_dl, names_dl = extract_features(feature_extractor, OM_DIR)
    print(f"\nExtracted: {X_dl.shape[0]} images, {X_dl.shape[1]} features each")

    # 3. Analyze DL features
    dl_pca, pca_dl = analyze_discrimination(X_dl, y_dl, "ResNet50 Deep Features (2048D -> PCA)")

    # 4. Load traditional features
    trad_result = load_traditional_features()
    if trad_result is not None:
        X_trad, y_trad, names_trad, trad_cols = trad_result
        print(f"\nTraditional features: {X_trad.shape[0]} images, {X_trad.shape[1]} features")

        # Only use images that exist in both datasets
        trad_pca, pca_trad = analyze_discrimination(
            X_trad, y_trad,
            f"Traditional Segmentation Features ({X_trad.shape[1]}D -> PCA)"
        )

        # 5. Plot comparison
        # Align lengths (dl and trad may have different image order)
        min_len = min(len(y_dl), len(y_trad))
        plot_comparison(
            dl_pca[:min_len], trad_pca[:min_len],
            y_dl[:min_len], y_trad[:min_len],
            pca_dl, pca_trad
        )
    else:
        print("\nTraditional features not found, skipping comparison")

    print("\nDone.")


if __name__ == "__main__":
    main()
