"""
机器学习论文风格图表生成器
- 蓝白渐变背景
- 左上角A/B/C/D标签
- 2.5线宽加粗
- 符合SCI机器学习论文格式
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "utils"))
from plot_style import (
    setup_plot_style, style_axes, create_gradient_rect,
    add_subplot_label, save_fig, COLORS, CONFIG
)
setup_plot_style()


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "data_full.csv"), encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    df["heat_input"] = df["power_w"] / (df["扫描速度(mm/min)"] * df["送粉速率(g/min)"])
    df["hall_petch"] = 1.0 / np.sqrt(df["熔覆层平均晶粒尺寸(μm)"])
    return df


def add_paper_labels(axes, labels=None, x=-0.10, y=1.08):
    """添加论文风格的A/B/C/D标签在左上角"""
    if labels is None:
        labels = [f"({chr(65+i)})" for i in range(len(axes))]
    elif isinstance(labels, str):
        labels = [f"({labels}{chr(65+i)})" for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        add_subplot_label(ax, label, x=x, y=y)


def apply_gradient_to_axes(axes):
    """给所有子图添加蓝白渐变背景"""
    for ax in axes:
        create_gradient_rect(ax, color_top="#67B3E9", color_bottom="#FFFFFF", alpha=0.25, zorder=-2)
        style_axes(ax)


def fig1_model_comparison(df):
    """图1: 模型性能对比 (a) 实测vs预测散点图 (b) 指标柱状图"""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    features = [
        "power_w", "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)",
        "eis_Rct_ohm", "eis_Z_max_ohm", "xrd_main_peak_2theta",
        "wear_friction_steady", "hall_petch", "heat_input"
    ]
    valid_feats = [f for f in features if f in df.columns and df[f].notna().all()]
    X = df[valid_feats].values
    y = df["mh_mean_hv"].values
    groups = df["power_w"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_specs = {
        "GBR": (GradientBoostingRegressor, {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "min_samples_leaf": 2, "subsample": 0.8, "random_state": 42
        }),
        "RFR": (RandomForestRegressor, {
            "n_estimators": 100, "max_depth": 5, "min_samples_leaf": 2,
            "max_features": 0.7, "random_state": 42, "n_jobs": -1
        }),
        "GPR": (GaussianProcessRegressor, {
            "kernel": ConstantKernel(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(0.5, (1e-5, 1e2)),
            "n_restarts_optimizer": 5, "random_state": 42
        }),
    }

    results = {}
    loo = LeaveOneOut()
    for name, (model_cls, params) in model_specs.items():
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X_scaled):
            m = model_cls(**params)
            m.fit(X_scaled[train_idx], y[train_idx])
            y_pred[test_idx] = m.predict(X_scaled[test_idx])
        results[name] = {
            "y_pred": y_pred,
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
        }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    apply_gradient_to_axes(axes)

    # (a) 实测vs预测散点图
    ax = axes[0]
    markers = ["o", "s", "^"]
    colors_m = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, res), marker, color in zip(results.items(), markers, colors_m):
        ax.scatter(y, res["y_pred"], marker=marker, c=color, s=55,
                   edgecolors="black", linewidths=0.6, alpha=0.85,
                   label=f"{name} (R²={res['r2']:.4f})")
    lims = [y.min() - 15, y.max() + 15]
    ax.plot(lims, lims, "k--", linewidth=1.8, label="1:1 Line")
    ax.fill_between(lims, [l - 5 for l in lims], [l + 5 for l in lims],
                    alpha=0.12, color="gray", label="±5 HV")
    ax.set_xlabel("Measured Hardness (HV)", fontsize=12)
    ax.set_ylabel("Predicted Hardness (HV)", fontsize=12)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")

    # (b) 指标柱状图
    ax = axes[1]
    x = np.arange(3)
    width = 0.25
    model_names = list(results.keys())
    r2_vals = [results[m]["r2"] for m in model_names]
    rmse_vals = [results[m]["rmse"] for m in model_names]
    mae_vals = [results[m]["mae"] for m in model_names]

    ax2 = ax.twinx()
    style_axes(ax2)

    bars1 = ax.bar(x - width, r2_vals, width, label="R²", color="#1f77b4",
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    bars2 = ax2.bar(x, rmse_vals, width, label="RMSE (HV)", color="#ff7f0e",
                    edgecolor="black", linewidth=1.2, alpha=0.85)
    bars3 = ax2.bar(x + width, mae_vals, width, label="MAE (HV)", color="#2ca02c",
                    edgecolor="black", linewidth=1.2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("R² Score", fontsize=12, color="#1f77b4")
    ax2.set_ylabel("RMSE / MAE (HV)", fontsize=12, color="#ff7f0e")
    ax.set_ylim(0.95, 1.01)
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#ff7f0e")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left")

    add_paper_labels(axes, labels=["(A)", "(B)"])

    plt.suptitle("Fig. 1 Model Performance Comparison (LOO-CV, n=50)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = save_fig(fig, "Fig1_Model_Performance_Comparison", FIG_DIR, dpi=600)
    return results, path


def fig2_shap_importance(df):
    """图2: SHAP特征重要性 (a) 蜂群图 (b) 柱状图"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    try:
        import shap
    except ImportError:
        print("  [SKIP] shap未安装，跳过图2")
        return None

    features = [
        "power_w", "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)",
        "eis_Rct_ohm", "eis_Z_max_ohm", "xrd_main_peak_2theta",
        "wear_friction_steady", "hall_petch", "heat_input"
    ]
    valid_feats = [f for f in features if f in df.columns and df[f].notna().all()]
    X = df[valid_feats].values
    y = df["mh_mean_hv"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=2, subsample=0.8, random_state=42
    )
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # (a) SHAP蜂群图
    ax = axes[0]
    plt.sca(ax)
    shap.summary_plot(shap_values, X_scaled, feature_names=valid_feats,
                      show=False, max_display=12, plot_type="dot")
    style_axes(ax)
    ax.set_title("")
    ax.set_xlabel("SHAP Value (impact on hardness prediction)", fontsize=11)

    # (b) SHAP柱状图
    ax = axes[1]
    importance = np.abs(shap_values).mean(axis=0)
    feat_imp = sorted(zip(valid_feats, importance), key=lambda x: -x[1])
    top_n = 12
    names = [n for n, _ in feat_imp[:top_n]][::-1]
    vals = [v for _, v in feat_imp[:top_n]][::-1]

    colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(len(names)), vals, color=colors_bar,
                   edgecolor="black", linewidth=1.0, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n if len(n) < 22 else n[:19] + "..." for n in names], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    style_axes(ax)
    ax.grid(True, alpha=0.25, linestyle="--", axis="x")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9, fontweight="bold")

    add_paper_labels(axes, labels=["(A)", "(B)"], x=-0.10, y=1.06)

    plt.suptitle("Fig. 2 SHAP Feature Importance Analysis (GBR Model)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = save_fig(fig, "Fig2_SHAP_Feature_Importance", FIG_DIR, dpi=600)
    return feat_imp, path


def fig3_partial_dependence(df, feat_imp=None):
    """图3: 偏依赖图组合 (A)(B)(C)(D) 四个关键特征"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    features = [
        "power_w", "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)",
        "eis_Rct_ohm", "eis_Z_max_ohm", "xrd_main_peak_2theta",
        "wear_friction_steady", "hall_petch", "heat_input"
    ]
    valid_feats = [f for f in features if f in df.columns and df[f].notna().all()]
    X = df[valid_feats].values
    y = df["mh_mean_hv"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=2, subsample=0.8, random_state=42
    )
    model.fit(X_scaled, y)

    if feat_imp:
        top_feats = [n for n, _ in feat_imp[:4]]
    else:
        top_feats = ["power_w", "xrd_main_peak_2theta", "wear_friction_steady", "eis_Rct_ohm"]
        top_feats = [f for f in top_feats if f in valid_feats][:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes_flat = axes.flatten()
    apply_gradient_to_axes(axes_flat)

    for i, feat in enumerate(top_feats):
        ax = axes_flat[i]
        feat_idx = valid_feats.index(feat)
        x_scaled_min, x_scaled_max = X_scaled[:, feat_idx].min(), X_scaled[:, feat_idx].max()
        x_scaled_range = np.linspace(x_scaled_min, x_scaled_max, 200)

        X_pd = np.tile(X_scaled.mean(axis=0), (200, 1))
        X_pd[:, feat_idx] = x_scaled_range
        y_pd = model.predict(X_pd)

        x_real = df[feat].values
        x_vals = np.linspace(x_real.min(), x_real.max(), 200)

        ax.plot(x_vals, y_pd, "#1f77b4", linewidth=2.8, label="PDP")
        ax.fill_between(x_vals, y_pd - 2.5, y_pd + 2.5,
                        alpha=0.2, color="#1f77b4", label="±2.5 HV")

        scatter = ax.scatter(x_real, y, c=df["power_w"], cmap="viridis", s=45,
                             edgecolors="black", linewidths=0.6, alpha=0.8, zorder=5,
                             label="Samples")

        short = feat if len(feat) < 24 else feat[:21] + "..."
        ax.set_xlabel(short, fontsize=11)
        ax.set_ylabel("Predicted Hardness (HV)", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--")

    add_paper_labels(axes_flat, labels=["(A)", "(B)", "(C)", "(D)"], x=-0.08, y=1.05)

    cbar = fig.colorbar(scatter, ax=axes_flat, orientation="horizontal",
                        fraction=0.03, pad=0.08)
    cbar.set_label("Laser Power (W)", fontsize=11)

    plt.suptitle("Fig. 3 Partial Dependence Plots (Top 4 Features, GBR Model)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = save_fig(fig, "Fig3_Partial_Dependence_Plots", FIG_DIR, dpi=600)
    return top_feats, path


def fig4_noise_analysis(df):
    """图4: 噪声鲁棒性分析 (A) R²随噪声变化 (B) RMSE对比"""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, mean_squared_error

    features = [
        "power_w", "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)",
        "eis_Rct_ohm", "eis_Z_max_ohm", "xrd_main_peak_2theta",
        "wear_friction_steady", "hall_petch", "heat_input"
    ]
    valid_feats = [f for f in features if f in df.columns and df[f].notna().all()]
    X = df[valid_feats].values
    y_true = df["mh_mean_hv"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo = LeaveOneOut()

    model_specs = {
        "GBR": (GradientBoostingRegressor, {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "min_samples_leaf": 2, "subsample": 0.8, "random_state": 42
        }),
        "RFR": (RandomForestRegressor, {
            "n_estimators": 100, "max_depth": 5, "min_samples_leaf": 2,
            "max_features": 0.7, "random_state": 42, "n_jobs": -1
        }),
        "GPR": (GaussianProcessRegressor, {
            "kernel": ConstantKernel(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(0.5, (1e-5, 1e2)),
            "n_restarts_optimizer": 5, "random_state": 42
        }),
    }

    noise_levels = [0, 2, 4, 6, 8, 10, 15]
    noise_results = {name: {"r2": [], "rmse": []} for name in model_specs}

    for noise_pct in noise_levels:
        np.random.seed(42)
        if noise_pct == 0:
            y_noisy = y_true.copy()
        else:
            noise_std = y_true.std() * noise_pct / 100
            y_noisy = y_true + np.random.normal(0, noise_std, len(y_true))

        for name, (model_cls, params) in model_specs.items():
            y_pred = np.zeros_like(y_noisy)
            for train_idx, test_idx in loo.split(X_scaled):
                m = model_cls(**params)
                m.fit(X_scaled[train_idx], y_noisy[train_idx])
                y_pred[test_idx] = m.predict(X_scaled[test_idx])
            noise_results[name]["r2"].append(r2_score(y_noisy, y_pred))
            noise_results[name]["rmse"].append(np.sqrt(mean_squared_error(y_noisy, y_pred)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    apply_gradient_to_axes(axes)

    # (A) R² vs 噪声水平
    ax = axes[0]
    markers = ["o", "s", "^"]
    colors_m = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, res), marker, color in zip(noise_results.items(), markers, colors_m):
        ax.plot(noise_levels, res["r2"], marker=marker, c=color, linewidth=2.5,
                markersize=9, markeredgecolor="black", markeredgewidth=0.8,
                label=name, alpha=0.9)
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("R² Score (LOO-CV)", fontsize=12)
    ax.set_xticks(noise_levels)
    ax.set_ylim(0.9, 1.01)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--")

    for i, (name, res) in enumerate(noise_results.items()):
        for nx, ny in zip(noise_levels, res["r2"]):
            if ny > 0.92:
                ax.annotate(f"{ny:.3f}", (nx, ny), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=7.5, color=colors_m[i])

    # (B) RMSE对比
    ax = axes[1]
    x = np.arange(len(noise_levels))
    width = 0.25
    for i, (name, res) in enumerate(noise_results.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, res["rmse"], width, label=name,
                      color=colors_m[i], edgecolor="black", linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("RMSE (HV)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}%" for n in noise_levels])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    add_paper_labels(axes, labels=["(A)", "(B)"])

    plt.suptitle("Fig. 4 Model Robustness Under Different Noise Levels",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = save_fig(fig, "Fig4_Noise_Robustness_Analysis", FIG_DIR, dpi=600)
    return noise_results, path


def fig5_hall_petch(df):
    """图5: Hall-Petch关系验证 (A) H vs d (B) H vs d^-1/2"""
    d = df["熔覆层平均晶粒尺寸(μm)"].values
    hv = df["mh_mean_hv"].values
    groups = df["power_w"].values

    d_inv_sqrt = 1.0 / np.sqrt(d)
    coeffs = np.polyfit(d_inv_sqrt, hv, 1)
    hv_fit = coeffs[0] * d_inv_sqrt + coeffs[1]
    from sklearn.metrics import r2_score
    r2_hp = r2_score(hv, hv_fit)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    apply_gradient_to_axes(axes)

    # (A) H vs d
    ax = axes[0]
    scatter = ax.scatter(d, hv, c=groups, cmap="viridis", s=65,
                         edgecolors="black", linewidths=0.8, alpha=0.85, zorder=5)
    d_smooth = np.linspace(d.min(), d.max(), 200)
    hv_smooth = coeffs[0] / np.sqrt(d_smooth) + coeffs[1]
    ax.plot(d_smooth, hv_smooth, "#d62728", linewidth=2.8, linestyle="--",
            label=f"Hall-Petch fit (R²={r2_hp:.4f})")
    ax.set_xlabel("Grain Size d (μm)", fontsize=12)
    ax.set_ylabel("Hardness (HV)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # (B) H vs d^-1/2
    ax = axes[1]
    scatter = ax.scatter(d_inv_sqrt, hv, c=groups, cmap="viridis", s=65,
                         edgecolors="black", linewidths=0.8, alpha=0.85, zorder=5)
    x_smooth = np.linspace(d_inv_sqrt.min(), d_inv_sqrt.max(), 200)
    ax.plot(x_smooth, coeffs[0] * x_smooth + coeffs[1], "#d62728", linewidth=2.8,
            linestyle="--",
            label=f"H = {coeffs[1]:.1f} + {coeffs[0]:.1f}·d⁻¹ᐟ²\n(R²={r2_hp:.4f})")
    ax.set_xlabel("d⁻¹ᐟ² (μm⁻¹ᐟ²)", fontsize=12)
    ax.set_ylabel("Hardness (HV)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal",
                        fraction=0.03, pad=0.08)
    cbar.set_label("Laser Power (W)", fontsize=11)

    add_paper_labels(axes, labels=["(A)", "(B)"])

    plt.suptitle("Fig. 5 Hall-Petch Relationship Analysis",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = save_fig(fig, "Fig5_Hall_Petch_Relationship", FIG_DIR, dpi=600)
    return coeffs, r2_hp, path


def fig6_correlation_heatmap(df):
    """图6: Pearson相关系数热力图"""
    import seaborn as sns

    feats_corr = [
        "power_w", "mh_mean_hv",
        "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)", "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)",
        "eis_Rct_ohm", "eis_Z_max_ohm", "xrd_main_peak_2theta",
        "wear_friction_steady", "heat_input", "hall_petch"
    ]
    valid = [f for f in feats_corr if f in df.columns and df[f].notna().all()]
    corr_df = df[valid].copy()
    short_names = {
        "power_w": "Power",
        "mh_mean_hv": "Hardness",
        "熔覆层组织面积占比(%)": "Clad area",
        "析出相/碳化物面积占比(%)": "Precipitate",
        "气孔孔隙率(%)": "Porosity",
        "熔覆层平均晶粒尺寸(μm)": "Grain size",
        "基体稀释率(%)": "Dilution",
        "eis_Rct_ohm": "Rct",
        "eis_Z_max_ohm": "Zmax",
        "xrd_main_peak_2theta": "2θ peak",
        "wear_friction_steady": "Friction",
        "heat_input": "Heat input",
        "hall_petch": "d⁻¹ᐟ²",
    }
    corr_df.columns = [short_names.get(c, c) for c in corr_df.columns]
    corr_matrix = corr_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, square=True, linewidths=1.5, linecolor="white",
                annot_kws={"size": 9, "weight": "bold"}, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    style_axes(ax)
    ax.set_title("")
    ax.tick_params(labelsize=10)

    add_paper_labels([ax], labels=["(A)"], x=-0.06, y=1.02)

    plt.suptitle("Fig. 6 Pearson Correlation Heatmap",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = save_fig(fig, "Fig6_Correlation_Heatmap", FIG_DIR, dpi=600)
    return corr_matrix, path


def main():
    print("=" * 70)
    print("生成机器学习论文风格图表")
    print("=" * 70)
    print(f"\n输出目录: {FIG_DIR}")

    df = load_data()
    print(f"\n数据加载完成: {len(df)} 样本")

    print("\n[1/6] 生成图1: 模型性能对比...")
    results, p1 = fig1_model_comparison(df)

    print("\n[2/6] 生成图2: SHAP特征重要性...")
    feat_imp, p2 = fig2_shap_importance(df)

    print("\n[3/6] 生成图3: 偏依赖图组合...")
    top_feats, p3 = fig3_partial_dependence(df, feat_imp)

    print("\n[4/6] 生成图4: 噪声鲁棒性分析...")
    noise_res, p4 = fig4_noise_analysis(df)

    print("\n[5/6] 生成图5: Hall-Petch关系验证...")
    coeffs, r2_hp, p5 = fig5_hall_petch(df)

    print("\n[6/6] 生成图6: Pearson相关系数热力图...")
    corr_mat, p6 = fig6_correlation_heatmap(df)

    print("\n" + "=" * 70)
    print("✅ 所有论文图表生成完成！")
    print("=" * 70)
    print(f"\n共生成 6 张论文级图表:")
    for i, p in enumerate([p1, p2, p3, p4, p5, p6], 1):
        fname = os.path.basename(p) if p else "(skipped)"
        print(f"  Fig.{i}: {fname}")
    print(f"\n保存位置: {FIG_DIR}")
    print("\n图表特点:")
    print("  ✅ 左上角A/B/C/D标签")
    print("  ✅ 蓝白渐变背景")
    print("  ✅ 2.5线宽加粗坐标轴")
    print("  ✅ 600 DPI高分辨率")
    print("  ✅ SCI机器学习论文格式")


if __name__ == "__main__":
    main()
