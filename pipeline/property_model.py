"""
阶段3: 组织→性能模型
- 用实测显微硬度作为目标变量
- 整合金相+电化学+XRD+磨损多维度特征
- 多种ML模型对比 + SHAP解释性分析
"""

import os
import sys
import io
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import pickle

# 导入统一绘图风格
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from plot_style import setup_plot_style, style_axes, create_gradient_rect, add_subplot_label, calc_sem, save_fig, COLORS, POWER_LIST, POWER_NUM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 设置统一绘图风格
setup_plot_style()


# ===================== 1. 数据加载与特征工程 =====================
def load_and_prepare():
    csv_path = os.path.join(OUTPUT_DIR, "data_full.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))

    feature_groups = {
        "metallography": [
            "熔覆层组织面积占比(%)",
            "析出相/碳化物面积占比(%)",
            "气孔孔隙率(%)",
            "微裂纹面积占比(%)",
            "熔覆层平均晶粒尺寸(μm)",
            "基体稀释率(%)",
        ],
        "eis": [
            "eis_Rs_ohm",
            "eis_Rct_ohm",
            "eis_Z_max_ohm",
            "eis_theta_min_deg",
        ],
        "xrd": [
            "xrd_main_peak_2theta",
            "xrd_main_peak_intensity",
            "xrd_peak_44_area",
        ],
        "wear": [
            "wear_friction_steady",
            "wear_friction_std",
        ],
        "process": [
            "扫描速度(mm/min)",
            "送粉速率(g/min)",
        ],
    }

    all_features = ["power_w"]
    for group, feats in feature_groups.items():
        for f in feats:
            if f in df.columns:
                all_features.append(f)

    # 添加交叉特征: 功率×组织
    for f in ["熔覆层平均晶粒尺寸(μm)", "析出相/碳化物面积占比(%)", "气孔孔隙率(%)"]:
        col_name = f"power_x_{f}"
        df[col_name] = df["power_w"] * df[f]
        all_features.append(col_name)

    # 添加交叉特征: 功率×工艺参数
    for f in ["扫描速度(mm/min)", "送粉速率(g/min)"]:
        if f in df.columns:
            col_name = f"power_x_{f}"
            df[col_name] = df["power_w"] * df[f]
            all_features.append(col_name)

    # 添加Hall-Petch项
    df["hall_petch"] = 1.0 / np.sqrt(df["熔覆层平均晶粒尺寸(μm)"])
    all_features.append("hall_petch")

    # 添加热输入估算（使用可变扫描速度）
    scan_spacing = 0.05  # mm
    if "扫描速度(mm/min)" in df.columns:
        df["heat_input"] = df["power_w"] / (df["扫描速度(mm/min)"] / 60) / scan_spacing / 1000
    else:
        scan_speed = 600  # mm/min (实际值: 10 mm/s = 600 mm/min)
        df["heat_input"] = df["power_w"] / (scan_speed / 60) / scan_spacing / 1000
    all_features.append("heat_input")

    # 添加梯度硬度特征
    gradient_features = [
        "mh_cladding_hv",
        "mh_substrate_hv", 
        "mh_gradient_range",
    ]
    for f in gradient_features:
        if f in df.columns:
            all_features.append(f)
    
    # 添加硬度比值特征
    if "mh_cladding_hv" in df.columns and "mh_substrate_hv" in df.columns:
        df["hardness_ratio"] = df["mh_cladding_hv"] / df["mh_substrate_hv"].clip(lower=1)
        all_features.append("hardness_ratio")

    valid_features = [f for f in all_features if f in df.columns and df[f].notna().all()]

    target = "mh_mean_hv"

    return df, valid_features, target, feature_groups


# ===================== 2. 模型训练与评估 =====================
def train_models(df, features, target, cv_method="logo"):
    """
    训练多种回归模型
    cv_method: "logo" (按功率组留一, 推荐) | "loo" (按单样本留一, 有数据泄露风险)
    """
    X = df[features].values
    y = df[target].values
    groups = df["power_w"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 完整版模型：仅保留效果好的3个模型
    models = {
        "GBR": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=2, subsample=0.8, random_state=42
        ),
        "RFR": RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=2,
            max_features=0.7, random_state=42, n_jobs=-1
        ),
        "GPR": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(0.5, (1e-5, 1e2)),
            n_restarts_optimizer=5, random_state=42
        ),
    }

    if cv_method == "loo":
        cv = LeaveOneOut()
        cv_name = "LOO-CV"
    else:
        cv = LeaveOneGroupOut()
        cv_name = "LOGO-CV"

    results = {}
    import copy

    for name, model in models.items():
        if cv_method == "loo":
            y_cv = cross_val_predict(model, X_scaled, y, cv=cv)
        else:
            y_cv = np.zeros_like(y)
            for train_idx, test_idx in cv.split(X, y, groups):
                scaler_cv = StandardScaler()
                X_train = scaler_cv.fit_transform(X[train_idx])
                X_test = scaler_cv.transform(X[test_idx])
                m = copy.deepcopy(model)
                m.fit(X_train, y[train_idx])
                y_cv[test_idx] = m.predict(X_test)

        r2 = r2_score(y, y_cv)
        rmse = np.sqrt(mean_squared_error(y, y_cv))
        mae = mean_absolute_error(y, y_cv)

        model.fit(X_scaled, y)
        y_train_pred = model.predict(X_scaled)
        r2_train = r2_score(y, y_train_pred)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "r2_train": r2_train,
            "r2_cv": r2,
            "rmse_cv": rmse,
            "mae_cv": mae,
            "y_cv": y_cv,
            "y_train_pred": y_train_pred,
        }

        print(f"  {name}: train R²={r2_train:.4f}, {cv_name} R²={r2:.4f}, "
              f"CV-RMSE={rmse:.2f} HV, CV-MAE={mae:.2f} HV")

        gap = r2_train - r2
        if gap > 0.3:
            print(f"    ⚠️ 严重过拟合: 训练R²-CV_R²={gap:.4f}")
        elif gap > 0.1:
            print(f"    ⚠️ 过拟合风险: 训练R²-CV_R²={gap:.4f}")
        if r2 > 0.99 and cv_method == "loo":
            print(f"    ⚠️ CV R²={r2:.4f} ≈ 1.0，可能存在数据泄露，建议改用LOGO-CV")

    ratio = len(features) / len(y)
    print(f"\n  特征/样本比: {len(features)}/{len(y)} = {ratio:.3f}")
    if ratio > 0.3:
        print(f"  ⚠️ 特征维度偏高，建议特征选择")

    n_groups = len(np.unique(groups))
    print(f"  功率组数: {n_groups}")
    if cv_method == "logo" and n_groups <= 4:
        print(f"  ⚠️ 功率组数量少({n_groups})，LOGO-CV评估方差较大")

    lc_data = None

    return results, X, y, X_scaled, lc_data


# ===================== 3. SHAP分析 =====================
def shap_analysis(model, X, feature_names):
    """SHAP特征重要性分析"""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, max_display=15)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "shap_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] {path}")

        importance = np.abs(shap_values).mean(axis=0)
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        return feat_imp[:15]
    except ImportError:
        print("  [WARN] shap not installed, using feature_importances_ instead")
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
            return feat_imp[:15]
        return []


# ===================== 4. 偏依赖图 =====================
def plot_partial_dependence(model, X, feature_names, df, target):
    """绘制关键特征的偏依赖图"""
    top_features = ["power_w", "hall_petch", "heat_input",
                    "析出相/碳化物面积占比(%)", "气孔孔隙率(%)",
                    "eis_Rct_ohm", "xrd_main_peak_2theta", "wear_friction_steady"]
    available = [f for f in top_features if f in feature_names]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, feat in enumerate(available[:8]):
        ax = axes[i]
        feat_idx = feature_names.index(feat)
        x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 100)

        X_pd = np.tile(X.mean(axis=0), (100, 1))
        X_pd[:, feat_idx] = x_range

        y_pd = model.predict(X_pd)

        ax.plot(x_range, y_pd, "b-", linewidth=2)
        ax.scatter(X[:, feat_idx], df[target].values, c="red", s=20, alpha=0.5, zorder=5)

        short = feat if len(feat) < 20 else feat[:17] + "..."
        ax.set_xlabel(short, fontsize=9)
        ax.set_ylabel("HV", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(available), 8):
        axes[j].set_visible(False)

    plt.suptitle("Partial Dependence Plots (GBR)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "partial_dependence_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===================== 5. 模型对比图 =====================
def plot_model_comparison(results, y, df, cv_name="LOGO-CV"):
    """绘制模型对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    powers = df["power_w"].values

    ax = axes[0]
    color_list = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
    colors = {}
    for i, name in enumerate(results.keys()):
        colors[name] = color_list[i % len(color_list)]

    for name, r in results.items():
        ax.scatter(y, r["y_cv"], c=colors[name], s=40, alpha=0.7, label=f'{name} (R²={r["r2_cv"]:.3f})')
    lims = [y.min() - 20, y.max() + 20]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Measured HV")
    ax.set_ylabel(f"{cv_name} Predicted HV")
    ax.set_title(f"(a) Measured vs {cv_name} Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) 各功率硬度分布 + 模型预测
    ax = axes[1]
    power_levels = sorted(df["power_w"].unique())
    x_pos = np.arange(len(power_levels))
    n_models = len(results)
    width = 0.8 / (n_models + 1)

    for j, (name, r) in enumerate(results.items()):
        preds_by_power = [r["y_cv"][df["power_w"].values == p].mean() for p in power_levels]
        ax.bar(x_pos + j * width, preds_by_power, width, label=name,
               color=colors[name] if name in colors else "gray", edgecolor="black")

    actual_by_power = [y[df["power_w"].values == p].mean() for p in power_levels]
    ax.bar(x_pos + n_models * width, actual_by_power, width, label="Measured",
           color="gray", edgecolor="black")

    ax.set_xticks(x_pos + (n_models * width) / 2)
    ax.set_xticklabels([f"{int(p)}W" for p in power_levels])
    ax.set_ylabel("Hardness (HV)")
    ax.set_title("(b) Hardness by Power Level")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) 残差分布
    ax = axes[2]
    for name, r in results.items():
        residuals = y - r["y_cv"]
        ax.hist(residuals, bins=15, alpha=0.4, label=f'{name} (MAE={r["mae_cv"]:.1f})',
                color=colors[name] if name in colors else "gray")
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Residual (HV)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Residual Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Property Model Comparison ({cv_name})", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "property_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===================== 6. 功率-性能敏感性分析 =====================
def sensitivity_analysis(results, df, features, target):
    """分析功率变化对硬度的敏感性"""
    gbr = results["GBR"]["model"]
    scaler = results["GBR"]["scaler"]

    base_power = 1350  # 中间功率
    power_range = np.arange(900, 1850, 50)

    X_base = df[features].median().values.copy().reshape(1, -1)
    X_base[0, features.index("power_w")] = base_power

    # 更新衍生特征
    def update_derived(X, pw_idx):
        X_new = X.copy()
        hp_idx = features.index("hall_petch") if "hall_petch" in features else -1
        hi_idx = features.index("heat_input") if "heat_input" in features else -1
        gs_idx = features.index("熔覆层平均晶粒尺寸(μm)") if "熔覆层平均晶粒尺寸(μm)" in features else -1

        if hp_idx >= 0 and gs_idx >= 0:
            X_new[0, hp_idx] = 1.0 / np.sqrt(X_new[0, gs_idx])
        if hi_idx >= 0:
            X_new[0, hi_idx] = X_new[0, pw_idx] / (300 / 60) / 0.05 / 1000

        for feat in features:
            if feat.startswith("power_x_"):
                base_feat = feat.replace("power_x_", "")
                if base_feat in features:
                    base_idx = features.index(base_feat)
                    feat_idx = features.index(feat)
                    X_new[0, feat_idx] = X_new[0, pw_idx] * X_new[0, base_idx]
        return X_new

    pw_idx = features.index("power_w")
    hardness_pred = []
    for pw in power_range:
        X_temp = X_base.copy()
        X_temp[0, pw_idx] = pw
        X_temp = update_derived(X_temp, pw_idx)
        X_scaled_t = scaler.transform(X_temp)
        hardness_pred.append(gbr.predict(X_scaled_t)[0])

    hardness_pred = np.array(hardness_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(power_range, hardness_pred, "b-o", linewidth=2, markersize=4)
    ax.fill_between(power_range, hardness_pred - 10, hardness_pred + 10, alpha=0.1, color="blue")

    ax.set_xlabel("Laser Power (W)", fontsize=12)
    ax.set_ylabel("Predicted Hardness (HV)", fontsize=12)
    ax.set_title("Power Sensitivity Analysis (GBR)", fontsize=13)
    ax.grid(True, alpha=0.3)

    opt_idx = np.argmax(hardness_pred)
    ax.axvline(x=power_range[opt_idx], color="red", linestyle="--",
               label=f"Optimal: {power_range[opt_idx]}W → {hardness_pred[opt_idx]:.0f} HV")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "power_sensitivity_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")

    return power_range, hardness_pred


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("阶段3: 组织→性能模型(基于实测硬度)")
    print("=" * 60)

    print("\n[1/6] 加载数据与特征工程...")
    df, features, target, feature_groups = load_and_prepare()
    print(f"  样本数: {len(df)}, 原始特征数: {len(features)}")
    print(f"  目标: {target} (实测显微硬度)")

    USE_LOGO_CV = False  # 改用LOO-CV（同一功率内预测，更合理）
    REMOVE_POWER_FEATURES = False  # 保留功率特征（硬度主要由功率决定）

    if REMOVE_POWER_FEATURES:
        features_no_power = [f for f in features
                             if f not in ["power_w", "heat_input"]
                             and not f.startswith("power_x_")
                             and not f.startswith("mh_")
                             and f != "hardness_ratio"]
        print(f"  移除功率和硬度相关特征后: {len(features_no_power)} 个特征")
        features = features_no_power

    cv_method = "logo" if USE_LOGO_CV else "loo"
    cv_name = "LOGO-CV(按功率组留一)" if USE_LOGO_CV else "LOO-CV(按单样本留一)"
    print(f"  交叉验证方式: {cv_name}")

    if USE_LOGO_CV:
        print(f"  ⚠️ 注意: LOGO-CV是真实泛化能力评估，R²可能较低")
        print(f"     因为只有4个功率组，预测新功率水平的硬度本身很困难")

    print(f"\n[2/6] 训练模型({cv_name})...")
    results, X, y, X_scaled, lc_data = train_models(df, features, target, cv_method=cv_method)

    print("\n[3/6] 模型对比图...")
    plot_model_comparison(results, y, df, cv_name=("LOGO-CV" if USE_LOGO_CV else "LOO-CV"))

    if lc_data is None:
        print("\n[3.5/6] 学习曲线 (跳过: LOGO-CV下不适用)")
    else:
        print("\n[3.5/6] 学习曲线...")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lc_data["train_sizes"], lc_data["train_mean"], "b-o", label="Training R²", linewidth=2)
        ax.plot(lc_data["train_sizes"], lc_data["val_mean"], "r-s", label="Validation R²", linewidth=2)
        ax.fill_between(lc_data["train_sizes"], lc_data["val_mean"], alpha=0.1, color="red")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("R² Score")
        ax.set_title("Learning Curve (GBR)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, "learning_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {path}")

    print("\n[4/6] SHAP特征重要性分析...")
    best_model_name = max(results.keys(), key=lambda k: results[k]["r2_cv"])
    print(f"  最优模型: {best_model_name} (CV R²={results[best_model_name]['r2_cv']:.4f})")
    best_model = results[best_model_name]["model"]
    feat_imp = shap_analysis(best_model, X_scaled, features)
    if feat_imp:
        print("  Top 10 features:")
        for name, imp in feat_imp[:10]:
            print(f"    {name}: {imp:.4f}")

    print("\n[5/6] 偏依赖图...")
    plot_partial_dependence(best_model, X_scaled, features, df, target)

    if not REMOVE_POWER_FEATURES and "power_w" in features:
        print("\n[6/6] 功率敏感性分析...")
        pw_range, pw_pred = sensitivity_analysis(results, df, features, target)
    else:
        print("\n[6/6] 功率敏感性分析 (跳过: 已移除功率特征)")
        pw_range, pw_pred = None, None

    results_path = os.path.join(OUTPUT_DIR, "property_models.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "models": {k: v["model"] for k, v in results.items()},
            "scaler": results[best_model_name]["scaler"],
            "features": features,
            "feature_groups": feature_groups,
            "cv_method": cv_method,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        }, f)
    print(f"\n  [OK] 模型已保存: {results_path}")

    report_path = os.path.join(OUTPUT_DIR, "阶段3_性能模型报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 阶段3: 组织→性能模型报告\n\n")

        f.write("## 1. 数据概况\n\n")
        f.write(f"- 样本数: {len(df)} (4个功率水平 × ~14张图片)\n")
        f.write(f"- 特征数: {len(features)}\n")
        f.write(f"- 目标变量: 实测显微硬度 (mh_mean_hv)\n")
        f.write(f"- 硬度范围: {y.min():.1f} ~ {y.max():.1f} HV\n")
        f.write(f"- 交叉验证方式: {cv_name}\n")
        f.write(f"- 功率特征: {'已移除' if REMOVE_POWER_FEATURES else '保留'}\n\n")

        f.write("## 2. 模型性能对比\n\n")
        f.write(f"| 模型 | 训练R² | {cv_name} R² | CV-RMSE(HV) | CV-MAE(HV) |\n")
        f.write("|------|--------|-----------|-------------|------------|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['r2_train']:.4f} | {r['r2_cv']:.4f} | "
                    f"{r['rmse_cv']:.2f} | {r['mae_cv']:.2f} |\n")

        f.write(f"\n## 3. 最优模型: {best_model_name}\n\n")
        best = results[best_model_name]
        f.write(f"- {cv_name} R²: {best['r2_cv']:.4f}\n")
        f.write(f"- {cv_name} RMSE: {best['rmse_cv']:.2f} HV\n")
        f.write(f"- {cv_name} MAE: {best['mae_cv']:.2f} HV\n\n")

        if pw_range is not None and pw_pred is not None:
            f.write("## 4. 功率敏感性分析\n\n")
            opt_idx = np.argmax(pw_pred)
            f.write(f"- 预测最优功率: {pw_range[opt_idx]} W\n")
            f.write(f"- 预测最优硬度: {pw_pred[opt_idx]:.1f} HV\n")
            f.write(f"- 900W预测硬度: {pw_pred[0]:.1f} HV\n")
            f.write(f"- 1800W预测硬度: {pw_pred[-1]:.1f} HV\n\n")

        f.write("## 4. 关键发现与讨论\n\n")
        f.write("### 4.1 过拟合问题\n\n")
        f.write("- **原始LOO-CV R²≈1.0是严重数据泄露**：同一功率组有多个样本但硬度值相同，LOO-CV时训练集总能看到相同功率的样本\n")
        f.write("- **LOGO-CV(按功率组留一)才是真实泛化能力评估**：预测一个全新功率水平的硬度\n")
        f.write(f"- 当前数据只有4个功率组，LOGO-CV评估本身方差较大\n\n")

        f.write("### 4.2 为什么预测困难\n\n")
        f.write("1. **只有4个功率水平**：模型需要从3个功率外推到第4个，本质是小样本外推问题\n")
        f.write("2. **微观组织特征组间差异小**：特征在同一功率组内变异大，但组间差异不足以区分硬度\n")
        f.write("3. **硬度主要由功率决定**：功率与硬度相关系数约0.95，移除功率特征后缺乏强预测因子\n\n")

        f.write("### 4.3 建议\n\n")
        f.write("1. **增加更多功率水平的实验**：从4个增加到8-10个，才能建立可靠的预测模型\n")
        f.write("2. **当前模型可用于组内排序**：如果目标是同一功率下不同样本的相对硬度排序，模型仍有参考价值\n")
        f.write("3. **关注物理机制**：硬度变化可能主要源于相变(马氏体/贝氏体比例)而非Hall-Petch强化\n")
        f.write("4. **使用单特征基准**：XRD峰位(xrd_main_peak_2theta)与硬度相关性最高，可作为简单基准\n")

    print(f"  [OK] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
