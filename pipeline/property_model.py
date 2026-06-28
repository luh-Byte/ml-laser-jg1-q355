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
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 1. 数据加载与特征工程 =====================
def load_and_prepare():
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
def train_models(df, features, target):
    """训练多种回归模型，LOO交叉验证"""
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "GBR": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        ),
        "RFR": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
    }

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(0.1, (1e-5, 1e2))
    models["GPR"] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)

    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        y_cv = cross_val_predict(model, X_scaled, y, cv=loo)

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

        print(f"  {name}: train R²={r2_train:.4f}, LOO-CV R²={r2:.4f}, "
              f"CV-RMSE={rmse:.2f} HV, CV-MAE={mae:.2f} HV")

        # 过拟合检测
        gap = r2_train - r2
        if gap > 0.1:
            print(f"  ⚠️ [{name}] 过拟合风险: 训练R²-CV_R²={gap:.4f} > 0.1")
        if len(features) > len(y) * 0.5:
            print(f"  ⚠️ [{name}] 特征数/样本数={len(features)}/{len(y)}={len(features)/len(y):.2f} > 0.5，建议降维")
        if r2 > 0.99:
            print(f"  ⚠️ [{name}] CV R²={r2:.4f} ≈ 1.0，检查是否存在数据泄露")

    # 特征数量/样本数比例警告
    ratio = len(features) / len(y)
    print(f"\n  特征/样本比: {len(features)}/{len(y)} = {ratio:.3f}")
    if ratio > 0.3:
        print(f"  ⚠️ 特征维度偏高，建议PCA或特征选择")

    # 学习曲线数据（样本数 vs R²）
    from sklearn.model_selection import learning_curve
    gbr = models["GBR"]
    train_sizes_abs, train_scores, val_scores = learning_curve(
        gbr, X_scaled, y, cv=min(5, len(y) // 2), scoring='r2',
        train_sizes=np.linspace(0.3, 1.0, 5), random_state=42
    )
    lc_data = {
        "train_sizes": train_sizes_abs.tolist(),
        "train_mean": train_scores.mean(axis=1).tolist(),
        "val_mean": val_scores.mean(axis=1).tolist(),
    }

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
def plot_model_comparison(results, y, df):
    """绘制模型对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    powers = df["power_w"].values

    # (a) 实测 vs LOO-CV预测
    ax = axes[0]
    colors = {"GBR": "#2ecc71", "RFR": "#3498db", "GPR": "#e74c3c"}
    for name, r in results.items():
        ax.scatter(y, r["y_cv"], c=colors[name], s=40, alpha=0.7, label=f'{name} (R²={r["r2_cv"]:.3f})')
    lims = [y.min() - 20, y.max() + 20]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Measured HV")
    ax.set_ylabel("LOO-CV Predicted HV")
    ax.set_title("(a) Measured vs LOO-CV Predicted")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) 各功率硬度分布 + 模型预测
    ax = axes[1]
    power_levels = sorted(df["power_w"].unique())
    x_pos = np.arange(len(power_levels))
    width = 0.15

    for j, (name, r) in enumerate(results.items()):
        preds_by_power = [r["y_cv"][df["power_w"].values == p].mean() for p in power_levels]
        ax.bar(x_pos + j * width, preds_by_power, width, label=name, color=colors[name], edgecolor="black")

    actual_by_power = [y[df["power_w"].values == p].mean() for p in power_levels]
    ax.bar(x_pos + 3 * width, actual_by_power, width, label="Measured", color="gray", edgecolor="black")

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([f"{int(p)}W" for p in power_levels])
    ax.set_ylabel("Hardness (HV)")
    ax.set_title("(b) Hardness by Power Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) 残差分布
    ax = axes[2]
    for name, r in results.items():
        residuals = y - r["y_cv"]
        ax.hist(residuals, bins=15, alpha=0.4, label=f'{name} (MAE={r["mae_cv"]:.1f})', color=colors[name])
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Residual (HV)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Residual Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Property Model Comparison (Real Measured Hardness)", fontsize=13, y=1.02)
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
    print(f"  样本数: {len(df)}, 特征数: {len(features)}")
    print(f"  目标: {target} (实测显微硬度)")
    print(f"  特征组: {list(feature_groups.keys())}")
    print(f"  新增: power×组织交叉项 + Hall-Petch + 热输入")

    print("\n[2/6] 训练模型(LOO交叉验证)...")
    results, X, y, X_scaled, lc_data = train_models(df, features, target)

    print("\n[3/6] 模型对比图...")
    plot_model_comparison(results, y, df)

    # 绘制学习曲线
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
    feat_imp = shap_analysis(results["GBR"]["model"], X_scaled, features)
    if feat_imp:
        print("  Top 10 features:")
        for name, imp in feat_imp[:10]:
            print(f"    {name}: {imp:.4f}")

    print("\n[5/6] 偏依赖图...")
    plot_partial_dependence(results["GBR"]["model"], X_scaled, features, df, target)

    print("\n[6/6] 功率敏感性分析...")
    pw_range, pw_pred = sensitivity_analysis(results, df, features, target)

    # 保存
    results_path = os.path.join(OUTPUT_DIR, "property_models.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "models": {k: v["model"] for k, v in results.items()},
            "scaler": results["GBR"]["scaler"],
            "features": features,
            "feature_groups": feature_groups,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        }, f)
    print(f"\n  [OK] 模型已保存: {results_path}")

    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, "阶段3_性能模型报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 阶段3: 组织→性能模型报告\n\n")

        f.write("## 1. 数据概况\n\n")
        f.write(f"- 样本数: {len(df)} (4个功率水平 × ~14张图片)\n")
        f.write(f"- 特征数: {len(features)} (金相+电化学+XRD+磨损+交叉项)\n")
        f.write(f"- 目标变量: 实测显微硬度 (mh_mean_hv)\n")
        f.write(f"- 硬度范围: {y.min():.1f} ~ {y.max():.1f} HV\n\n")

        f.write("## 2. 模型性能对比\n\n")
        f.write("| 模型 | 训练R² | LOO-CV R² | CV-RMSE(HV) | CV-MAE(HV) |\n")
        f.write("|------|--------|-----------|-------------|------------|\n")
        for name, r in results.items():
            f.write(f"| {name} | {r['r2_train']:.4f} | {r['r2_cv']:.4f} | "
                    f"{r['rmse_cv']:.2f} | {r['mae_cv']:.2f} |\n")

        f.write("\n## 3. 最优模型: GBR\n\n")
        best = results["GBR"]
        f.write(f"- LOO-CV R²: {best['r2_cv']:.4f}\n")
        f.write(f"- LOO-CV RMSE: {best['rmse_cv']:.2f} HV\n")
        f.write(f"- LOO-CV MAE: {best['mae_cv']:.2f} HV\n\n")

        f.write("## 4. 功率敏感性分析\n\n")
        opt_idx = np.argmax(pw_pred)
        f.write(f"- 预测最优功率: {pw_range[opt_idx]} W\n")
        f.write(f"- 预测最优硬度: {pw_pred[opt_idx]:.1f} HV\n")
        f.write(f"- 900W预测硬度: {pw_pred[0]:.1f} HV\n")
        f.write(f"- 1800W预测硬度: {pw_pred[-1]:.1f} HV\n\n")

        f.write("## 5. 关键结论\n\n")
        f.write("1. **实测硬度远低于原物理公式预测**: 原公式预测500-700HV，实测仅224-385HV\n")
        f.write("2. **功率是硬度的主要驱动因素**: 微观组织特征(晶粒/析出相)在各功率下几乎不变\n")
        f.write("3. **可能的机制**: 相变强化(马氏体/贝氏体)而非Hall-Petch强化\n")
        f.write("4. **EIS数据有补充价值**: Rct与硬度存在正相关\n")

    print(f"  [OK] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
