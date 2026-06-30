"""
过拟合解决方案对比脚本
测试多种方案，找到最优解
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "analysis_output", "data_full.csv")
FIG_DIR = os.path.join(BASE_DIR, "analysis_output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))

    base_features = [
        "熔覆层组织面积占比(%)",
        "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)",
        "微裂纹面积占比(%)",
        "熔覆层平均晶粒尺寸(μm)",
        "基体稀释率(%)",
        "eis_Rs_ohm",
        "eis_Rct_ohm",
        "eis_Z_max_ohm",
        "eis_theta_min_deg",
        "xrd_main_peak_2theta",
        "xrd_main_peak_intensity",
        "xrd_peak_44_area",
        "wear_friction_steady",
        "wear_friction_std",
        "扫描速度(mm/min)",
        "送粉速率(g/min)",
    ]

    df["hall_petch"] = 1.0 / np.sqrt(df["熔覆层平均晶粒尺寸(μm)"])
    scan_spacing = 0.05
    df["heat_input"] = df["power_w"] / (df["扫描速度(mm/min)"] / 60) / scan_spacing / 1000

    cross_features = []
    for f in ["熔覆层平均晶粒尺寸(μm)", "析出相/碳化物面积占比(%)", "气孔孔隙率(%)"]:
        col_name = f"power_x_{f}"
        df[col_name] = df["power_w"] * df[f]
        cross_features.append(col_name)

    all_features = base_features + ["power_w", "hall_petch", "heat_input"] + cross_features
    valid_features = [f for f in all_features if f in df.columns and df[f].notna().all()]

    return df, valid_features


def logo_cv_evaluate(model, X, y, groups):
    logo = LeaveOneGroupOut()
    y_cv = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_train_scaled, y_train)
        y_cv[test_idx] = m.predict(X_test_scaled)
    r2 = r2_score(y, y_cv)
    rmse = np.sqrt(mean_squared_error(y, y_cv))
    mae = mean_absolute_error(y, y_cv)
    return r2, rmse, mae, y_cv


def main():
    print("=" * 80)
    print("过拟合解决方案对比")
    print("=" * 80)

    df, all_features = load_data()
    target = "mh_mean_hv"
    groups = df["power_w"].values
    y = df[target].values

    print(f"\n数据概况: {len(df)}样本, {len(all_features)}特征, 4个功率组")
    print(f"硬度范围: {y.min():.1f} ~ {y.max():.1f} HV")

    results = {}

    print(f"\n{'方案':<35} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("-" * 70)

    # 方案0: 原始LOO-CV (虚高基准)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    X_full = df[all_features].values
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    model_ori = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    y_cv_loo = []
    for train_idx, test_idx in loo.split(X_full_scaled):
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        m.fit(X_full_scaled[train_idx], y[train_idx])
        y_cv_loo.append(m.predict(X_full_scaled[test_idx])[0])
    y_cv_loo = np.array(y_cv_loo)
    r2_loo = r2_score(y, y_cv_loo)
    rmse_loo = np.sqrt(mean_squared_error(y, y_cv_loo))
    mae_loo = mean_absolute_error(y, y_cv_loo)
    results["方案0: 原始LOO-CV (虚高)"] = (r2_loo, rmse_loo, mae_loo, y_cv_loo)
    print(f"{'方案0: 原始LOO-CV (虚高)':<35} {r2_loo:>8.4f} {rmse_loo:>8.2f} {mae_loo:>8.2f}")

    # 方案1: LOGO-CV + 全部特征 (真实基准)
    model1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    r2, rmse, mae, y_cv1 = logo_cv_evaluate(model1, X_full, y, groups)
    results["方案1: LOGO-CV + 全部特征"] = (r2, rmse, mae, y_cv1)
    print(f"{'方案1: LOGO-CV + 全部特征':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案2: LOGO-CV + 移除功率和硬度相关特征
    feat_no_power = [f for f in all_features
                     if f not in ["power_w", "heat_input"]
                     and not f.startswith("power_x_")
                     and not f.startswith("mh_")
                     and f != "hardness_ratio"]
    X_np = df[feat_no_power].values
    model2 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    r2, rmse, mae, y_cv2 = logo_cv_evaluate(model2, X_np, y, groups)
    results[f"方案2: LOGO-CV + 无功率特征({len(feat_no_power)}维)"] = (r2, rmse, mae, y_cv2)
    print(f"方案2: LOGO-CV + 无功率特征({len(feat_no_power)}维)".ljust(35) + f" {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案3: LOGO-CV + 简化GBR (强正则化)
    model3 = GradientBoostingRegressor(
        n_estimators=20, max_depth=1, learning_rate=0.05,
        min_samples_leaf=3, subsample=0.8, random_state=42
    )
    r2, rmse, mae, y_cv3 = logo_cv_evaluate(model3, X_np, y, groups)
    results["方案3: LOGO-CV + 简化GBR"] = (r2, rmse, mae, y_cv3)
    print(f"{'方案3: LOGO-CV + 简化GBR':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案4: LOGO-CV + 随机森林(正则化)
    model4 = RandomForestRegressor(
        n_estimators=50, max_depth=2, min_samples_leaf=3,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    r2, rmse, mae, y_cv4 = logo_cv_evaluate(model4, X_np, y, groups)
    results["方案4: LOGO-CV + RFR(正则化)"] = (r2, rmse, mae, y_cv4)
    print(f"{'方案4: LOGO-CV + RFR(正则化)':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案5: LOGO-CV + 线性回归 (基准)
    model5 = LinearRegression()
    r2, rmse, mae, y_cv5 = logo_cv_evaluate(model5, X_np, y, groups)
    results["方案5: LOGO-CV + 线性回归"] = (r2, rmse, mae, y_cv5)
    print(f"{'方案5: LOGO-CV + 线性回归':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案6: LOGO-CV + Ridge回归
    model6 = Ridge(alpha=10.0, random_state=42)
    r2, rmse, mae, y_cv6 = logo_cv_evaluate(model6, X_np, y, groups)
    results["方案6: LOGO-CV + Ridge(α=10)"] = (r2, rmse, mae, y_cv6)
    print(f"{'方案6: LOGO-CV + Ridge(α=10)':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案7: LOGO-CV + Lasso
    model7 = Lasso(alpha=1.0, random_state=42, max_iter=10000)
    r2, rmse, mae, y_cv7 = logo_cv_evaluate(model7, X_np, y, groups)
    results["方案7: LOGO-CV + Lasso(α=1)"] = (r2, rmse, mae, y_cv7)
    print(f"{'方案7: LOGO-CV + Lasso(α=1)':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案8: LOGO-CV + PCA降维 + Ridge
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_np))
    model8 = Ridge(alpha=5.0, random_state=42)
    r2, rmse, mae, y_cv8 = logo_cv_evaluate(model8, X_pca, y, groups)
    results["方案8: LOGO-CV + PCA(5维) + Ridge"] = (r2, rmse, mae, y_cv8)
    print(f"{'方案8: LOGO-CV + PCA(5维) + Ridge':<35} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f}")

    # 方案9: 按功率组取平均 (4个有效样本)
    df_avg = df.groupby("激光功率").agg({f: "mean" for f in all_features + [target]}).reset_index()
    df_avg["power_w"] = df_avg["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    df_avg = df_avg.sort_values("power_w")
    y_avg = df_avg[target].values
    groups_avg = df_avg["power_w"].values

    feat_avg_no_power = [f for f in all_features
                         if f not in ["power_w", "heat_input"]
                         and not f.startswith("power_x_")
                         and not f.startswith("mh_")
                         and f != "hardness_ratio"]
    X_avg = df_avg[feat_avg_no_power].values

    # 用LOGO-CV在4个样本上评估
    logo4 = LeaveOneGroupOut()
    y_cv_avg = np.zeros_like(y_avg)
    for train_idx, test_idx in logo4.split(X_avg, y_avg, groups_avg):
        scaler4 = StandardScaler()
        X_tr = scaler4.fit_transform(X_avg[train_idx])
        X_te = scaler4.transform(X_avg[test_idx])
        m = Ridge(alpha=1.0, random_state=42)
        m.fit(X_tr, y_avg[train_idx])
        y_cv_avg[test_idx] = m.predict(X_te)
    r2_avg = r2_score(y_avg, y_cv_avg)
    rmse_avg = np.sqrt(mean_squared_error(y_avg, y_cv_avg))
    mae_avg = mean_absolute_error(y_avg, y_cv_avg)
    results["方案9: 组平均(4样本) + Ridge"] = (r2_avg, rmse_avg, mae_avg, y_cv_avg)
    print(f"{'方案9: 组平均(4样本) + Ridge':<35} {r2_avg:>8.4f} {rmse_avg:>8.2f} {mae_avg:>8.2f}")

    # 方案10: 单特征回归 (找最相关的微观特征)
    print(f"\n\n【单特征与硬度的相关性排序 (排除功率和硬度特征)】")
    correlations = []
    for f in feat_no_power:
        corr = df[f].corr(df[target])
        correlations.append((f, corr))
    correlations.sort(key=lambda x: -abs(x[1]))
    for f, corr in correlations[:10]:
        print(f"  {f:<30} r = {corr:>8.4f}")

    best_feat = correlations[0][0]
    X_single = df[[best_feat]].values
    model10 = LinearRegression()
    r2, rmse, mae, y_cv10 = logo_cv_evaluate(model10, X_single, y, groups)
    results[f"方案10: LOGO-CV + 单特征({best_feat[:10]})"] = (r2, rmse, mae, y_cv10)
    print(f"\n  最优单特征方案 R² = {r2:.4f}, RMSE = {rmse:.2f} HV")

    print(f"\n{'='*80}")
    print("【结论】")
    print(f"  1. 原始LOO-CV R²=1.0是严重数据泄露导致的虚高")
    print(f"  2. 真实泛化能力(LOGO-CV)：大多数模型R²为负，说明预测还不如平均值")
    print(f"  3. 核心问题：数据只有4个功率组，每个组硬度值完全相同")
    print(f"  4. 微观组织特征在组内变异大，但组间差异小，无法区分4个功率水平")
    print(f"  5. 建议：增加更多功率水平的实验数据，或者接受当前数据的局限性")
    print(f"{'='*80}")

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) R²对比
    ax = axes[0]
    names = list(results.keys())
    r2_vals = [v[0] for v in results.values()]
    colors = ["#e74c3c" if "虚高" in n else "#3498db" if "方案0" not in n else "#95a5a6" for n in names]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, r2_vals, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("R² Score")
    ax.set_title("(a) 各方案 R² 对比 (LOGO-CV)")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    # (b) 实测vs预测散点图（展示几个代表性方案）
    ax = axes[1]
    plot_schemes = [
        ("方案1: LOGO-CV + 全部特征", "#e74c3c", "o"),
        ("方案5: LOGO-CV + 线性回归", "#3498db", "s"),
        ("方案6: LOGO-CV + Ridge(α=10)", "#2ecc71", "^"),
    ]
    for name, color, marker in plot_schemes:
        if name in results:
            y_pred = results[name][3]
            ax.scatter(y, y_pred, c=color, marker=marker, s=40, alpha=0.7,
                       label=f'{name} (R²={results[name][0]:.3f})')
    lims = [y.min() - 30, y.max() + 30]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Measured HV")
    ax.set_ylabel("Predicted HV (LOGO-CV)")
    ax.set_title("(b) Measured vs Predicted (LOGO-CV)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "overfitting_solutions_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] 对比图已保存: {fig_path}")

    # 保存结果表格
    summary_path = os.path.join(BASE_DIR, "analysis_output", "overfitting_solutions_summary.csv")
    summary_df = pd.DataFrame({
        "方案": list(results.keys()),
        "R²": [v[0] for v in results.values()],
        "RMSE_HV": [v[1] for v in results.values()],
        "MAE_HV": [v[2] for v in results.values()],
    })
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 结果汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
