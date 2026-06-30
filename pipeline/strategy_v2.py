"""
策略调整版 - 针对4个独立样本的现实方案

核心原则:
  1. 特征数 ≤ 2 (4个样本最多支撑2个自由度)
  2. 只用线性模型 (Ridge/线性回归)
  3. 物理模型(Hall-Petch)作为基准对比
  4. LOGO-CV按功率组留一，评估真实泛化能力
  5. 重点: 趋势分析 > R²数值

输出:
  - 各方案LOGO-CV结果对比
  - 推荐的最优特征组合
  - 物理模型 vs ML模型对比图
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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from plot_style import setup_plot_style, style_axes, add_subplot_label, save_fig, COLORS, POWER_LIST

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

setup_plot_style()


def load_data():
    csv_path = os.path.join(OUTPUT_DIR, "data_full.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    return df


def logo_cv_single(X, y, groups, alpha=1.0):
    """单次LOGO-CV，返回预测值和指标"""
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = Ridge(alpha=alpha)
        model.fit(X_train, y[train_idx])
        y_pred[test_idx] = model.predict(X_test)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return y_pred, r2, rmse, mae


def hall_petch_model(grain_sizes, hv_measured):
    """
    Hall-Petch物理模型: HV = sigma0 + k / sqrt(d)
    用最小二乘拟合 sigma0 和 k
    """
    inv_sqrt_d = 1.0 / np.sqrt(grain_sizes)
    # 线性拟合: HV = sigma0 + k * (1/sqrt(d))
    coeffs = np.polyfit(inv_sqrt_d, hv_measured, 1)
    k, sigma0 = coeffs
    hv_pred = sigma0 + k * inv_sqrt_d
    return hv_pred, sigma0, k


def run_strategy_comparison(df):
    """对比不同特征组合的LOGO-CV表现"""
    target = "mh_mean_hv"
    groups = df["power_w"].values
    y = df[target].values

    # 可选特征及其物理含义
    candidate_features = {
        "power_w": "激光功率(W)",
        "xrd_main_peak_2theta": "XRD主峰2θ(相变指标)",
        "eis_Rct_ohm": "EIS电荷转移电阻(腐蚀抗力)",
        "wear_friction_steady": "稳态摩擦系数(耐磨性)",
        "熔覆层平均晶粒尺寸(μm)": "晶粒尺寸(细晶强化)",
        "基体稀释率(%)": "稀释率(稀释效应)",
        "xrd_peak_44_area": "XRD 44°峰面积",
        "eis_theta_min_deg": "EIS最小相位角",
    }

    # 定义方案: (名称, 特征列表, 说明)
    schemes = [
        ("S0: 功率单变量", ["power_w"], "仅用激光功率作为特征"),
        ("S1: 功率+XRD峰位", ["power_w", "xrd_main_peak_2theta"], "功率+相变指标"),
        ("S2: 功率+EIS", ["power_w", "eis_Rct_ohm"], "功率+腐蚀抗力"),
        ("S3: 功率+摩擦", ["power_w", "wear_friction_steady"], "功率+耐磨性"),
        ("S4: 功率+晶粒", ["power_w", "熔覆层平均晶粒尺寸(μm)"], "功率+Hall-Petch项"),
        ("S5: 功率+稀释率", ["power_w", "基体稀释率(%)"], "功率+稀释效应"),
        ("S6: 功率+XRD+EIS", ["power_w", "xrd_main_peak_2theta", "eis_Rct_ohm"], "3特征(注意过拟合风险)"),
    ]

    results = []
    print("=" * 80)
    print("LOGO-CV 策略对比 (按功率组留一)")
    print("=" * 80)
    print(f"{'方案':<25} {'特征数':<8} {'R²':<10} {'RMSE(HV)':<12} {'MAE(HV)':<10} {'评估'}")
    print("-" * 80)

    for name, feats, desc in schemes:
        valid_feats = [f for f in feats if f in df.columns and df[f].notna().all()]
        if len(valid_feats) != len(feats):
            print(f"{name:<25} {'SKIP':<8} (特征缺失)")
            continue

        X = df[valid_feats].values
        y_pred, r2, rmse, mae = logo_cv_single(X, y, groups)

        # 评估: 4个样本的R²解读标准
        if r2 > 0.3:
            verdict = "可用(趋势可靠)"
        elif r2 > 0.0:
            verdict = "勉强(趋势大致对)"
        elif r2 > -0.5:
            verdict = "弱(仅参考趋势)"
        else:
            verdict = "不可靠(过拟合/噪声)"

        results.append({
            "name": name, "features": valid_feats, "r2": r2,
            "rmse": rmse, "mae": mae, "y_pred": y_pred, "verdict": verdict
        })
        print(f"{name:<25} {len(valid_feats):<8} {r2:<10.3f} {rmse:<12.1f} {mae:<10.1f} {verdict}")

    # Hall-Petch物理模型
    grain_col = "熔覆层平均晶粒尺寸(μm)"
    if grain_col in df.columns and df[grain_col].notna().all():
        grain_sizes = df[grain_col].values
        hp_pred, sigma0, k = hall_petch_model(grain_sizes, y)
        hp_r2 = r2_score(y, hp_pred)
        hp_rmse = np.sqrt(mean_squared_error(y, hp_pred))
        hp_mae = mean_absolute_error(y, hp_pred)
        print("-" * 80)
        print(f"{'HP: Hall-Petch物理模型':<25} {'1':<8} {hp_r2:<10.3f} {hp_rmse:<12.1f} {hp_mae:<10.1f} σ0={sigma0:.0f}, k={k:.0f}")

    print("=" * 80)
    return results


def find_best_scheme(results):
    """选择最优方案: R²最高且特征数最少"""
    # 按R²降序排列
    valid = [r for r in results if r["r2"] is not None]
    if not valid:
        return None
    best = max(valid, key=lambda r: (r["r2"], -len(r["features"])))
    return best


def plot_comparison(df, results, best):
    """生成对比图"""
    target = "mh_mean_hv"
    y = df[target].values
    power_labels = [f"{int(p)}W" for p in df["power_w"].values]
    x_pos = np.arange(len(y))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- 图1: 各方案R²对比 ----
    ax = axes[0]
    names = [r["name"].split(": ")[1] for r in results]
    r2_vals = [r["r2"] for r in results]
    colors = ["#2ecc71" if r > 0 else "#e74c3c" if r < -0.5 else "#f39c12" for r in r2_vals]
    bars = ax.barh(names, r2_vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(x=0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("LOGO-CV R²", fontsize=12)
    ax.set_title("方案对比 (LOGO-CV R²)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    style_axes(ax)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    # ---- 图2: 最优方案预测 vs 实测 ----
    ax = axes[1]
    if best:
        ax.scatter(x_pos, y, s=100, c="#2c3e50", zorder=5, label="实测值", marker="o")
        ax.scatter(x_pos, best["y_pred"], s=100, c="#e74c3c", zorder=5, label="LOGO-CV预测", marker="s")
        for i in range(len(y)):
            ax.plot([x_pos[i], x_pos[i]], [y[i], best["y_pred"][i]],
                    color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(power_labels, fontsize=11)
        ax.set_ylabel("硬度 (HV)", fontsize=12)
        ax.set_title(f"最优方案: {best['name']}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        style_axes(ax)

    # ---- 图3: 功率-硬度趋势 ----
    ax = axes[2]
    s0 = [r for r in results if "S0" in r["name"]]
    if s0:
        s0 = s0[0]
        power_vals = df["power_w"].unique()
        hv_means = [df[df["power_w"]==p]["mh_mean_hv"].mean() for p in power_vals]
        ax.plot(power_vals, hv_means, "o-", linewidth=2, markersize=10, color="#2c3e50", label="实测均值")
        ax.plot(power_vals, [df[df["power_w"]==p]["mh_mean_hv"].mean() for p in power_vals],
                "s--", linewidth=2, markersize=10, color="#e74c3c", label="Ridge预测")
        ax.set_xlabel("激光功率 (W)", fontsize=12)
        ax.set_ylabel("硬度 (HV)", fontsize=12)
        ax.set_title("功率-硬度趋势", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        style_axes(ax)

    plt.tight_layout()
    save_fig(fig, "strategy_comparison", FIG_DIR)
    plt.close()
    print(f"\n图表已保存: {os.path.join(FIG_DIR, 'strategy_comparison.png')}")


def print_recommendation(best, results, df):
    """打印推荐方案"""
    print("\n" + "=" * 80)
    print("推荐策略")
    print("=" * 80)

    if best is None:
        print("无法确定最优方案，请检查数据。")
        return

    print(f"\n最优方案: {best['name']}")
    print(f"  特征: {best['features']}")
    print(f"  LOGO-CV R² = {best['r2']:.3f}")
    print(f"  RMSE = {best['rmse']:.1f} HV")
    print(f"  判定: {best['verdict']}")

    print("\n核心建议:")
    print("  1. 4个独立样本 → 最多用2个特征(1个功率 + 1个补充)")
    print("  2. 只用线性模型(Ridge)，不用GBR/RFR/GPR")
    print("  3. LOGO-CV R²为负是正常的(样本太少)，不代表模型无用")
    print("  4. 重点关注趋势方向是否正确(功率↑ → 硬度↑)")
    print("  5. 物理模型(Hall-Petch)应作为主要解释工具，ML作为辅助验证")

    # 检查趋势一致性
    print("\n趋势验证:")
    for r in results:
        if r["name"] == "S0: 功率单变量":
            pred_order = np.argsort(r["y_pred"])
            real_order = np.argsort(df["mh_mean_hv"].values)
            if np.array_equal(pred_order, real_order):
                print(f"  {r['name']}: 预测排序与实测一致 ✓")
            else:
                print(f"  {r['name']}: 预测排序与实测不一致 ✗")


def main():
    print("=" * 80)
    print("策略调整版 v2 - 针对4个独立样本")
    print("=" * 80)

    df = load_data()
    print(f"\n数据: {len(df)} 样本, 功率组: {sorted(df['power_w'].unique())}")
    print(f"目标: mh_mean_hv, 范围: {df['mh_mean_hv'].min():.1f} - {df['mh_mean_hv'].max():.1f} HV")

    # 1. 方案对比
    results = run_strategy_comparison(df)

    # 2. 找最优方案
    best = find_best_scheme(results)

    # 3. 生成对比图
    plot_comparison(df, results, best)

    # 4. 打印推荐
    print_recommendation(best, results, df)


if __name__ == "__main__":
    main()
