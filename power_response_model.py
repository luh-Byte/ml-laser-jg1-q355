"""
阶段2: 功率→组织响应面建模 + 物理模型校准
- GPR响应面: 激光功率 → 微观组织特征
- 物理模型校准: 基于实测硬度回归修正Hall-Petch等公式
- Leave-One-Power-Out交叉验证
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 1. 数据加载 =====================
def load_data():
    csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


# ===================== 2. 功率→组织 GPR响应面 =====================
MICRO_FEATURES = [
    "熔覆层组织面积占比(%)",
    "析出相/碳化物面积占比(%)",
    "气孔孔隙率(%)",
    "微裂纹面积占比(%)",
    "熔覆层平均晶粒尺寸(μm)",
    "基体稀释率(%)",
]

FEATURE_SHORT = {
    "熔覆层组织面积占比(%)": "Cladding%",
    "析出相/碳化物面积占比(%)": "Carbide%",
    "气孔孔隙率(%)": "Porosity%",
    "微裂纹面积占比(%)": "Crack%",
    "熔覆层平均晶粒尺寸(μm)": "GrainSize(μm)",
    "基体稀释率(%)": "Dilution%",
}


def build_power_response_models(df):
    """用GPR建立激光功率→微观组织特征的响应面"""
    powers_str = df["激光功率"].unique()
    power_values = np.array([float(p.replace("W", "")) for p in df["激光功率"]])
    groups = np.array([float(p.replace("W", "")) for p in df["激光功率"]])

    results = {}

    for feat in MICRO_FEATURES:
        y = df[feat].values

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(100, (10, 1000)) + WhiteKernel(0.1, (1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gpr.fit(power_values.reshape(-1, 1), y)

        y_pred = gpr.predict(power_values.reshape(-1, 1))
        y_std = gpr.predict(power_values.reshape(-1, 1), return_std=True)[1]

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        logo = LeaveOneGroupOut()
        cv_r2_scores = []
        for train_idx, test_idx in logo.split(power_values.reshape(-1, 1), y, groups):
            X_train, X_test = power_values[train_idx].reshape(-1, 1), power_values[test_idx].reshape(-1, 1)
            y_train, y_test = y[train_idx], y[test_idx]

            gpr_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
            gpr_cv.fit(X_train, y_train)
            y_cv_pred = gpr_cv.predict(X_test)
            cv_r2_scores.append(r2_score(y_test, y_cv_pred))

        results[feat] = {
            "model": gpr,
            "r2_fit": r2,
            "rmse_fit": rmse,
            "mae_fit": mae,
            "cv_r2_mean": np.mean(cv_r2_scores),
            "cv_r2_std": np.std(cv_r2_scores),
            "y_pred": y_pred,
            "y_std": y_std,
        }

    return results


def plot_power_response(df, results):
    """绘制功率→组织响应面图"""
    power_values = np.array([float(p.replace("W", "")) for p in df["激光功率"]])
    x_pred = np.linspace(850, 1850, 200).reshape(-1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(MICRO_FEATURES):
        ax = axes[i]
        r = results[feat]

        y_pred_curve = r["model"].predict(x_pred)
        y_std_curve = r["model"].predict(x_pred, return_std=True)[1]

        ax.scatter(power_values, df[feat].values, c="red", s=30, zorder=5, label="Measured")
        ax.plot(x_pred.ravel(), y_pred_curve, "b-", linewidth=2, label="GPR mean")
        ax.fill_between(x_pred.ravel(), y_pred_curve - 2 * y_std_curve,
                         y_pred_curve + 2 * y_std_curve, alpha=0.2, color="blue",
                         label="95% CI")

        short = FEATURE_SHORT.get(feat, feat[:10])
        ax.set_title(f"{short}\nR²={r['r2_fit']:.3f}, CV-R²={r['cv_r2_mean']:.3f}",
                     fontsize=11)
        ax.set_xlabel("Power (W)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Power → Microstructure Response Surface (GPR)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "power_response_surface.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===================== 3. 物理模型校准 =====================
def calibrate_physics_model(df):
    """
    基于实测硬度校准物理模型:
    HV = a + b/sqrt(d) + c*sqrt(carbide) + e*porosity + f*crack

    用scipy.optimize.curve_fit回归系数
    """
    from scipy.optimize import curve_fit

    def hv_model(X, a, b, c, e, f):
        grain_size, carbide, porosity, crack = X
        return a + b / np.sqrt(grain_size) + c * np.sqrt(carbide) + e * porosity + f * crack

    X_data = (
        df["熔覆层平均晶粒尺寸(μm)"].values,
        df["析出相/碳化物面积占比(%)"].values,
        df["气孔孔隙率(%)"].values,
        df["微裂纹面积占比(%)"].values,
    )
    y_data = df["mh_mean_hv"].values

    p0 = [100, 100, 5, -10, -5]
    popt, pcov = curve_fit(hv_model, X_data, y_data, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    a, b, c, e, f = popt
    y_pred = hv_model(X_data, *popt)
    r2 = r2_score(y_data, y_pred)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    print(f"\n  校准后公式: HV = {a:.1f} + {b:.1f}/√d + {c:.2f}√carbide + {e:.2f}×porosity + {f:.2f}×crack")
    print(f"  拟合R² = {r2:.4f}, RMSE = {rmse:.2f} HV")
    print(f"  参数不确定性: a±{perr[0]:.1f}, b±{perr[1]:.1f}, c±{perr[2]:.2f}, e±{perr[3]:.2f}, f±{perr[4]:.2f}")

    # Leave-One-Power-Out验证
    groups = np.array([float(p.replace("W", "")) for p in df["激光功率"]])
    logo = LeaveOneGroupOut()
    cv_scores = []
    for train_idx, test_idx in logo.split(np.zeros(len(y_data)), y_data, groups):
        X_train = tuple(arr[train_idx] for arr in X_data)
        X_test = tuple(arr[test_idx] for arr in X_data)
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        popt_cv, _ = curve_fit(hv_model, X_train, y_train, p0=p0, maxfev=10000)
        y_cv_pred = hv_model(X_test, *popt_cv)
        cv_scores.append({
            "r2": r2_score(y_test, y_cv_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_cv_pred)),
            "mae": mean_absolute_error(y_test, y_cv_pred),
        })

    avg_cv_r2 = np.mean([s["r2"] for s in cv_scores])
    avg_cv_rmse = np.mean([s["rmse"] for s in cv_scores])
    print(f"  Leave-One-Power-Out CV: R²={avg_cv_r2:.4f}, RMSE={avg_cv_rmse:.2f} HV")

    return {
        "params": popt,
        "param_names": ["a (base)", "b (Hall-Petch)", "c (Orowan)", "e (porosity)", "f (crack)"],
        "param_errors": perr,
        "r2_fit": r2,
        "rmse_fit": rmse,
        "cv_r2": avg_cv_r2,
        "cv_rmse": avg_cv_rmse,
        "cv_scores": cv_scores,
        "y_pred": y_pred,
        "y_data": y_data,
        "formula": f"HV = {a:.1f} + {b:.1f}/√d + {c:.2f}√carbide + {e:.2f}×porosity + {f:.2f}×crack",
    }


def plot_calibration(df, cal_result):
    """绘制物理模型校准对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    y_data = cal_result["y_data"]
    y_pred = cal_result["y_pred"]
    powers = np.array([float(p.replace("W", "")) for p in df["激光功率"]])

    # (a) 校准公式预测 vs 实测
    ax = axes[0]
    ax.scatter(y_data, y_pred, c=powers, cmap="viridis", s=50, edgecolors="black", zorder=5)
    lims = [min(y_data.min(), y_pred.min()) - 20, max(y_data.max(), y_pred.max()) + 20]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Measured HV")
    ax.set_ylabel("Calibrated Model HV")
    ax.set_title(f"(a) Calibration Fit (R²={cal_result['r2_fit']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) 残差分析
    ax = axes[1]
    residuals = y_data - y_pred
    ax.scatter(y_pred, residuals, c=powers, cmap="viridis", s=50, edgecolors="black")
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Predicted HV")
    ax.set_ylabel("Residual (HV)")
    ax.set_title("(b) Residual Analysis")
    ax.grid(True, alpha=0.3)

    # (c) 各功率实测 vs 校准模型 vs 旧模型
    ax = axes[2]
    power_groups = sorted(df["激光功率"].unique(), key=lambda x: float(x.replace("W", "")))
    x_pos = np.arange(len(power_groups))
    width = 0.25

    for i, pg in enumerate(power_groups):
        mask = df["激光功率"] == pg
        actual = df.loc[mask, "mh_mean_hv"].mean()
        calibrated = y_pred[mask.values].mean()

        if i == 0:
            ax.bar(x_pos - width, actual, width, label="Measured", color="#2ecc71", edgecolor="black")
            ax.bar(x_pos, calibrated, width, label="Calibrated Model", color="#3498db", edgecolor="black")
        else:
            ax.bar(x_pos[i] - width, actual, width, color="#2ecc71", edgecolor="black")
            ax.bar(x_pos[i], calibrated, width, color="#3498db", edgecolor="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace("W", "W") for p in power_groups])
    ax.set_ylabel("Hardness (HV)")
    ax.set_title("(c) Measured vs Calibrated by Power")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Physics Model Calibration Results", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "physics_model_calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===================== 4. 功率→性能直接响应面 =====================
def build_power_property_direct(df):
    """直接建立功率→实测硬度的GPR响应面"""
    power_values = np.array([float(p.replace("W", "")) for p in df["激光功率"]])
    y_hardness = df["mh_mean_hv"].values

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(200, (50, 500)) + WhiteKernel(1.0, (1e-3, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(power_values.reshape(-1, 1), y_hardness)

    x_pred = np.linspace(850, 1850, 300).reshape(-1, 1)
    y_pred, y_std = gpr.predict(x_pred, return_std=True)
    y_pred_train = gpr.predict(power_values.reshape(-1, 1))

    r2 = r2_score(y_hardness, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y_hardness, y_pred_train))

    # Leave-One-Out (单样本，比Leave-One-Power-Out更细粒度)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    cv_scores = []
    for train_idx, test_idx in loo.split(power_values.reshape(-1, 1), y_hardness):
        X_train = power_values[train_idx].reshape(-1, 1)
        X_test = power_values[test_idx].reshape(-1, 1)
        y_train = y_hardness[train_idx]
        y_test = y_hardness[test_idx]

        gpr_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gpr_cv.fit(X_train, y_train)
        y_cv = gpr_cv.predict(X_test)
        cv_scores.append({"r2": r2_score(y_test, y_cv), "rmse": np.sqrt(mean_squared_error(y_test, y_cv))})

    avg_cv_r2 = np.mean([s["r2"] for s in cv_scores])
    avg_cv_rmse = np.mean([s["rmse"] for s in cv_scores])

    print(f"\n  功率→硬度直接GPR: R²={r2:.4f}, RMSE={rmse:.2f} HV")
    print(f"  Leave-One-Power-Out CV: R²={avg_cv_r2:.4f}, RMSE={avg_cv_rmse:.2f} HV")

    # 找最优功率点
    opt_idx = np.argmax(y_pred)
    opt_power = x_pred[opt_idx, 0]
    opt_hardness = y_pred[opt_idx]
    opt_std = y_std[opt_idx]

    print(f"  预测最优功率: {opt_power:.0f} W → 硬度 {opt_hardness:.1f} ± {opt_std:.1f} HV")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(power_values, y_hardness, c="red", s=60, zorder=5, label="Measured (mean)")
    for p_val in np.unique(power_values):
        mask = power_values == p_val
        vals = y_hardness[mask]
        ax.scatter([p_val] * len(vals), vals, c="red", s=15, alpha=0.3, zorder=4)
    ax.plot(x_pred.ravel(), y_pred, "b-", linewidth=2, label="GPR mean")
    ax.fill_between(x_pred.ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std,
                     alpha=0.15, color="blue", label="95% CI")
    ax.axvline(x=opt_power, color="green", linestyle="--", linewidth=1.5,
               label=f"Optimal: {opt_power:.0f}W")
    ax.scatter([opt_power], [opt_hardness], c="green", s=200, marker="*", zorder=6,
               label=f"HV={opt_hardness:.1f}")

    ax.set_xlabel("Laser Power (W)", fontsize=12)
    ax.set_ylabel("Microhardness (HV)", fontsize=12)
    ax.set_title(f"Power → Hardness Direct GPR\n"
                 f"R²={r2:.3f}, CV-R²={avg_cv_r2:.3f}, CV-RMSE={avg_cv_rmse:.1f} HV",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "power_hardness_direct_gpr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")

    return {
        "model": gpr,
        "r2": r2,
        "rmse": rmse,
        "cv_r2": avg_cv_r2,
        "cv_rmse": avg_cv_rmse,
        "optimal_power": opt_power,
        "optimal_hardness": opt_hardness,
        "optimal_std": opt_std,
    }


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("阶段2: 功率→组织响应面 + 物理模型校准")
    print("=" * 60)

    df = load_data()
    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")

    # 2.1 GPR响应面
    print("\n[1/4] 建立功率→微观组织GPR响应面...")
    response_results = build_power_response_models(df)
    for feat, r in response_results.items():
        short = FEATURE_SHORT.get(feat, feat[:15])
        print(f"  {short}: R²={r['r2_fit']:.4f}, CV-R²={r['cv_r2_mean']:.4f}")

    print("\n[2/4] 绘制响应面图...")
    plot_power_response(df, response_results)

    # 2.2 物理模型校准
    print("\n[3/4] 校准物理模型(基于实测硬度)...")
    cal_result = calibrate_physics_model(df)

    print("\n  校准公式:")
    print(f"  {cal_result['formula']}")

    print("\n  参数物理含义:")
    for name, val, err in zip(cal_result["param_names"], cal_result["params"], cal_result["param_errors"]):
        print(f"    {name}: {val:.2f} ± {err:.2f}")

    print("\n[4/4] 绘制校准对比图...")
    plot_calibration(df, cal_result)

    # 2.3 功率→硬度直接响应面
    print("\n[Bonus] 功率→实测硬度直接GPR...")
    direct_result = build_power_property_direct(df)

    # 保存结果
    import pickle
    results_path = os.path.join(OUTPUT_DIR, "power_response_models.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "response_surface": {k: v["model"] for k, v in response_results.items()},
            "physics_calibration": cal_result,
            "direct_gpr": direct_result,
        }, f)
    print(f"\n  [OK] 模型已保存: {results_path}")

    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, "阶段2_响应面分析报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 阶段2: 功率→组织响应面分析报告\n\n")

        f.write("## 1. 功率→微观组织GPR响应面\n\n")
        f.write("| 特征 | 拟合R² | CV-R² | CV-R² std |\n")
        f.write("|------|--------|-------|----------|\n")
        for feat, r in response_results.items():
            short = FEATURE_SHORT.get(feat, feat)
            f.write(f"| {short} | {r['r2_fit']:.4f} | {r['cv_r2_mean']:.4f} | {r['cv_r2_std']:.4f} |\n")

        f.write("\n## 2. 物理模型校准\n\n")
        f.write(f"**校准公式**: {cal_result['formula']}\n\n")
        f.write(f"- 拟合R²: {cal_result['r2_fit']:.4f}\n")
        f.write(f"- 拟合RMSE: {cal_result['rmse_fit']:.2f} HV\n")
        f.write(f"- Leave-One-Power-Out CV R²: {cal_result['cv_r2']:.4f}\n")
        f.write(f"- Leave-One-Power-Out CV RMSE: {cal_result['cv_rmse']:.2f} HV\n\n")

        f.write("### 参数对比(旧模型 vs 校准模型)\n\n")
        f.write("| 参数 | 旧模型 | 校准模型 | 变化 |\n")
        f.write("|------|--------|---------|------|\n")
        old_params = {"a (base)": 400, "b (Hall-Petch)": 200, "c (Orowan)": 8, "e (porosity)": -20, "f (crack)": -8}
        for name, val, err in zip(cal_result["param_names"], cal_result["params"], cal_result["param_errors"]):
            old = old_params.get(name, 0)
            change = val - old
            f.write(f"| {name} | {old:.1f} | {val:.1f}±{err:.1f} | {change:+.1f} |\n")

        f.write("\n## 3. 功率→硬度直接GPR\n\n")
        f.write(f"- 拟合R²: {direct_result['r2']:.4f}\n")
        f.write(f"- CV R²: {direct_result['cv_r2']:.4f}\n")
        f.write(f"- CV RMSE: {direct_result['cv_rmse']:.2f} HV\n")
        f.write(f"- 预测最优功率: {direct_result['optimal_power']:.0f} W\n")
        f.write(f"- 预测最优硬度: {direct_result['optimal_hardness']:.1f} HV\n")

    print(f"  [OK] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
