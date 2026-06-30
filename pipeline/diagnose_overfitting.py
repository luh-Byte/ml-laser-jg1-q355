"""
过拟合诊断脚本
分析当前ML模型过拟合的根本原因
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "analysis_output", "data_full.csv")

def main():
    print("=" * 80)
    print("过拟合诊断报告")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"\n【数据概况】")
    print(f"  总样本数: {len(df)}")
    print(f"  功率组数量: {df['激光功率'].nunique()}")
    print(f"  功率组: {df['激光功率'].unique()}")

    print(f"\n【问题1: 目标变量按功率组的唯一性】")
    for power in sorted(df["激光功率"].unique(), key=lambda x: int(x.replace("W", ""))):
        group = df[df["激光功率"] == power]
        unique_hv = group["mh_mean_hv"].unique()
        print(f"  {power}: 样本数={len(group)}, 唯一硬度值={len(unique_hv)}, 值={unique_hv}")
        if len(unique_hv) == 1:
            print(f"    ⚠️  警告: 该功率组所有样本的硬度值完全相同！")

    print(f"\n【问题2: 特征工程分析】")
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

    derived_features = [
        "power_w",
        "hall_petch",
        "heat_input",
    ]

    cross_features = []
    for f in ["熔覆层平均晶粒尺寸(μm)", "析出相/碳化物面积占比(%)", "气孔孔隙率(%)"]:
        col_name = f"power_x_{f}"
        df[col_name] = df["power_w"] * df[f]
        cross_features.append(col_name)

    df["hall_petch"] = 1.0 / np.sqrt(df["熔覆层平均晶粒尺寸(μm)"])
    scan_spacing = 0.05
    df["heat_input"] = df["power_w"] / (df["扫描速度(mm/min)"] / 60) / scan_spacing / 1000

    all_features = base_features + derived_features + cross_features
    valid_features = [f for f in all_features if f in df.columns and df[f].notna().all()]

    print(f"  基础特征数: {len(base_features)}")
    print(f"  衍生特征数: {len(derived_features)}")
    print(f"  交叉特征数: {len(cross_features)}")
    print(f"  总特征数: {len(valid_features)}")
    print(f"  特征/样本比: {len(valid_features)}/{len(df)} = {len(valid_features)/len(df):.3f}")

    print(f"\n【问题3: 功率与硬度的相关性】")
    corr = df["power_w"].corr(df["mh_mean_hv"])
    print(f"  power_w 与 mh_mean_hv 的相关系数: {corr:.4f}")
    if abs(corr) > 0.95:
        print(f"  ⚠️  警告: 功率与硬度几乎完全相关！模型只需记住功率就能预测硬度")

    print(f"\n【问题4: 交叉验证方式对比】")
    target = "mh_mean_hv"
    X = df[valid_features].values
    y = df[target].values
    groups = df["power_w"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )

    print(f"\n  (a) LOO-CV (按单样本留一) - 当前使用的方式")
    loo = LeaveOneOut()
    y_cv_loo = []
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        m.fit(X_train, y_train)
        y_cv_loo.append(m.predict(X_test)[0])
    y_cv_loo = np.array(y_cv_loo)
    r2_loo = r2_score(y, y_cv_loo)
    rmse_loo = np.sqrt(mean_squared_error(y, y_cv_loo))
    print(f"      R² = {r2_loo:.4f}")
    print(f"      RMSE = {rmse_loo:.2f} HV")

    print(f"\n  (b) Leave-One-Group-Out (按功率组留一) - 正确的方式")
    logo = LeaveOneGroupOut()
    y_cv_logo = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        m.fit(X_train, y_train)
        y_cv_logo[test_idx] = m.predict(X_test)
    r2_logo = r2_score(y, y_cv_logo)
    rmse_logo = np.sqrt(mean_squared_error(y, y_cv_logo))
    print(f"      R² = {r2_logo:.4f}")
    print(f"      RMSE = {rmse_logo:.2f} HV")

    print(f"\n  对比:")
    print(f"    LOO-CV R²: {r2_loo:.4f} (虚高，因为训练集已见过相同功率的样本)")
    print(f"    LOGO-CV R²: {r2_logo:.4f} (真实泛化能力)")
    print(f"    差距: {r2_loo - r2_logo:.4f}")

    if r2_loo > 0.9 and r2_logo < 0.3:
        print(f"\n  ❌ 结论: 严重过拟合！LOO-CV结果严重虚高")
        print(f"     原因: 同一功率组有多个样本，但目标值相同")
        print(f"           LOO-CV时，测试样本的功率在训练集中出现过")
        print(f"           模型只需'记住'功率→硬度映射即可，无需学习微观组织规律")

    print(f"\n【问题5: 移除功率相关特征后的表现】")
    features_no_power = [f for f in valid_features if f not in ["power_w", "heat_input"] and not f.startswith("power_x_")]
    print(f"  移除功率相关特征后，特征数: {len(features_no_power)}")

    X_np = df[features_no_power].values
    X_np_scaled = StandardScaler().fit_transform(X_np)

    y_cv_logo_np = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X_np_scaled, y, groups):
        X_train, X_test = X_np_scaled[train_idx], X_np_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        m.fit(X_train, y_train)
        y_cv_logo_np[test_idx] = m.predict(X_test)
    r2_logo_np = r2_score(y, y_cv_logo_np)
    rmse_logo_np = np.sqrt(mean_squared_error(y, y_cv_logo_np))
    print(f"  LOGO-CV R² = {r2_logo_np:.4f}")
    print(f"  LOGO-CV RMSE = {rmse_logo_np:.2f} HV")

    print(f"\n【总结: 过拟合的根本原因】")
    print(f"  1. 目标变量(mh_mean_hv)是按功率组测量的，同组所有样本值相同")
    print(f"  2. LOO-CV按单样本划分，训练集中总能找到相同功率的样本")
    print(f"  3. 特征中包含power_w、heat_input、power_x_*等功率直接相关特征")
    print(f"  4. 模型只需'记住'功率→硬度映射，不需要学习微观组织规律")
    print(f"  5. 因此LOO-CV R²接近1.0，但这是数据泄露导致的虚高")

    print(f"\n【建议的解决方案】")
    print(f"  1. 使用Leave-One-Group-Out (按功率组留一) 替代LOO-CV")
    print(f"  2. 从特征中移除power_w、heat_input、power_x_*等直接功率特征")
    print(f"  3. 按功率组取平均，减少到4个有效样本后建模（样本太少，需谨慎）")
    print(f"  4. 简化模型：减少树深度、减少树数量、增加正则化")
    print(f"  5. 使用更简单的模型（如线性回归）作为基准对比")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
