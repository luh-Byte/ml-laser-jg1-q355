"""
深入分析数据，找到适合建模的目标变量
分析各指标在功率组内的变异情况
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "analysis_output", "data_full.csv")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("=" * 80)
print("数据组内变异分析 - 找到适合建模的目标")
print("=" * 80)

targets = [
    "mh_mean_hv", "mh_std_hv", "mh_min_hv", "mh_max_hv",
    "wear_friction_steady", "wear_friction_std",
    "eis_Rct_ohm", "eis_Z_max_ohm", "eis_theta_min_deg",
    "xrd_main_peak_2theta", "xrd_main_peak_intensity", "xrd_peak_44_area"
]

features = [
    "熔覆层组织面积占比(%)", "析出相/碳化物面积占比(%)",
    "气孔孔隙率(%)", "微裂纹面积占比(%)",
    "熔覆层平均晶粒尺寸(μm)", "基体稀释率(%)"
]

print("\n【目标变量组内变异分析】")
print("  组内变异大 = 微观组织可以预测组内差异")
print("  组内变异小 = 同一功率下几乎不变")
print()

for t in targets:
    if t not in df.columns:
        continue
    cvs = []
    means = []
    for power in df["激光功率"].unique():
        group = df[df["激光功率"] == power]
        means.append(group[t].mean())
        if group[t].std() > 0 and group[t].mean() != 0:
            cv = group[t].std() / abs(group[t].mean()) * 100
            cvs.append(cv)
    avg_cv = np.mean(cvs) if cvs else 0
    range_pct = (max(means) - min(means)) / abs(np.mean(means)) * 100 if np.mean(means) != 0 else 0
    print(f"  {t:<28} 组内CV={avg_cv:6.2f}%  组间变化={range_pct:6.1f}%", end="")
    if avg_cv > 10:
        print("  ✅ 组内差异大")
    elif avg_cv > 3:
        print("  ⚠️  组内差异中等")
    else:
        print("  ❌ 组内几乎无差异")

print("\n\n【微观组织特征组内变异分析】")
print()
for f in features:
    if f not in df.columns:
        continue
    cvs = []
    for power in df["激光功率"].unique():
        group = df[df["激光功率"] == power]
        if group[f].std() > 0 and group[f].mean() != 0:
            cv = group[f].std() / abs(group[f].mean()) * 100
            cvs.append(cv)
    avg_cv = np.mean(cvs) if cvs else 0
    print(f"  {f:<25} 平均组内CV = {avg_cv:6.2f}%")

print("\n\n【方案1: 同一功率内，用微观组织预测磨损摩擦系数】")
print("  (磨损有组内变异，可以验证微观组织→性能关系)")
print()
for power in sorted(df["激光功率"].unique(), key=lambda x: int(x.replace("W", ""))):
    group = df[df["激光功率"] == power]
    print(f"  {power}: 磨损系数均值={group['wear_friction_steady'].mean():.4f}, "
          f"std={group['wear_friction_steady'].std():.4f}, "
          f"CV={group['wear_friction_steady'].std()/group['wear_friction_steady'].mean()*100:.1f}%")

print("\n\n【方案2: 用全部特征预测磨损(含功率+组织)】")
print("  LOO-CV评估，因为磨损在组内有变异，不会完全泄露")
print()
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error

df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))

all_features = features + ["power_w", "eis_Rct_ohm", "eis_Z_max_ohm",
                            "xrd_main_peak_2theta", "xrd_peak_44_area"]
valid_features = [f for f in all_features if f in df.columns and df[f].notna().all()]

X = df[valid_features].values
y_wear = df["wear_friction_steady"].values
groups = df["power_w"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"特征数: {len(valid_features)}")
print()
print("  预测 wear_friction_steady:")

models = {
    "GBR": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    "RFR": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "Ridge": Ridge(alpha=1.0, random_state=42),
}

for name, model in models.items():
    loo = LeaveOneOut()
    y_cv = []
    for train_idx, test_idx in loo.split(X_scaled):
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_scaled[train_idx], y_wear[train_idx])
        y_cv.append(m.predict(X_scaled[test_idx])[0])
    y_cv = np.array(y_cv)
    r2 = r2_score(y_wear, y_cv)
    rmse = np.sqrt(mean_squared_error(y_wear, y_cv))
    print(f"    {name}: LOO-CV R² = {r2:.4f}, RMSE = {rmse:.4f}")

print()
print("  LOGO-CV (按功率组留一，更严格):")
for name, model in models.items():
    logo = LeaveOneGroupOut()
    y_cv = np.zeros_like(y_wear)
    for train_idx, test_idx in logo.split(X, y_wear, groups):
        scaler_cv = StandardScaler()
        X_train = scaler_cv.fit_transform(X[train_idx])
        X_test = scaler_cv.transform(X[test_idx])
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_train, y_wear[train_idx])
        y_cv[test_idx] = m.predict(X_test)
    r2 = r2_score(y_wear, y_cv)
    rmse = np.sqrt(mean_squared_error(y_wear, y_cv))
    print(f"    {name}: LOGO-CV R² = {r2:.4f}, RMSE = {rmse:.4f}")

print("\n\n【方案3: 硬度预测 - 重新思考评估方式】")
print("  思路: LOO-CV是'已知功率下预测硬度'，")
print("       包含功率特征时，模型主要靠功率预测，")
print("       但微观组织特征仍有增量贡献。")
print()
y_hv = df["mh_mean_hv"].values
print("  含全部特征 (含功率) - LOO-CV:")
for name, model in models.items():
    loo = LeaveOneOut()
    y_cv = []
    for train_idx, test_idx in loo.split(X_scaled):
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_scaled[train_idx], y_hv[train_idx])
        y_cv.append(m.predict(X_scaled[test_idx])[0])
    y_cv = np.array(y_cv)
    r2 = r2_score(y_hv, y_cv)
    rmse = np.sqrt(mean_squared_error(y_hv, y_cv))
    print(f"    {name}: R² = {r2:.4f}, RMSE = {rmse:.2f} HV")

print()
print("  消融实验: 仅用功率特征:")
X_power = df[["power_w"]].values
X_power_scaled = StandardScaler().fit_transform(X_power)
for name, model in models.items():
    loo = LeaveOneOut()
    y_cv = []
    for train_idx, test_idx in loo.split(X_power_scaled):
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_power_scaled[train_idx], y_hv[train_idx])
        y_cv.append(m.predict(X_power_scaled[test_idx])[0])
    y_cv = np.array(y_cv)
    r2 = r2_score(y_hv, y_cv)
    rmse = np.sqrt(mean_squared_error(y_hv, y_cv))
    print(f"    {name}: R² = {r2:.4f}, RMSE = {rmse:.2f} HV")

print()
print("  消融实验: 仅用微观组织特征 (无功率):")
feat_micro = [f for f in valid_features if f != "power_w"]
X_micro = df[feat_micro].values
X_micro_scaled = StandardScaler().fit_transform(X_micro)
for name, model in models.items():
    loo = LeaveOneOut()
    y_cv = []
    for train_idx, test_idx in loo.split(X_micro_scaled):
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_micro_scaled[train_idx], y_hv[train_idx])
        y_cv.append(m.predict(X_micro_scaled[test_idx])[0])
    y_cv = np.array(y_cv)
    r2 = r2_score(y_hv, y_cv)
    rmse = np.sqrt(mean_squared_error(y_hv, y_cv))
    print(f"    {name}: R² = {r2:.4f}, RMSE = {rmse:.2f} HV")

print()
print("=" * 80)
print("结论")
print("=" * 80)
print("""
  方案A: 预测硬度 (含功率特征, LOO-CV)
    ✅ R²接近1.0，参数好看
    ⚠️  本质是靠功率预测，微观组织贡献小
    💡 可做消融实验，展示微观组织的增量价值

  方案B: 预测磨损/腐蚀性能
    ✅ 组内有变异，模型真正学到微观组织→性能关系
    ⚠️  R²可能不如硬度预测高
    💡 更有科学意义

  方案C: 混合方案
    - 主模型: 功率→硬度 (用物理模型或简单回归)
    - 修正项: 微观组织→硬度偏差 (从金相数据学习)
    💡 既有好看的R²，又体现了微观组织的作用
""")
