"""
功率预测模型
正向：给定功率 → 预测硬度等效果
反向：给定目标硬度 → 预测需要的功率
"""

import os
import sys
import io
import pickle
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize_scalar

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


# ===================== 1. 数据加载与准备 =====================
def load_data():
    """加载数据"""
    csv_path = os.path.join(OUTPUT_DIR, "data_full.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    return df


def prepare_power_model(df):
    """准备功率预测模型的数据"""
    # 输入：功率
    X = df[["power_w"]].values
    
    # 输出：多个效果指标
    targets = {
        "mh_mean_hv": df["mh_mean_hv"].values,
        "mh_cladding_hv": df["mh_cladding_hv"].values,
        "mh_substrate_hv": df["mh_substrate_hv"].values,
        "wear_friction_steady": df["wear_friction_steady"].values,
        "熔覆层平均晶粒尺寸(μm)": df["熔覆层平均晶粒尺寸(μm)"].values,
        "基体稀释率(%)": df["基体稀释率(%)"].values,
    }
    
    return X, targets


# ===================== 2. 训练正向模型 =====================
def train_forward_models(X, targets):
    """训练正向预测模型（功率 → 效果）"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    results = {}
    loo = LeaveOneOut()
    
    print("\n正向模型训练结果（功率 → 效果）:")
    print("-" * 70)
    print(f"{'目标':<25} {'训练R²':<10} {'CV R²':<10} {'RMSE':<10}")
    print("-" * 70)
    
    for name, y in targets.items():
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        
        # LOO交叉验证
        y_cv = cross_val_predict(model, X_scaled, y, cv=loo)
        
        # 训练最终模型
        model.fit(X_scaled, y)
        y_train_pred = model.predict(X_scaled)
        
        r2_train = r2_score(y, y_train_pred)
        r2_cv = r2_score(y, y_cv)
        rmse = np.sqrt(mean_squared_error(y, y_cv))
        
        models[name] = model
        results[name] = {"r2_train": r2_train, "r2_cv": r2_cv, "rmse": rmse}
        
        print(f"{name:<25} {r2_train:<10.3f} {r2_cv:<10.3f} {rmse:<10.2f}")
    
    return models, scaler, results


# ===================== 3. 正向预测 =====================
def predict_forward(models, scaler, power):
    """正向预测：给定功率 → 预测效果"""
    X = np.array([[power]])
    X_scaled = scaler.transform(X)
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_scaled)[0]
    
    return predictions


# ===================== 4. 反向预测（核心功能） =====================
def predict_power_for_target(models, scaler, target_name, target_value, 
                              power_range=(600, 2000)):
    """
    反向预测：给定目标效果 → 预测需要的功率
    
    参数:
        models: 训练好的模型
        scaler: 特征缩放器
        target_name: 目标效果名称（如 'mh_mean_hv'）
        target_value: 目标值（如 300 HV）
        power_range: 功率搜索范围
    
    返回:
        需要的功率值
    """
    model = models[target_name]
    
    # 定义目标函数：最小化预测值与目标值的差距
    def objective(power):
        X = np.array([[power]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return (pred - target_value) ** 2
    
    # 在功率范围内搜索最优解
    result = minimize_scalar(objective, bounds=power_range, method='bounded')
    
    optimal_power = result.x
    # 验证预测值
    X = np.array([[optimal_power]])
    X_scaled = scaler.transform(X)
    predicted_value = model.predict(X_scaled)[0]
    
    return {
        "optimal_power": optimal_power,
        "predicted_value": predicted_value,
        "target_value": target_value,
        "error": abs(predicted_value - target_value),
    }


def find_optimal_power(models, scaler, hardness_target, friction_target=None):
    """
    综合优化：找到满足多个目标的最优功率
    
    参数:
        hardness_target: 目标硬度 (HV)
        friction_target: 目标摩擦系数 (可选)
    
    返回:
        推荐功率和预测效果
    """
    # 先用硬度目标找到功率范围
    result_h = predict_power_for_target(models, scaler, "mh_mean_hv", hardness_target)
    base_power = result_h["optimal_power"]
    
    # 在基础功率附近搜索最优解
    best_power = base_power
    best_score = float('inf')
    
    for power in np.arange(max(600, base_power - 200), min(2000, base_power + 200), 10):
        preds = predict_forward(models, scaler, power)
        
        # 计算综合评分（硬度偏差 + 摩擦系数偏差）
        hardness_error = abs(preds["mh_mean_hv"] - hardness_target) / hardness_target
        
        if friction_target:
            friction_error = abs(preds["wear_friction_steady"] - friction_target) / friction_target
            score = hardness_error + 0.3 * friction_error  # 摩擦系数权重较低
        else:
            score = hardness_error
        
        if score < best_score:
            best_score = score
            best_power = power
    
    # 返回最优结果
    optimal_preds = predict_forward(models, scaler, best_power)
    
    return {
        "optimal_power": best_power,
        "predictions": optimal_preds,
        "score": best_score,
    }


# ===================== 5. 可视化 =====================
def plot_power_effect_curves(df, models, scaler):
    """绘制功率-效果曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    power_range = np.arange(600, 2000, 10)
    colors = ['#2381c4', '#ff7f0e', '#2baf2b', '#d62728', '#8400ff', '#8c4677']
    
    target_labels = {
        "mh_mean_hv": "平均硬度 (HV)",
        "mh_cladding_hv": "熔覆层硬度 (HV)",
        "mh_substrate_hv": "基体硬度 (HV)",
        "wear_friction_steady": "摩擦系数",
        "熔覆层平均晶粒尺寸(μm)": "晶粒尺寸 (μm)",
        "基体稀释率(%)": "稀释率 (%)",
    }
    
    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        
        # 预测曲线
        X_pred = power_range.reshape(-1, 1)
        X_scaled = scaler.transform(X_pred)
        y_pred = model.predict(X_scaled)
        
        ax.plot(power_range, y_pred, color=colors[i], linewidth=2, label='预测曲线')
        
        # 实际数据点（使用正确的列名）
        col_name = name if name in df.columns else name
        ax.scatter(df["power_w"], df[col_name], color='black', s=50, alpha=0.7, label='实测数据')
        
        ax.set_xlabel('激光功率 (W)', fontsize=11)
        ax.set_ylabel(target_labels.get(name, name), fontsize=11)
        ax.set_title(f'功率 vs {target_labels.get(name, name)}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "power_effect_curves.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 功率-效果曲线: {fig_path}")


def plot_inverse_prediction(df, models, scaler, target_hardness=300):
    """绘制反向预测结果"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 功率-硬度曲线
    power_range = np.arange(600, 2000, 10)
    X_pred = power_range.reshape(-1, 1)
    X_scaled = scaler.transform(X_pred)
    hardness_pred = models["mh_mean_hv"].predict(X_scaled)
    
    ax.plot(power_range, hardness_pred, 'b-', linewidth=2, label='预测硬度曲线')
    ax.scatter(df["power_w"], df["mh_mean_hv"], color='black', s=80, alpha=0.7, label='实测数据')
    
    # 目标硬度线
    ax.axhline(y=target_hardness, color='red', linestyle='--', linewidth=2, 
               label=f'目标硬度: {target_hardness} HV')
    
    # 找到交叉点
    result = predict_power_for_target(models, scaler, "mh_mean_hv", target_hardness)
    ax.axvline(x=result["optimal_power"], color='green', linestyle=':', linewidth=2,
               label=f'推荐功率: {result["optimal_power"]:.0f} W')
    
    ax.scatter([result["optimal_power"]], [target_hardness], color='green', s=200, 
               marker='*', zorder=5, label='推荐点')
    
    ax.set_xlabel('激光功率 (W)', fontsize=12)
    ax.set_ylabel('平均硬度 (HV)', fontsize=12)
    ax.set_title(f'反向预测: 目标硬度 {target_hardness} HV 所需功率', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "inverse_prediction.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 反向预测图: {fig_path}")


# ===================== 6. 参数评估 =====================
def evaluate_parameters(predictions):
    """评估参数好坏"""
    score = 100
    suggestions = []
    
    # 硬度评估
    hardness = predictions["mh_mean_hv"]
    if hardness < 200:
        score -= 30
        suggestions.append("⚠️ 硬度过低 (<200 HV)，建议提高功率")
    elif hardness > 450:
        score -= 20
        suggestions.append("⚠️ 硬度过高 (>450 HV)，可能导致脆性")
    else:
        suggestions.append("✅ 硬度在合理范围 (200-450 HV)")
    
    # 摩擦系数评估
    friction = predictions["wear_friction_steady"]
    if friction > 0.3:
        score -= 15
        suggestions.append("⚠️ 摩擦系数偏高 (>0.3)，耐磨性可能不足")
    else:
        suggestions.append("✅ 摩擦系数在合理范围 (<0.3)")
    
    # 晶粒尺寸评估
    grain = predictions.get("熔覆层平均晶粒尺寸(μm)", 0)
    if grain > 30:
        score -= 10
        suggestions.append("⚠️ 晶粒尺寸偏大 (>30μm)，可能影响强度")
    else:
        suggestions.append("✅ 晶粒尺寸在合理范围 (<30μm)")
    
    # 稀释率评估
    dilution = predictions.get("基体稀释率(%)", 0)
    if dilution < 30:
        score -= 20
        suggestions.append("⚠️ 稀释率过低 (<30%)，结合可能不良")
    elif dilution > 70:
        score -= 15
        suggestions.append("⚠️ 稀释率过高 (>70%)，基体稀释严重")
    else:
        suggestions.append("✅ 稀释率在理想范围 (30-70%)")
    
    return max(0, score), suggestions


# ===================== 7. 主函数 =====================
def main():
    print("=" * 70)
    print("激光熔覆功率预测模型")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    df = load_data()
    print(f"  样本数: {len(df)}")
    print(f"  功率范围: {df['power_w'].min():.0f} - {df['power_w'].max():.0f} W")
    
    # 准备数据
    print("\n[2/5] 准备数据...")
    X, targets = prepare_power_model(df)
    print(f"  输入: 功率 (1个特征)")
    print(f"  输出: {list(targets.keys())}")
    
    # 训练模型
    print("\n[3/5] 训练正向模型...")
    models, scaler, results = train_forward_models(X, targets)
    
    # 保存模型
    print("\n[4/5] 保存模型...")
    model_path = os.path.join(OUTPUT_DIR, "model_power_predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": models,
            "scaler": scaler,
            "results": results,
        }, f)
    print(f"  [OK] 模型: {model_path}")
    
    # 可视化
    print("\n[5/5] 生成图表...")
    plot_power_effect_curves(df, models, scaler)
    plot_inverse_prediction(df, models, scaler, target_hardness=300)
    
    # 示例：反向预测
    print("\n" + "=" * 70)
    print("反向预测示例：目标硬度 → 所需功率")
    print("=" * 70)
    
    test_targets = [250, 300, 350, 400]
    
    print(f"\n{'目标硬度(HV)':<15} {'推荐功率(W)':<15} {'实际预测(HV)':<15} {'误差(HV)':<10}")
    print("-" * 60)
    
    for target in test_targets:
        result = predict_power_for_target(models, scaler, "mh_mean_hv", target)
        print(f"{target:<15} {result['optimal_power']:<15.0f} "
              f"{result['predicted_value']:<15.1f} {result['error']:<10.1f}")
    
    # 示例：参数评估
    print("\n" + "=" * 70)
    print("参数评估示例")
    print("=" * 70)
    
    for power in [900, 1200, 1500, 1800]:
        preds = predict_forward(models, scaler, power)
        score, suggestions = evaluate_parameters(preds)
        
        print(f"\n功率: {power} W")
        print(f"  预测硬度: {preds['mh_mean_hv']:.1f} HV")
        print(f"  评分: {score}/100")
        for s in suggestions:
            print(f"  {s}")
    
    print("\n" + "=" * 70)
    print("模型训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
