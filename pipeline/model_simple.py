"""
简化ML模型 - 精简特征 + 多输出预测
输入: 功率 + XRD + 磨损 (8个特征)
输出: 硬度、摩擦系数、晶粒尺寸、稀释率 (4个目标)
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 1. 精简特征定义 =====================
# 输入特征：不包含输出目标，避免数据泄露
INPUT_FEATURES = [
    "power_w",                    # 激光功率
    "xrd_main_peak_2theta",       # XRD主峰位置
    "xrd_main_peak_intensity",    # XRD主峰强度
    "xrd_peak_44_area",           # XRD 44°峰面积
    "eis_Rct_ohm",                # EIS电荷转移电阻
    "eis_theta_min_deg",          # EIS相位角
]

# 输出目标：可以同时预测多个性能指标
OUTPUT_TARGETS = [
    "mh_mean_hv",                 # 平均显微硬度
    "mh_cladding_hv",             # 熔覆层硬度
    "mh_substrate_hv",            # 基体硬度
    "wear_friction_steady",       # 稳态摩擦系数
    "熔覆层平均晶粒尺寸(μm)",      # 晶粒尺寸
    "基体稀释率(%)",               # 稀释率
]


# ===================== 2. 数据加载 =====================
def load_data():
    """加载数据"""
    csv_path = os.path.join(OUTPUT_DIR, "data_full.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    return df


def prepare_features(df, features, targets):
    """准备特征和目标"""
    # 过滤有效特征
    valid_features = [f for f in features if f in df.columns and df[f].notna().all()]
    valid_targets = [t for t in targets if t in df.columns and df[t].notna().all()]
    
    X = df[valid_features].values
    y = df[valid_targets].values
    
    return X, y, valid_features, valid_targets


# ===================== 3. 模型训练 =====================
def train_multi_output_model(X, y, feature_names, target_names):
    """训练多输出模型（为每个目标单独训练GBR）"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    results = {}
    loo = LeaveOneOut()
    
    print("\n模型训练结果:")
    print("-" * 70)
    print(f"{'目标':<20} {'训练R²':<10} {'CV R²':<10} {'RMSE':<10} {'MAE':<10}")
    print("-" * 70)
    
    for i, target in enumerate(target_names):
        y_target = y[:, i]
        
        # 训练GBR模型
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        # LOO交叉验证
        y_cv = cross_val_predict(model, X_scaled, y_target, cv=loo)
        
        # 训练最终模型
        model.fit(X_scaled, y_target)
        y_train_pred = model.predict(X_scaled)
        
        # 计算指标
        r2_train = r2_score(y_target, y_train_pred)
        r2_cv = r2_score(y_target, y_cv)
        rmse = np.sqrt(mean_squared_error(y_target, y_cv))
        mae = np.mean(np.abs(y_target - y_cv))
        
        models[target] = model
        results[target] = {
            "r2_train": r2_train,
            "r2_cv": r2_cv,
            "rmse": rmse,
            "mae": mae,
            "y_cv": y_cv,
        }
        
        print(f"{target:<20} {r2_train:<10.3f} {r2_cv:<10.3f} {rmse:<10.2f} {mae:<10.2f}")
    
    return models, scaler, results


# ===================== 4. 预测函数 =====================
def predict(models, scaler, features, input_dict):
    """
    使用模型预测
    
    参数:
        models: 训练好的模型字典
        scaler: 特征缩放器
        features: 特征名列表
        input_dict: 输入参数字典
    
    返回:
        预测结果字典
    """
    # 构建特征向量
    X = np.array([[input_dict.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)
    
    # 预测每个目标
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(X_scaled)[0]
    
    return predictions


# ===================== 5. 可视化 =====================
def plot_results(df, results, feature_names, target_names):
    """绘制预测结果对比图"""
    n = len(target_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = ['#2381c4', '#ff7f0e', '#2baf2b', '#d62728', '#8400ff', '#8c4677']
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        y_true = df[target].values
        y_pred = results[target]["y_cv"]
        
        ax.scatter(y_true, y_pred, c=colors[i % len(colors)], s=50, alpha=0.7,
                   edgecolors='black', linewidth=1)
        
        lims = [min(y_true.min(), y_pred.min()) - 10,
                max(y_true.max(), y_pred.max()) + 10]
        ax.plot(lims, lims, 'k--', linewidth=2)
        
        r2 = results[target]["r2_cv"]
        rmse = results[target]["rmse"]
        ax.text(0.05, 0.95, f'R²={r2:.3f}\nRMSE={rmse:.1f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('实测值', fontsize=10)
        ax.set_ylabel('预测值', fontsize=10)
        ax.set_title(target, fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "simple_model_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] 结果图: {fig_path}")


def plot_feature_importance(models, feature_names, target_names):
    """绘制特征重要性"""
    n = len(target_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        model = models[target]
        
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)
        
        ax.barh(range(len(sorted_idx)), importance[sorted_idx], color='steelblue')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[j] for j in sorted_idx], fontsize=9)
        ax.set_xlabel('重要性', fontsize=10)
        ax.set_title(target, fontsize=11)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "simple_model_importance.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 重要性图: {fig_path}")


# ===================== 6. 主函数 =====================
def main():
    print("=" * 70)
    print("简化ML模型 - 精简特征 + 多输出预测")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    df = load_data()
    print(f"  样本数: {len(df)}")
    
    # 准备特征
    print("\n[2/4] 准备特征...")
    X, y, features, targets = prepare_features(df, INPUT_FEATURES, OUTPUT_TARGETS)
    print(f"  输入特征 ({len(features)}): {features}")
    print(f"  输出目标 ({len(targets)}): {targets}")
    
    # 训练模型
    print("\n[3/4] 训练模型...")
    models, scaler, results = train_multi_output_model(X, y, features, targets)
    
    # 保存模型
    print("\n[4/4] 保存模型...")
    model_path = os.path.join(OUTPUT_DIR, "model_simple.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": models,
            "scaler": scaler,
            "features": features,
            "targets": targets,
            "results": results,
        }, f)
    print(f"  [OK] 模型: {model_path}")
    
    # 可视化
    plot_results(df, results, features, targets)
    plot_feature_importance(models, features, targets)
    
    # 示例预测
    print("\n" + "=" * 70)
    print("预测示例")
    print("=" * 70)
    
    test_inputs = [
        {"power_w": 900, "xrd_main_peak_2theta": 44.7, "xrd_main_peak_intensity": 4972,
         "xrd_peak_44_area": 51520, "eis_Rct_ohm": 1114, "eis_theta_min_deg": -39.1},
        {"power_w": 1200, "xrd_main_peak_2theta": 44.8, "xrd_main_peak_intensity": 4745,
         "xrd_peak_44_area": 49800, "eis_Rct_ohm": 2647, "eis_theta_min_deg": -45.2},
        {"power_w": 1500, "xrd_main_peak_2theta": 43.7, "xrd_main_peak_intensity": 3495,
         "xrd_peak_44_area": 38500, "eis_Rct_ohm": 4464, "eis_theta_min_deg": -56.2},
        {"power_w": 1800, "xrd_main_peak_2theta": 43.6, "xrd_main_peak_intensity": 4058,
         "xrd_peak_44_area": 42100, "eis_Rct_ohm": 2941, "eis_theta_min_deg": -52.1},
    ]
    
    print(f"\n{'功率':<8} {'平均硬度':<12} {'熔覆层硬度':<12} {'基体硬度':<12} {'摩擦系数':<12} {'晶粒尺寸':<12} {'稀释率':<10}")
    print("-" * 80)
    
    for inp in test_inputs:
        pred = predict(models, scaler, features, inp)
        print(f"{inp['power_w']:<8} {pred['mh_mean_hv']:<12.1f} "
              f"{pred['mh_cladding_hv']:<12.1f} {pred['mh_substrate_hv']:<12.1f} "
              f"{pred['wear_friction_steady']:<12.3f} "
              f"{pred.get('熔覆层平均晶粒尺寸(μm)', 0):<12.1f} "
              f"{pred.get('基体稀释率(%)', 0):<10.1f}")
    
    print("\n" + "=" * 70)
    print("模型训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
