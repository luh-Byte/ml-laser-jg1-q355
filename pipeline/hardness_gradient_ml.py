"""
显微硬度梯度ML训练模块
基于梯度硬度数据训练预测模型
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


# ===================== 1. 加载梯度硬度数据 =====================
def load_gradient_data():
    """加载梯度硬度ML训练数据"""
    csv_path = os.path.join(OUTPUT_DIR, "梯度硬度ML训练数据.csv")
    if not os.path.exists(csv_path):
        print(f"  [ERROR] 梯度硬度ML训练数据.csv 不存在，请先运行 hardness_gradient.py")
        return None
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  加载数据: {len(df)} 条记录")
    print(f"  功率水平: {sorted(df['激光功率(W)'].unique())}")
    print(f"  位置范围: {df['测量位置'].min()}-{df['测量位置'].max()}")
    
    return df


def load_integrated_data():
    """加载整合实验数据（包含梯度统计）"""
    csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    if not os.path.exists(csv_path):
        print(f"  [ERROR] 完整实验数据汇总.csv 不存在")
        return None
    
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


# ===================== 2. 特征工程 =====================
def prepare_gradient_features(df):
    """
    为梯度硬度训练准备特征
    
    特征:
    - 激光功率(W)
    - 测量位置 (1-10)
    - 位置归一化 (0-1)
    - 区域标签 (0=基体, 1=熔覆层)
    """
    features = ['激光功率(W)', '测量位置', '位置归一化', '区域标签']
    
    # 添加功率×位置交叉特征
    df['power_x_position'] = df['激光功率(W)'] * df['测量位置']
    features.append('power_x_position')
    
    # 添加功率×区域交叉特征
    df['power_x_region'] = df['激光功率(W)'] * df['区域标签']
    features.append('power_x_region')
    
    # 验证特征
    valid_features = [f for f in features if f in df.columns and df[f].notna().all()]
    
    return df, valid_features


def prepare_integrated_features(df):
    """
    为整合数据训练准备特征
    使用梯度统计信息
    """
    features = ['power_w']
    
    # 添加金相特征
    metallography_features = [
        "熔覆层组织面积占比(%)",
        "析出相/碳化物面积占比(%)",
        "气孔孔隙率(%)",
        "微裂纹面积占比(%)",
        "熔覆层平均晶粒尺寸(μm)",
        "基体稀释率(%)",
    ]
    for f in metallography_features:
        if f in df.columns:
            features.append(f)
    
    # 添加梯度硬度特征
    gradient_features = [
        "mh_cladding_hv",
        "mh_substrate_hv",
        "mh_gradient_range",
    ]
    for f in gradient_features:
        if f in df.columns:
            features.append(f)
    
    # 添加交叉特征
    if "mh_cladding_hv" in df.columns and "mh_substrate_hv" in df.columns:
        df["hardness_ratio"] = df["mh_cladding_hv"] / df["mh_substrate_hv"].clip(lower=1)
        features.append("hardness_ratio")
    
    # 添加Hall-Petch项
    if "熔覆层平均晶粒尺寸(μm)" in df.columns:
        df["hall_petch"] = 1.0 / np.sqrt(df["熔覆层平均晶粒尺寸(μm)"])
        features.append("hall_petch")
    
    # 添加热输入
    scan_spacing = 0.05  # mm
    if "扫描速度(mm/min)" in df.columns:
        df["heat_input"] = df["power_w"] / (df["扫描速度(mm/min)"] / 60) / scan_spacing / 1000
    else:
        scan_speed = 600  # mm/min (实际值: 10 mm/s = 600 mm/min)
        df["heat_input"] = df["power_w"] / (scan_speed / 60) / scan_spacing / 1000
    features.append("heat_input")
    
    valid_features = [f for f in features if f in df.columns and df[f].notna().all()]
    
    return df, valid_features


# ===================== 3. 模型训练 =====================
def train_gradient_models(df, features, target="显微硬度(HV)"):
    """训练梯度硬度预测模型"""
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
    
    print("\n模型训练结果:")
    print("-" * 60)
    
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
    
    return results, X, y, X_scaled


def train_integrated_models(df, features, target="mh_mean_hv"):
    """训练整合数据模型（使用梯度统计特征）"""
    X = df[features].values
    y = df[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n整合数据模型:")
    print(f"  GBR: R²={r2:.4f}, RMSE={rmse:.2f} HV")
    
    # 特征重要性
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])
        print("\n  特征重要性:")
        for name, imp in feat_imp[:10]:
            print(f"    {name}: {imp:.4f}")
    
    return model, scaler, features


# ===================== 4. 预测函数 =====================
def predict_hardness_at_position(model, scaler, features, power_w, position, region_label):
    """
    预测指定功率和位置的硬度
    
    参数:
        model: 训练好的模型
        scaler: 特征缩放器
        features: 特征列表
        power_w: 激光功率(W)
        position: 测量位置 (1-10)
        region_label: 区域标签 (0=基体, 1=熔覆层)
    
    返回:
        预测硬度值(HV)
    """
    position_normalized = (position - 1) / 9.0
    power_x_position = power_w * position
    power_x_region = power_w * region_label
    
    # 构建特征向量
    feature_dict = {
        '激光功率(W)': power_w,
        '测量位置': position,
        '位置归一化': position_normalized,
        '区域标签': region_label,
        'power_x_position': power_x_position,
        'power_x_region': power_x_region,
    }
    
    X = np.array([[feature_dict.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)
    
    return model.predict(X_scaled)[0]


def predict_hardness_gradient(model, scaler, features, power_w):
    """
    预测指定功率的完整硬度梯度
    
    返回:
        list: 10个位置的预测硬度值
    """
    predictions = []
    for pos in range(1, 11):
        region_label = 1 if pos <= 5 else 0
        pred = predict_hardness_at_position(model, scaler, features, power_w, pos, region_label)
        predictions.append(pred)
    return predictions


# ===================== 5. 可视化 =====================
def plot_gradient_comparison(df, results, features):
    """绘制梯度硬度对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    powers = sorted(df['激光功率(W)'].unique())
    colors = {900: '#2381c4', 1200: '#ff7f0e', 1500: '#2baf2b', 1800: '#d62728'}
    
    # (a) 实测vs预测散点图
    ax = axes[0, 0]
    best_model = max(results.keys(), key=lambda k: results[k]['r2_cv'])
    y_cv = results[best_model]['y_cv']
    y_true = df['显微硬度(HV)'].values
    
    ax.scatter(y_true, y_cv, c=[colors.get(p, 'gray') for p in df['激光功率(W)']], 
               s=50, alpha=0.7, edgecolors='black', linewidth=1)
    lims = [min(y_true.min(), y_cv.min()) - 20, max(y_true.max(), y_cv.max()) + 20]
    ax.plot(lims, lims, 'k--', linewidth=2, label='y=x')
    ax.set_xlabel('实测硬度 (HV)')
    ax.set_ylabel('预测硬度 (HV)')
    ax.set_title(f'(a) 实测vs预测 ({best_model}, R²={results[best_model]["r2_cv"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) 硬度梯度曲线
    ax = axes[0, 1]
    for power in powers:
        power_data = df[df['激光功率(W)'] == power]
        positions = power_data['测量位置'].values
        hv_values = power_data['显微硬度(HV)'].values
        
        # 实测值
        ax.plot(positions, hv_values, 'o-', color=colors[power], 
                label=f'{power}W 实测', markersize=5, linewidth=1.5)
        
        # 预测值
        pred_values = results[best_model]['y_cv'][df['激光功率(W)'] == power]
        ax.plot(positions, pred_values, 's--', color=colors[power], 
                alpha=0.6, label=f'{power}W 预测', markersize=4)
    
    ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5, label='界面')
    ax.set_xlabel('测量位置')
    ax.set_ylabel('硬度 (HV)')
    ax.set_title('(b) 硬度梯度曲线')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # (c) 模型对比
    ax = axes[1, 0]
    model_names = list(results.keys())
    r2_values = [results[m]['r2_cv'] for m in model_names]
    rmse_values = [results[m]['rmse_cv'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, r2_values, width=0.4, label='R²', color='steelblue')
    ax.bar(x_pos + 0.4, [r/500 for r in rmse_values], width=0.4, label='RMSE/500', color='coral')
    
    ax.set_xticks(x_pos + 0.2)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('值')
    ax.set_title('(c) 模型性能对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # (d) 残差分布
    ax = axes[1, 1]
    for name, r in results.items():
        residuals = y_true - r['y_cv']
        ax.hist(residuals, bins=10, alpha=0.5, label=f'{name} (MAE={r["mae_cv"]:.1f})')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel('残差 (HV)')
    ax.set_ylabel('频数')
    ax.set_title('(d) 残差分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "hardness_gradient_ml.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 图表: {fig_path}")


def plot_predicted_gradient(model, scaler, features):
    """绘制预测的硬度梯度"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    powers = [900, 1200, 1500, 1800]
    colors = {900: '#2381c4', 1200: '#ff7f0e', 1500: '#2baf2b', 1800: '#d62728'}
    
    positions = np.arange(1, 11)
    
    for power in powers:
        predictions = predict_hardness_gradient(model, scaler, features, power)
        ax.plot(positions, predictions, 'o-', color=colors[power], 
                label=f'{power}W', markersize=6, linewidth=2)
    
    ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5, label='界面')
    ax.fill_between([0.5, 5.5], 0, 600, alpha=0.1, color='blue', label='熔覆层')
    ax.fill_between([5.5, 10.5], 0, 600, alpha=0.1, color='green', label='基体')
    
    ax.set_xlabel('测量位置', fontsize=12)
    ax.set_ylabel('预测硬度 (HV)', fontsize=12)
    ax.set_title('各功率预测硬度梯度分布', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{i:02d}' for i in positions])
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(100, 550)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "predicted_hardness_gradient.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 预测梯度图: {fig_path}")


# ===================== 6. 主流程 =====================
def main():
    print("=" * 60)
    print("显微硬度梯度ML训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载梯度硬度数据...")
    df_gradient = load_gradient_data()
    if df_gradient is None:
        return
    
    print("\n[2/5] 准备特征...")
    df_gradient, features_grad = prepare_gradient_features(df_gradient)
    print(f"  梯度特征: {features_grad}")
    
    # 2. 训练梯度模型
    print("\n[3/5] 训练梯度硬度模型...")
    results, X, y, X_scaled = train_gradient_models(df_gradient, features_grad)
    
    # 3. 保存模型
    print("\n[4/5] 保存模型...")
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2_cv'])
    best_result = results[best_model_name]
    
    model_path = os.path.join(OUTPUT_DIR, "hardness_gradient_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_result["model"],
            "scaler": best_result["scaler"],
            "features": features_grad,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        }, f)
    print(f"  [OK] 模型: {model_path}")
    
    # 4. 可视化
    print("\n[5/5] 生成可视化...")
    plot_gradient_comparison(df_gradient, results, features_grad)
    plot_predicted_gradient(best_result["model"], best_result["scaler"], features_grad)
    
    # 5. 预测示例
    print("\n预测示例:")
    print("-" * 60)
    for power in [900, 1200, 1500, 1800]:
        predictions = predict_hardness_gradient(
            best_result["model"], best_result["scaler"], features_grad, power
        )
        cladding_avg = np.mean(predictions[:5])
        substrate_avg = np.mean(predictions[5:])
        print(f"  {power}W: 熔覆层平均={cladding_avg:.1f} HV, 基体平均={substrate_avg:.1f} HV, "
              f"梯度差={cladding_avg - substrate_avg:.1f} HV")
    
    print("\n" + "=" * 60)
    print("梯度硬度ML训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
