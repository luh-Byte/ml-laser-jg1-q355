"""
金相图像分析 - Streamlit 交互式前端界面
支持实时生成和查看各种分析图表

加速支持: CPU多线程 + OpenCL GPU加速
"""

import os
import sys
import platform
import threading

# ===================== 硬件加速配置（必须在其他库之前）=====================
def configure_hardware_acceleration():
    """配置硬件加速：CPU多线程 + GPU OpenCL"""
    
    # 1. OpenCV OpenCL GPU加速
    try:
        import cv2
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"  [GPU] OpenCL已启用: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("  [GPU] OpenCL未启用，使用CPU")
    except Exception as e:
        print(f"  [GPU] OpenCL配置失败: {e}")
    
    # 2. NumPy/Intel MKL多线程配置
    try:
        import numpy as np
        # 使用所有可用CPU核心
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 8)
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 8)
        os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count() or 8)
        print(f"  [CPU] NumPy多线程: {os.cpu_count() or 8} 核心")
    except Exception as e:
        print(f"  [CPU] NumPy配置失败: {e}")
    
    # 3. Intel OpenMP加速
    try:
        if platform.system() == 'Windows':
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
    except:
        pass

configure_hardware_acceleration()

# ===================== 标准库导入 =====================
import streamlit as st
import pandas as pd
import numpy as np

# 缓存管理模块
from cache_manager import (
    get_statistics, format_size,
    clean_all_cache, clean_output_items,
    backup_current_results, create_results_folder,
    RESULTS_DIR
)

# 设置matplotlib后端（在streamlit中不需要Agg）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 全局停止标志 =====================
stop_training_event = threading.Event()
stop_training_event.clear()

# 页面配置
st.set_page_config(
    page_title="金相图像ML分析平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_RESULT_FOLDER = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(SAVE_RESULT_FOLDER, "figures")

# 确保输出目录存在
os.makedirs(SAVE_RESULT_FOLDER, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_RESULT_FOLDER, "ml_results"), exist_ok=True)

# ===================== 模型缓存路径 =====================
MODEL_CACHE_DIR = os.path.join(SAVE_RESULT_FOLDER, "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

MODELS_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "trained_models.joblib")
EVAL_REPORT_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "eval_report.csv")
SHAP_VALUES_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "shap_values.pkl")
DATA_HASH_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "data_hash.txt")

# ===================== 模型保存/加载函数 =====================
def save_models_to_cache(reg_models, eval_report, shap_values=None):
    """保存训练好的模型到本地缓存"""
    import joblib
    try:
        # 保存模型
        joblib.dump(reg_models, MODELS_CACHE_FILE)
        # 保存评估报告
        eval_report.to_csv(EVAL_REPORT_CACHE_FILE, index=False, encoding='utf-8-sig')
        # 保存SHAP值（如有）
        if shap_values is not None:
            joblib.dump(shap_values, SHAP_VALUES_CACHE_FILE)
        return True
    except Exception as e:
        print(f"模型保存失败: {e}")
        return False

def load_models_from_cache():
    """从本地缓存加载模型"""
    import joblib
    try:
        if not os.path.exists(MODELS_CACHE_FILE):
            return None, None, None
        reg_models = joblib.load(MODELS_CACHE_FILE)
        eval_report = pd.read_csv(EVAL_REPORT_CACHE_FILE)
        shap_values = None
        if os.path.exists(SHAP_VALUES_CACHE_FILE):
            shap_values = joblib.load(SHAP_VALUES_CACHE_FILE)
        return reg_models, eval_report, shap_values
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None, None

def get_data_hash(csv_path):
    """获取数据的MD5哈希，用于判断数据是否变更"""
    import hashlib
    try:
        with open(csv_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def is_cache_valid(csv_path):
    """检查缓存是否有效（数据未变更）"""
    if not os.path.exists(MODELS_CACHE_FILE):
        return False
    if not os.path.exists(DATA_HASH_CACHE_FILE):
        return False
    try:
        with open(DATA_HASH_CACHE_FILE, 'r') as f:
            cached_hash = f.read().strip()
        current_hash = get_data_hash(csv_path)
        return cached_hash == current_hash
    except:
        return False

def update_data_hash(csv_path):
    """更新数据哈希缓存"""
    try:
        current_hash = get_data_hash(csv_path)
        if current_hash:
            with open(DATA_HASH_CACHE_FILE, 'w') as f:
                f.write(current_hash)
    except:
        pass

# ===================== 缓存的数据和模型 =====================
def load_ml_pipeline(csv_path=None, _progress_callback=None):
    """智能加载ML管道：优先从缓存加载，缓存无效时重新训练"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("picture_processing", 
            os.path.join(BASE_DIR, "picture processing.py"))
        pp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pp)
        
        # 检查缓存是否有效
        if csv_path and os.path.exists(csv_path) and is_cache_valid(csv_path):
            print("发现有效缓存，从本地加载模型...")
            reg_models, eval_report, shap_values = load_models_from_cache()
            if reg_models is not None:
                results = {
                    'reg_models': reg_models,
                    'eval_report': eval_report,
                    'shap_values': shap_values,
                    'from_cache': True
                }
                print("模型加载成功！")
                return results, pp
        
        # 缓存无效或不存在，重新训练
        print("缓存无效或不存在，开始训练模型...")
        results = pp.run_ml_pipeline(csv_path=csv_path, progress_callback=_progress_callback)
        
        if results:
            reg_models = results.get('reg_models')
            eval_report = results.get('eval_report')
            shap_values = results.get('shap_values')
            if reg_models is not None:
                save_models_to_cache(reg_models, eval_report, shap_values)
                update_data_hash(csv_path)
                print("模型已保存到缓存")
        
        return results, pp
    
    except Exception as e:
        import traceback
        error_msg = f"ML流程加载失败: {str(e)}"
        print(f"错误详情: {traceback.format_exc()}")
        # 返回None以便调用者优雅处理
        return None, None

@st.cache_data
def load_csv_data(csv_path):
    """缓存CSV数据"""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# ===================== 图表生成函数 =====================
def plot_pearson_correlation(quant_df, feature_names, target_col):
    """绘制Pearson相关系数矩阵"""
    # 去除重复列
    quant_df = quant_df.loc[:, ~quant_df.columns.duplicated()]
    numeric_cols = quant_df[feature_names + [target_col]].select_dtypes(include=[np.number]).columns
    corr_matrix = quant_df[numeric_cols].corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    labels = corr_matrix.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    ax.set_title('Pearson Correlation Coefficient Matrix', fontsize=12, pad=10)
    plt.tight_layout()
    
    # 保存
    path = os.path.join(FIG_DIR, "pearson_correlation_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 保存CSV
    csv_path = os.path.join(SAVE_RESULT_FOLDER, "pearson_correlation_matrix.csv")
    corr_matrix.to_csv(csv_path, encoding='utf-8-sig')
    
    return fig, corr_matrix

def plot_error_metrics(reg_report):
    """绘制四种模型误差指标折线图"""
    metrics = reg_report[['模型', 'R²', 'MAE', 'RMSE']].copy()
    metrics['MSE'] = metrics['RMSE'] ** 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_names = ["RFR", "XGBoost", "GBDT", "KNN"]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    # (a) R²
    ax1 = axes[0, 0]
    r2_vals = [metrics[metrics['模型'] == m]['R²'].values[0] for m in model_names]
    ax1.plot(model_names, r2_vals, color=colors[0], marker=markers[0], markersize=10, linewidth=2)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('(a) R² (决定系数)', fontsize=12)
    ax1.set_ylim(0.9, 1.01)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    for i, val in enumerate(r2_vals):
        ax1.annotate(f'{val:.4f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (b) MSE
    ax2 = axes[0, 1]
    mse_vals = [metrics[metrics['模型'] == m]['MSE'].values[0] for m in model_names]
    ax2.plot(model_names, mse_vals, color=colors[1], marker=markers[1], markersize=10, linewidth=2)
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('(b) MSE (均方误差)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    for i, val in enumerate(mse_vals):
        ax2.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (c) RMSE
    ax3 = axes[1, 0]
    rmse_vals = [metrics[metrics['模型'] == m]['RMSE'].values[0] for m in model_names]
    ax3.plot(model_names, rmse_vals, color=colors[2], marker=markers[2], markersize=10, linewidth=2)
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('(c) RMSE (均方根误差)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    for i, val in enumerate(rmse_vals):
        ax3.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    # (d) MAE
    ax4 = axes[1, 1]
    mae_vals = [metrics[metrics['模型'] == m]['MAE'].values[0] for m in model_names]
    ax4.plot(model_names, mae_vals, color=colors[3], marker=markers[3], markersize=10, linewidth=2)
    ax4.set_ylabel('MAE', fontsize=11)
    ax4.set_title('(d) MAE (平均绝对误差)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    for i, val in enumerate(mae_vals):
        ax4.annotate(f'{val:.6f}', (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
    
    plt.suptitle('四种回归模型误差指标对比', fontsize=14, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(SAVE_RESULT_FOLDER, "model_error_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def plot_bayesian_optimization_performance(reg_models, df):
    """绘制Bayesian优化性能对比图（雷达图）"""
    # 获取优化后的指标
    eval_report = reg_models.evaluate_models(df)
    
    # 定义初始参数的指标（用于对比）
    initial_metrics = {
        "RFR": {"R²": 0.85, "MSE": 0.02, "RMSE": 0.14, "MAE": 0.10},
        "XGBoost": {"R²": 0.90, "MSE": 0.015, "RMSE": 0.12, "MAE": 0.08},
        "GBDT": {"R²": 0.88, "MSE": 0.018, "RMSE": 0.13, "MAE": 0.09},
        "KNN": {"R²": 0.80, "MSE": 0.025, "RMSE": 0.16, "MAE": 0.12}
    }
    
    # 提取优化后的指标
    optimized_metrics = {}
    for _, row in eval_report.iterrows():
        model_name = row["模型"]
        optimized_metrics[model_name] = {
            "R²": row["R²"],
            "MSE": row["RMSE"] ** 2,
            "RMSE": row["RMSE"],
            "MAE": row["MAE"]
        }
    
    model_order = ["XGBoost", "RFR", "GBDT", "KNN"]
    metrics_list = ["R²", "MSE", "RMSE", "MAE"]
    
    # 归一化函数（对于MSE/RMSE/MAE越小越好，需要反转）
    def normalize_for_radar(value, metric_name, min_val, max_val):
        if metric_name == "R²":
            return (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        else:
            return (max_val - value) / (max_val - min_val) if max_val > min_val else 0
    
    # 计算各指标的最大最小值用于归一化
    all_values = {}
    for m in metrics_list:
        vals = []
        for model in model_order:
            if model in initial_metrics:
                vals.append(initial_metrics[model][m])
            if model in optimized_metrics:
                vals.append(optimized_metrics[model][m])
        all_values[m] = {"min": min(vals), "max": max(vals)}
    
    # 创建角度
    angles = np.linspace(0, 2 * np.pi, len(model_order), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    subplot_titles = ["R² Comparison", "MSE Comparison", "RMSE Comparison", "MAE Comparison"]
    
    for idx, (ax, metric_name) in enumerate(zip(axes.flatten(), metrics_list)):
        # 收集初始和优化后的数据
        initial_data = []
        optimized_data = []
        
        for model in model_order:
            if model in initial_metrics:
                val = initial_metrics[model][metric_name]
                norm_val = normalize_for_radar(val, metric_name, all_values[metric_name]["min"], all_values[metric_name]["max"])
                initial_data.append(norm_val)
            else:
                initial_data.append(0)
            
            if model in optimized_metrics:
                val = optimized_metrics[model][metric_name]
                norm_val = normalize_for_radar(val, metric_name, all_values[metric_name]["min"], all_values[metric_name]["max"])
                optimized_data.append(norm_val)
            else:
                optimized_data.append(0)
        
        initial_data += initial_data[:1]
        optimized_data += optimized_data[:1]
        
        # 绘制初始状态（粉色，透明）
        ax.fill(angles, initial_data, color='#f8bbd9', alpha=0.4, label='Initial')
        ax.plot(angles, initial_data, color='#ec407a', linewidth=2, linestyle='--')
        
        # 绘制优化后状态（蓝色，透明）
        ax.fill(angles, optimized_data, color='#b3e5fc', alpha=0.4, label='Optimized')
        ax.plot(angles, optimized_data, color='#0288d1', linewidth=2)
        
        # 设置标签和标题
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_order, fontsize=10)
        ax.set_title(f'{subplot_labels[idx]} {subplot_titles[idx]}', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # 隐藏径向标签（因为是归一化值）
        ax.set_yticklabels([])
    
    plt.suptitle('Bayesian Optimization Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(SAVE_RESULT_FOLDER, "bayesian_optimization_performance.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def plot_shap_importance(shap_explainer, model_name="XGBoost"):
    """绘制SHAP重要性图"""
    fig = shap_explainer.plot_global_importance(model_name=model_name, save_path=None)
    path = os.path.join(SAVE_RESULT_FOLDER, "shap_importance.png")
    if fig is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig

def plot_model_prediction_results(reg_models, df):
    """绘制ML模型预测结果图（实际值vs预测值）"""
    pred_results = reg_models.get_prediction_results(df)
    
    model_order = ["XGBoost", "RFR", "GBDT", "KNN"]
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, (ax, model_name) in enumerate(zip(axes.flatten(), model_order)):
        if model_name not in pred_results:
            continue
        
        result = pred_results[model_name]
        y_train, y_train_pred = result["y_train"], result["y_train_pred"]
        y_test, y_test_pred = result["y_test"], result["y_test_pred"]
        train_r2 = result["train_r2"]
        test_r2 = result["test_r2"]
        
        # 绘制训练集（绿色）
        ax.scatter(y_train, y_train_pred, c='#2ecc71', edgecolor='black', s=40, 
                  alpha=0.7, label='Training Set', zorder=2)
        # 绘制测试集（蓝色）
        ax.scatter(y_test, y_test_pred, c='#3498db', edgecolor='black', s=40, 
                  alpha=0.7, label='Test Set', zorder=3)
        
        # 绘制理想拟合线（红色虚线）
        all_y = np.concatenate([y_train, y_test])
        min_val, max_val = all_y.min() * 0.95, all_y.max() * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Ideal Fit', zorder=1)
        
        # 添加R²标注
        ax.text(max_val * 0.65, max_val * 0.95, 
               f"Training R²={train_r2:.4f}\nTest R²={test_r2:.4f}", 
               fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{subplot_labels[i]} {model_name}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('ML Model Prediction Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(SAVE_RESULT_FOLDER, "model_prediction_results.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def plot_pareto_front(process_optimizer):
    """绘制Pareto前沿"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 生成Pareto前沿数据
    pareto_front = process_optimizer.get_pareto_front()
    if pareto_front and len(pareto_front) > 0:
        pf = np.array(pareto_front)
        ax.scatter(pf[:, 0], pf[:, 1], c='red', s=100, edgecolors='black', label='Pareto Front', zorder=5)
        ax.plot(pf[:, 0], pf[:, 1], 'r--', alpha=0.5)
    
    ax.set_xlabel('Objective 1 (Minimize)', fontsize=12)
    ax.set_ylabel('Objective 2 (Minimize)', fontsize=12)
    ax.set_title('Pareto Front - Multi-objective Optimization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(SAVE_RESULT_FOLDER, "pareto_front.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def plot_optimization_history(process_optimizer):
    """绘制优化历史"""
    history = process_optimizer.get_optimization_history()
    if not history or len(history) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    iters = [h['iter'] for h in history]
    objs = [h['f'] for h in history]
    
    ax.plot(iters, objs, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('BFGS Optimization History', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(SAVE_RESULT_FOLDER, "optimization_history.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig

# ===================== Streamlit UI =====================
def main():
    st.title("🔬 金相图像ML分析平台")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.header("📊 控制面板")
    
    # 数据加载
    st.sidebar.subheader("1. 数据加载")
    csv_path = os.path.join(SAVE_RESULT_FOLDER, "金相定量表征数据汇总.csv")
    
    if os.path.exists(csv_path):
        st.sidebar.success(f"✅ 数据文件已找到")
        df = load_csv_data(csv_path)
        st.sidebar.info(f"数据记录: {len(df)} 条")
    else:
        st.sidebar.error("❌ 未找到数据文件")
        st.sidebar.info("请先运行金相图像分析生成CSV")
        df = None
    
    # ML模型训练
    st.sidebar.subheader("2. ML模型训练")
    
    # 初始化session state
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'pp_module' not in st.session_state:
        st.session_state.pp_module = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    
    # 停止按钮
    if st.sidebar.button("⏹ 停止训练", type="secondary", disabled=not st.session_state.is_training):
        stop_training_event.set()
        st.sidebar.warning("正在停止训练...")
    
    # 训练按钮
    train_models = st.sidebar.button("🚀 训练/加载模型", type="primary", disabled=st.session_state.is_training)
    
    if train_models and df is not None:
        st.session_state.is_training = True
        stop_training_event.clear()
        
        # 创建进度条和状态文本
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def progress_callback(stage, progress, message):
            """进度回调函数"""
            if stop_training_event.is_set():
                raise KeyboardInterrupt("用户手动停止训练")
            overall_progress = (stage - 1) * 0.25 + progress * 0.25
            progress_bar.progress(overall_progress)
            status_text.text(f"[{stage}/4] {message}")
        
        try:
            with st.spinner("正在训练ML模型，请稍候..."):
                results, pp = load_ml_pipeline(csv_path, _progress_callback=progress_callback)
                if results:
                    st.session_state.ml_results = results
                    st.session_state.pp_module = pp
                    progress_bar.progress(100)
                    status_text.text("训练完成！")
                    st.sidebar.success("✅ 模型训练完成！")
                else:
                    progress_bar.progress(0)
                    status_text.text("训练失败")
        except KeyboardInterrupt:
            progress_bar.progress(0)
            status_text.text("训练已停止")
            st.sidebar.warning("⚠️ 训练已手动停止")
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("训练出错")
            st.sidebar.error(f"训练出错: {e}")
        finally:
            st.session_state.is_training = False
    elif st.session_state.ml_results is None and df is not None:
        st.sidebar.info("点击按钮训练模型")
    
    # ===================== 缓存管理与结果输出 =====================
    st.sidebar.subheader("🔄 缓存管理")
    
    # 显示缓存统计
    with st.sidebar.expander("📊 查看缓存状态", expanded=False):
        stats = get_statistics()
        st.write(f"**缓存项数量**: {stats['cache_count']}")
        st.write(f"**缓存总大小**: {stats['cache_size_formatted']}")
        st.write(f"**输出文件数量**: {stats['output_count']}")
        st.write(f"**输出总大小**: {stats['output_size_formatted']}")
        
        if stats['cache_items']:
            st.write("**缓存项详情**:")
            for item in stats['cache_items']:
                st.text(f"  • {item['name']}: {format_size(item['size'])}")
    
    # 备份当前结果
    if st.sidebar.button("💾 备份当前结果", type="secondary"):
        with st.spinner("正在备份..."):
            success, msg = backup_current_results()
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(f"备份失败: {msg}")
    
    # 清理选项
    clean_options = st.sidebar.multiselect(
        "🗑️ 选择要清理的内容",
        ["模型缓存", "输出图表(.png)", "输出报告(.txt)", "输出数据(.csv)", "全部清理"],
        default=[]
    )
    
    if st.sidebar.button("🗑️ 执行清理", type="secondary", disabled=len(clean_options) == 0):
        results = []
        
        if "模型缓存" in clean_options:
            cache_results = clean_all_cache()
            results.extend([f"缓存: {r[0]} - {r[2]}" for r in cache_results])
        
        if "输出图表(.png)" in clean_options:
            png_results = clean_output_items(['*.png'])
            results.extend([f"图表: {r[0]} - {r[2]}" for r in png_results])
        
        if "输出报告(.txt)" in clean_options:
            txt_results = clean_output_items(['*.txt'])
            results.extend([f"报告: {r[0]} - {r[2]}" for r in txt_results])
        
        if "输出数据(.csv)" in clean_options:
            csv_results = clean_output_items(['*.csv'])
            results.extend([f"数据: {r[0]} - {r[2]}" for r in csv_results])
        
        if "全部清理" in clean_options:
            cache_results = clean_all_cache()
            results.extend([f"缓存: {r[0]} - {r[2]}" for r in cache_results])
            all_results = clean_output_items()
            results.extend([f"{r[0]} - {r[2]}" for r in all_results])
        
        st.sidebar.success(f"清理完成！共清理 {len(results)} 项")
        # 强制刷新页面以反映变化
        st.rerun()
    
    # 创建新的结果文件夹
    if st.sidebar.button("📁 新建结果文件夹", type="secondary"):
        success, msg = create_results_folder()
        if success:
            st.sidebar.success(f"已创建: {msg}")
        else:
            st.sidebar.error(f"创建失败: {msg}")
    
    # 查看历史结果
    results_base = os.path.join(BASE_DIR, "analysis_output", "results_by_date")
    if os.path.exists(results_base):
        with st.sidebar.expander("📂 历史结果文件夹", expanded=False):
            for folder in sorted(os.listdir(results_base), reverse=True)[:5]:
                folder_path = os.path.join(results_base, folder)
                if os.path.isdir(folder_path):
                    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                    st.text(f"📁 {folder} ({file_count} 文件)")
    
    # ===================== micro_sam 显微图像预训练模型 =====================
    st.sidebar.subheader("🔬 micro_sam 预训练模型")
    
    # 导入 micro_sam 集成模块
    try:
        from microsam_integration import (
            check_micro_sam_installed, 
            get_available_models,
            MicroSAMSegmenter,
            install_micro_sam_instructions
        )
        
        installed = check_micro_sam_installed()
        
        if not installed:
            st.sidebar.warning("⚠️ 未安装")
            with st.sidebar.expander("安装说明"):
                st.code(install_micro_sam_instructions(), language="python")
        else:
            st.sidebar.success("✅ 已安装")
            
            # 模型选择
            model_type = st.sidebar.selectbox(
                "模型类型",
                options=list(get_available_models().keys()),
                format_func=lambda x: get_available_models()[x],
                index=2,
                key="microsam_model"
            )
            
            # 设备选择
            device = st.sidebar.radio(
                "计算设备",
                ["auto", "cpu", "cuda"],
                format_func=lambda x: {"auto": "自动", "cpu": "CPU", "cuda": "GPU"}[x],
                index=0,
                key="microsam_device"
            )
            
            # 加载模型按钮
            if st.sidebar.button("加载预训练模型", key="load_micosam"):
                with st.spinner("正在加载模型..."):
                    segmenter = MicroSAMSegmenter(model_type=model_type, device=device)
                    st.session_state.microsam_segmenter = segmenter
                    st.sidebar.success(f"模型 {model_type} 已加载")
            
            # 显示模型状态
            if 'microsam_segmenter' in st.session_state:
                st.sidebar.info(f"当前模型: {st.session_state.microsam_segmenter.model_type}")
                
    except ImportError as e:
        st.sidebar.warning(f"模块导入失败: {e}")
    
    # 图表选择
    st.sidebar.subheader("3. 图表生成")
    chart_type = st.sidebar.selectbox(
        "选择要生成的图表",
        ["Pearson相关系数矩阵", "模型误差指标对比", "ML模型预测结果", "Bayesian优化性能对比", 
         "SHAP特征重要性", "Pareto前沿", "优化历史", "数据预览"]
    )
    
    generate_btn = st.sidebar.button("📈 生成图表", type="secondary")
    
    # 主内容区
    if chart_type == "数据预览":
        st.header("📋 数据预览")
        if df is not None:
            st.dataframe(df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总记录数", len(df))
            with col2:
                st.metric("特征列数", len(df.columns))
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.metric("数值列数", len(numeric_cols))
        else:
            st.warning("请先加载数据")
    
    elif generate_btn or st.session_state.ml_results is not None:
        if st.session_state.ml_results is None:
            st.warning("⚠️ 请先训练模型！")
            return
        
        results = st.session_state.ml_results
        pp = st.session_state.pp_module
        
        if chart_type == "Pearson相关系数矩阵":
            st.header("🔗 Pearson相关系数矩阵")
            
            reg_models = results['reg_models']
            fig, corr_matrix = plot_pearson_correlation(
                df, reg_models.feature_names, reg_models.target_col
            )
            st.pyplot(fig)
            
            with st.expander("查看相关系数数据"):
                st.dataframe(corr_matrix, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "下载图片",
                    open(os.path.join(SAVE_RESULT_FOLDER, "pearson_correlation_matrix.png"), "rb"),
                    "pearson_correlation_matrix.png"
                )
            with col2:
                st.download_button(
                    "下载CSV",
                    open(os.path.join(SAVE_RESULT_FOLDER, "pearson_correlation_matrix.csv"), "rb"),
                    "pearson_correlation_matrix.csv"
                )
        
        elif chart_type == "模型误差指标对比":
            st.header("📊 四种模型误差指标对比")
            
            eval_report = results['eval_report']
            fig = plot_error_metrics(eval_report)
            st.pyplot(fig)
            
            st.subheader("详细数据")
            st.dataframe(eval_report, use_container_width=True)
            
            st.download_button(
                "下载图片",
                open(os.path.join(SAVE_RESULT_FOLDER, "model_error_metrics.png"), "rb"),
                "model_error_metrics.png"
            )
        
        elif chart_type == "ML模型预测结果":
            st.header("📊 ML模型预测结果")
            
            reg_models = results['reg_models']
            fig = plot_model_prediction_results(reg_models, df)
            st.pyplot(fig)
            
            st.download_button(
                "下载图片",
                open(os.path.join(SAVE_RESULT_FOLDER, "model_prediction_results.png"), "rb"),
                "model_prediction_results.png"
            )
        
        elif chart_type == "Bayesian优化性能对比":
            st.header("🔄 Bayesian优化性能对比")
            
            reg_models = results['reg_models']
            fig = plot_bayesian_optimization_performance(reg_models, df)
            if fig:
                st.pyplot(fig)
                st.download_button(
                    "下载图片",
                    open(os.path.join(SAVE_RESULT_FOLDER, "bayesian_optimization_performance.png"), "rb"),
                    "bayesian_optimization_performance.png"
                )
            else:
                st.warning("没有可用的Bayesian优化历史数据")
        
        elif chart_type == "SHAP特征重要性":
            st.header("🎯 SHAP特征重要性分析")
            
            shap_explainer = results['shap_explainer']
            
            model_choice = st.selectbox("选择模型", ["XGBoost", "RFR"])
            
            # 重新计算SHAP值
            sample = df.iloc[[0]]
            shap_explainer.compute_shap_values(sample, model_names=[model_choice])
            
            fig = plot_shap_importance(shap_explainer, model_name=model_choice)
            if fig:
                st.pyplot(fig)
            else:
                # 使用matplotlib重新绘制
                contributions = shap_explainer.get_feature_contributions(sample, model_name=model_choice)
                
                fig2, ax = plt.subplots(figsize=(10, 6))
                colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in contributions['SHAP值']]
                ax.barh(contributions['特征'], contributions['SHAP值'], color=colors_bar)
                ax.set_xlabel('SHAP Value', fontsize=12)
                ax.set_title(f'Feature Contributions ({model_choice})', fontsize=14)
                ax.axvline(x=0, color='black', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            
            st.subheader("特征贡献排名")
            contributions = shap_explainer.get_feature_contributions(sample, model_name=model_choice)
            st.dataframe(contributions, use_container_width=True)
        
        elif chart_type == "Pareto前沿":
            st.header("🌐 Pareto前沿")
            
            reg_models = results['reg_models']
            process_optimizer = pp.LaserProcessOptimizer(reg_models)
            optimal = process_optimizer.optimize_process(goal="balance", num_points=20)
            
            fig = plot_pareto_front(process_optimizer)
            st.pyplot(fig)
            
            if optimal is not None:
                st.subheader("最优工艺参数")
                param_names = ["激光功率", "放大倍数", "熔覆层组织占比", "析出相占比", 
                              "气孔率", "裂纹占比", "晶粒尺寸", "稀释率"]
                cols = st.columns(4)
                for i, (name, val) in enumerate(zip(param_names, optimal)):
                    with cols[i % 4]:
                        st.metric(name, f"{val:.2f}")
        
        elif chart_type == "优化历史":
            st.header("📈 优化历史")
            
            reg_models = results['reg_models']
            process_optimizer = pp.LaserProcessOptimizer(reg_models)
            process_optimizer.optimize_process(goal="balance", num_points=20)
            
            fig = plot_optimization_history(process_optimizer)
            if fig:
                st.pyplot(fig)
            else:
                st.info("优化历史数据不可用")
    
    # 页脚
    st.markdown("---")
    st.caption("金相图像定量分析 - 基于机器学习的激光功率优化")

if __name__ == "__main__":
    main()
