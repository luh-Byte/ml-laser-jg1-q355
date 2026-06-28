"""
单变量分析脚本
研究每个特征与目标变量(显微硬度)的关系
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings_module = __import__('warnings')
warnings_module.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# 导入统一绘图风格
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from plot_style import setup_plot_style, style_axes, create_gradient_rect, add_subplot_label, calc_sem, save_fig, COLORS, POWER_LIST, POWER_NUM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures", "univariate")
os.makedirs(FIG_DIR, exist_ok=True)

# 设置统一绘图风格
setup_plot_style()


def load_data():
    """加载数据"""
    csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["power_w"] = df["激光功率"].apply(lambda x: float(str(x).replace("W", "")))
    return df


def get_feature_groups():
    """定义特征分组"""
    return {
        "工艺参数": ["power_w"],
        "金相组织": [
            "熔覆层组织面积占比(%)",
            "析出相/碳化物面积占比(%)",
            "气孔孔隙率(%)",
            "微裂纹面积占比(%)",
            "熔覆层平均晶粒尺寸(μm)",
            "基体稀释率(%)",
        ],
        "EIS电化学": [
            "eis_Rs_ohm",
            "eis_Rct_ohm",
            "eis_Z_max_ohm",
            "eis_theta_min_deg",
        ],
        "XRD物相": [
            "xrd_main_peak_2theta",
            "xrd_main_peak_intensity",
            "xrd_peak_44_area",
        ],
        "摩擦磨损": [
            "wear_friction_steady",
            "wear_friction_std",
        ],
        "梯度硬度": [
            "mh_cladding_hv",
            "mh_substrate_hv",
            "mh_gradient_range",
        ],
    }


def calculate_correlation(df, feature, target="mh_mean_hv"):
    """计算单个特征与目标的相关性"""
    x = df[feature].values
    y = df[target].values
    
    # 去除NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 3:
        return None
    
    # Pearson相关
    r_pearson, p_pearson = stats.pearsonr(x, y)
    # Spearman相关
    r_spearman, p_spearman = stats.spearmanr(x, y)
    
    return {
        "feature": feature,
        "n_samples": len(x),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "x_mean": np.mean(x),
        "x_std": np.std(x),
        "x_min": np.min(x),
        "x_max": np.max(x),
    }


def plot_scatter(df, feature, target="mh_mean_hv", save=True):
    """绘制单变量散点图"""
    x = df[feature].values
    y = df[target].values
    powers = df["power_w"].values
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y, powers = x[mask], y[mask], powers[mask]
    
    if len(x) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {900: '#2381c4', 1200: '#ff7f0e', 1500: '#2baf2b', 1800: '#d62728'}
    for p in sorted(set(powers)):
        mask_p = powers == p
        ax.scatter(x[mask_p], y[mask_p], c=colors.get(p, 'gray'), 
                   s=80, alpha=0.7, edgecolors='black', linewidth=1, label=f'{int(p)}W')
    
    # 拟合线
    if len(x) > 2 and np.std(x) > 1e-10:
        try:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_fit(x_line), 'k--', linewidth=2, alpha=0.5)
            r, p_val = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass
    
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("显微硬度 (HV)", fontsize=12)
    ax.set_title(f"{feature} vs 显微硬度", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        safe_name = feature.replace("/", "_").replace("%", "pct").replace("(", "").replace(")", "")
        fig_path = os.path.join(FIG_DIR, f"scatter_{safe_name}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("单变量分析 - 特征与显微硬度关系")
    print("=" * 60)
    
    # 加载数据
    df = load_data()
    feature_groups = get_feature_groups()
    target = "mh_mean_hv"
    
    print(f"\n样本数: {len(df)}")
    print(f"目标变量: {target} (范围: {df[target].min():.1f} - {df[target].max():.1f} HV)")
    
    # 收集所有结果
    all_results = []
    
    for group_name, features in feature_groups.items():
        print(f"\n{'='*60}")
        print(f"特征组: {group_name}")
        print(f"{'='*60}")
        
        for feature in features:
            if feature not in df.columns:
                print(f"  [SKIP] {feature} 不存在")
                continue
            
            result = calculate_correlation(df, feature, target)
            if result is None:
                print(f"  [SKIP] {feature} 数据不足")
                continue
            
            result["group"] = group_name
            all_results.append(result)
            
            # 绘制散点图
            plot_scatter(df, feature, target)
            
            # 打印结果
            sig = "***" if result["pearson_p"] < 0.001 else "**" if result["pearson_p"] < 0.01 else "*" if result["pearson_p"] < 0.05 else ""
            print(f"  {feature}:")
            print(f"    Pearson r = {result['pearson_r']:.3f} {sig}")
            print(f"    Spearman r = {result['spearman_r']:.3f}")
    
    # 保存结果汇总
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values("pearson_r", key=abs, ascending=False)
        
        save_path = os.path.join(OUTPUT_DIR, "univariate_correlation_results.csv")
        df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*60}")
        print("相关性排名 (按|Pearson r|排序)")
        print(f"{'='*60}")
        for _, row in df_results.iterrows():
            sig = "***" if row["pearson_p"] < 0.001 else "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
            print(f"  {row['feature']:30s} r={row['pearson_r']:+.3f} {sig}")
        
        print(f"\n[OK] 结果已保存: {save_path}")
        print(f"[OK] 散点图已保存: {FIG_DIR}")
    
    print("\n" + "=" * 60)
    print("单变量分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
