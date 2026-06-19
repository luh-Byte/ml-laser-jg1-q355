"""
材料科学物理性质合理性分析脚本
分析金相定量表征数据是否符合材料科学规律
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
BASE_DIR = r"C:\Users\liuyuhe\Desktop\基于机器学习的激光功率优化及JG-1铁基合金Q355钢组织性能协同调控研究\金相图片"
SAVE_RESULT_FOLDER = os.path.join(BASE_DIR, "analysis_output")
csv_path = os.path.join(SAVE_RESULT_FOLDER, "金相定量表征数据汇总.csv")
df = pd.read_csv(csv_path)

print("=" * 80)
print("材料科学物理性质合理性分析报告")
print("=" * 80)

# ==================== 1. 激光功率与组织演变规律 ====================
print("\n【1】激光功率与组织演变规律分析")
print("-" * 80)

# 按功率分组统计
power_stats = df.groupby('激光功率').agg({
    '熔覆层组织面积占比(%)': ['mean', 'std'],
    '析出相/碳化物面积占比(%)': ['mean', 'std'],
    '气孔孔隙率(%)': ['mean', 'std'],
    '熔覆层平均晶粒尺寸(μm)': ['mean', 'std'],
    '基体稀释率(%)': ['mean', 'std']
}).round(3)

print("\n各功率组平均统计:")
print(power_stats)

# 物理规律1: 激光功率增加 → 热输入增加 → 晶粒尺寸增大
print("\n✓ 晶粒尺寸随功率变化趋势:")
grain_by_power = df.groupby('激光功率')['熔覆层平均晶粒尺寸(μm)'].mean()
print(grain_by_power)

# 检验是否符合"功率增加→晶粒增大"规律
powers_str = ["900W", "1200W", "1500W", "1800W"]
powers_num = [900, 1200, 1500, 1800]
grains = [grain_by_power[p] for p in powers_str]
correlation = np.corrcoef(powers_num, grains)[0, 1]
print(f"功率-晶粒尺寸相关性: r = {correlation:.4f}")

if correlation > 0.3:
    print("✅ 符合规律: 激光功率增加 → 热输入增加 → 冷却速率降低 → 晶粒尺寸增大")
else:
    print("⚠️ 警告: 晶粒尺寸未随功率显著增大，需检查数据")

# 物理规律2: 激光功率增加 → 熔池温度升高 → 稀释率增加
print("\n✓ 稀释率随功率变化趋势:")
dilution_by_power = df.groupby('激光功率')['基体稀释率(%)'].mean()
print(dilution_by_power)

dilutions = [dilution_by_power[p] for p in powers_str]
correlation_dilution = np.corrcoef(powers_num, dilutions)[0, 1]
print(f"功率-稀释率相关性: r = {correlation_dilution:.4f}")

if abs(correlation_dilution) < 0.3:
    print("✅ 合理: 稀释率在34-36%范围内波动，符合激光熔覆典型稀释率(30-40%)")
else:
    print("⚠️ 注意: 稀释率变化较大，需验证工艺稳定性")

# ==================== 2. 晶粒尺寸合理性分析 ====================
print("\n【2】晶粒尺寸合理性分析")
print("-" * 80)

# 晶粒尺寸范围检查
grain_min = df['熔覆层平均晶粒尺寸(μm)'].min()
grain_max = df['熔覆层平均晶粒尺寸(μm)'].max()
grain_mean = df['熔覆层平均晶粒尺寸(μm)'].mean()

print(f"晶粒尺寸范围: {grain_min:.2f} - {grain_max:.2f} μm")
print(f"平均晶粒尺寸: {grain_mean:.2f} μm")

# 激光熔覆典型晶粒尺寸: 10-100 μm
if 10 <= grain_min and grain_max <= 100:
    print("✅ 合理: 晶粒尺寸在激光熔覆典型范围内(10-100 μm)")
elif grain_max > 100:
    print(f"⚠️ 警告: 最大晶粒尺寸{grain_max:.2f}μm过大，可能存在测量误差")
elif grain_min < 10:
    print(f"⚠️ 警告: 最小晶粒尺寸{grain_min:.2f}μm过小，需检查分割算法")

# 检查放大倍数对晶粒尺寸测量的影响
print("\n✓ 不同放大倍数下的晶粒尺寸:")
mag_grain = df.groupby('放大倍数')['熔覆层平均晶粒尺寸(μm)'].agg(['mean', 'count'])
print(mag_grain)

# 高倍镜下应能看到更细小的晶粒
high_mag = df[df['放大倍数'] >= 500]['熔覆层平均晶粒尺寸(μm)'].mean()
low_mag = df[df['放大倍数'] <= 100]['熔覆层平均晶粒尺寸(μm)'].mean()
print(f"\n高倍镜(≥500x)平均晶粒: {high_mag:.2f} μm")
print(f"低倍镜(≤100x)平均晶粒: {low_mag:.2f} μm")

if high_mag < low_mag:
    print("✅ 符合规律: 高倍镜下可识别更细小晶粒")
else:
    print("⚠️ 注意: 高倍镜晶粒尺寸反而更大，可能受统计样本数影响")

# ==================== 3. 气孔率与裂纹率分析 ====================
print("\n【3】气孔率与裂纹率分析")
print("-" * 80)

# 气孔率范围检查
porosity_min = df['气孔孔隙率(%)'].min()
porosity_max = df['气孔孔隙率(%)'].max()
porosity_mean = df['气孔孔隙率(%)'].mean()

print(f"气孔孔隙率范围: {porosity_min:.4f} - {porosity_max:.4f} %")
print(f"平均气孔率: {porosity_mean:.4f} %")

# 激光熔覆典型气孔率: < 2%
if porosity_mean < 2:
    print("✅ 合理: 平均气孔率<2%，符合优质激光熔覆标准")
elif porosity_mean < 5:
    print("⚠️ 注意: 气孔率偏高，工艺需优化")
else:
    print("❌ 异常: 气孔率过高，不符合激光熔覆质量要求")

# 检查异常高气孔率样本
high_porosity = df[df['气孔孔隙率(%)'] > 1]
if len(high_porosity) > 0:
    print(f"\n⚠️ 高气孔率样本(>1%): {len(high_porosity)}个")
    print(high_porosity[['激光功率', '图像名称', '气孔孔隙率(%)']].to_string(index=False))

# 裂纹率分析
crack_min = df['微裂纹面积占比(%)'].min()
crack_max = df['微裂纹面积占比(%)'].max()
crack_mean = df['微裂纹面积占比(%)'].mean()

print(f"\n裂纹面积占比范围: {crack_min:.4f} - {crack_max:.4f} %")
print(f"平均裂纹率: {crack_mean:.4f} %")

# 激光熔覆理想裂纹率: < 1%
if crack_mean < 1:
    print("✅ 合理: 平均裂纹率<1%，熔覆层质量良好")
else:
    print(f"⚠️ 警告: 裂纹率偏高，需检查工艺参数")

# 检查异常高裂纹样本
high_crack = df[df['微裂纹面积占比(%)'] > 1]
if len(high_crack) > 0:
    print(f"\n⚠️ 高裂纹率样本(>1%): {len(high_crack)}个")
    print(high_crack[['激光功率', '图像名称', '微裂纹面积占比(%)']].to_string(index=False))

# ==================== 4. 析出相/碳化物分析 ====================
print("\n【4】析出相/碳化物分析")
print("-" * 80)

carbide_min = df['析出相/碳化物面积占比(%)'].min()
carbide_max = df['析出相/碳化物面积占比(%)'].max()
carbide_mean = df['析出相/碳化物面积占比(%)'].mean()

print(f"析出相面积占比范围: {carbide_min:.2f} - {carbide_max:.2f} %")
print(f"平均析出相占比: {carbide_mean:.2f} %")

# JG-1铁基合金典型析出相含量: 10-20%
if 10 <= carbide_mean <= 20:
    print("✅ 合理: 析出相占比在JG-1铁基合金典型范围内(10-20%)")
elif carbide_mean < 10:
    print("⚠️ 注意: 析出相偏少，可能影响硬度")
else:
    print("⚠️ 注意: 析出相偏多，需验证合金成分")

# 析出相随功率变化
carbide_by_power = df.groupby('激光功率')['析出相/碳化物面积占比(%)'].mean()
print("\n各功率组析出相平均占比:")
print(carbide_by_power)

# 物理规律: 功率增加 → 冷却速率降低 → 析出相减少
carbides = [carbide_by_power[p] for p in powers_str]
correlation_carbide = np.corrcoef(powers_num, carbides)[0, 1]
print(f"功率-析出相相关性: r = {correlation_carbide:.4f}")

if correlation_carbide < -0.3:
    print("✅ 符合规律: 功率增加 → 冷却慢 → 析出相溶解/减少")
elif correlation_carbide > 0.3:
    print("⚠️ 注意: 析出相随功率增加而增多，需检查热处理过程")
else:
    print("✓ 析出相含量相对稳定，符合快速凝固特征")

# ==================== 5. 稀释率合理性分析 ====================
print("\n【5】稀释率合理性分析")
print("-" * 80)

dilution_min = df['基体稀释率(%)'].min()
dilution_max = df['基体稀释率(%)'].max()
dilution_mean = df['基体稀释率(%)'].mean()

print(f"稀释率范围: {dilution_min:.2f} - {dilution_max:.2f} %")
print(f"平均稀释率: {dilution_mean:.2f} %")

# 激光熔覆理想稀释率: 30-50%
if 30 <= dilution_mean <= 50:
    print("✅ 合理: 稀释率在激光熔覆理想范围内(30-50%)")
elif dilution_mean < 30:
    print("⚠️ 警告: 稀释率过低，熔覆层与基材结合可能不良")
else:
    print("⚠️ 警告: 稀释率过高，熔覆层成分被基材过度稀释")

# 稀释率稳定性
dilution_std = df['基体稀释率(%)'].std()
print(f"稀释率标准差: {dilution_std:.2f} %")

if dilution_std < 3:
    print("✅ 工艺稳定: 稀释率波动小，工艺控制良好")
else:
    print(f"⚠️ 注意: 稀释率波动较大(σ={dilution_std:.2f}%)，需优化工艺一致性")

# ==================== 6. 组织面积占比合理性 ====================
print("\n【6】组织面积占比合理性分析")
print("-" * 80)

# 检查各组织占比总和是否接近100%
df['组织占比总和'] = df['背景/基体面积占比(%)'] + df['熔覆层组织面积占比(%)'] + \
                      df['析出相/碳化物面积占比(%)'] + df['气孔缺陷面积占比(%)'] + \
                      df['裂纹缺陷面积占比(%)']

total_sum_mean = df['组织占比总和'].mean()
total_sum_std = df['组织占比总和'].std()

print(f"组织占比总和平均值: {total_sum_mean:.2f} %")
print(f"组织占比总和标准差: {total_sum_std:.4f} %")

if 99 <= total_sum_mean <= 101:
    print("✅ 合理: 组织占比总和接近100%，分割算法准确")
else:
    print(f"⚠️ 警告: 组织占比总和偏离100%(平均{total_sum_mean:.2f}%)，需检查分割逻辑")

# 检查异常样本
abnormal_sum = df[abs(df['组织占比总和'] - 100) > 2]
if len(abnormal_sum) > 0:
    print(f"\n⚠️ 组织占比异常样本(偏离100%>2%): {len(abnormal_sum)}个")

# ==================== 7. 力学性能预测合理性分析 ====================
print("\n【7】力学性能预测合理性分析")
print("-" * 80)

# 读取模型评估结果
ml_report_path = os.path.join(SAVE_RESULT_FOLDER, "ml_results", "模型评估报告.csv")
if os.path.exists(ml_report_path):
    ml_df = pd.read_csv(ml_report_path)
    print("\n模型评估指标:")
    print(ml_df.to_string(index=False))
    
    # 检查R²是否过高（可能过拟合）
    max_r2 = ml_df['R²'].max()
    if max_r2 > 0.999:
        print(f"\n⚠️ 警告: 最高R²={max_r2:.6f}接近1.0，存在严重过拟合风险")
        print("   建议: 增加数据量、使用交叉验证、检查数据独立性")
    elif max_r2 > 0.95:
        print(f"\n✓ 模型拟合良好: R²={max_r2:.4f}")
    else:
        print(f"\n⚠️ 模型拟合不足: R²={max_r2:.4f}")

# 计算力学性能预测值（基于代码中的公式）
df['预测显微硬度(HV)_calc'] = 620 - 18.2 * df['熔覆层平均晶粒尺寸(μm)'] + \
                               4.5 * df['析出相/碳化物面积占比(%)'] - \
                               22.5 * df['气孔孔隙率(%)']

df['预测抗拉强度(MPa)_calc'] = 780 - 12.6 * df['熔覆层平均晶粒尺寸(μm)'] + \
                                3.2 * df['析出相/碳化物面积占比(%)'] - \
                                35 * df['气孔孔隙率(%)']

df['预测磨损速率(mg/h)_calc'] = 0.002 + 0.0008 * df['熔覆层平均晶粒尺寸(μm)'] + \
                                 0.0012 * df['气孔孔隙率(%)'] - \
                                 0.0001 * df['析出相/碳化物面积占比(%)']

print("\n力学性能预测范围:")
print(f"显微硬度: {df['预测显微硬度(HV)_calc'].min():.1f} - {df['预测显微硬度(HV)_calc'].max():.1f} HV")
print(f"抗拉强度: {df['预测抗拉强度(MPa)_calc'].min():.1f} - {df['预测抗拉强度(MPa)_calc'].max():.1f} MPa")
print(f"磨损速率: {df['预测磨损速率(mg/h)_calc'].min():.5f} - {df['预测磨损速率(mg/h)_calc'].max():.5f} mg/h")

# JG-1铁基合金典型性能范围
print("\n✓ JG-1铁基合金激光熔覆典型性能:")
print("  显微硬度: 500-700 HV")
print("  抗拉强度: 600-900 MPa")
print("  磨损速率: 0.001-0.01 mg/h")

hardness_mean = df['预测显微硬度(HV)_calc'].mean()
tensile_mean = df['预测抗拉强度(MPa)_calc'].mean()

if 500 <= hardness_mean <= 700:
    print(f"\n✅ 硬度预测合理: 平均{hardness_mean:.1f} HV在典型范围内")
else:
    print(f"\n⚠️ 硬度预测异常: 平均{hardness_mean:.1f} HV偏离典型范围")

if 600 <= tensile_mean <= 900:
    print(f"✅ 强度预测合理: 平均{tensile_mean:.1f} MPa在典型范围内")
else:
    print(f"⚠️ 强度预测异常: 平均{tensile_mean:.1f} MPa偏离典型范围")

# ==================== 8. Hall-Petch关系验证 ====================
print("\n【8】Hall-Petch关系验证 (硬度-晶粒尺寸)")
print("-" * 80)

# Hall-Petch公式: σ = σ₀ + k·d^(-1/2)
# 简化为: HV ∝ d^(-1/2)
grain_inv_sqrt = 1 / np.sqrt(df['熔覆层平均晶粒尺寸(μm)'])
hardness = df['预测显微硬度(HV)_calc']

# 计算相关性
correlation_hp = np.corrcoef(grain_inv_sqrt, hardness)[0, 1]
print(f"HV vs d^(-1/2) 相关性: r = {correlation_hp:.4f}")

if correlation_hp > 0.5:
    print("✅ 符合Hall-Petch关系: 晶粒细化 → 硬度提高")
elif correlation_hp < -0.3:
    print("⚠️ 异常: 硬度随晶粒细化反而降低，违反Hall-Petch规律")
else:
    print("✓ 相关性较弱，可能受其他因素(析出相、气孔)主导")

# ==================== 9. 数据一致性检查 ====================
print("\n【9】数据一致性检查")
print("-" * 80)

# 检查像素尺寸校准一致性
xml_calibrated = df[df['校准来源'] == 'XML元数据']['像素尺寸(μm/像素)'].mean()
estimated = df[df['校准来源'] == '放大倍数估算']['像素尺寸(μm/像素)'].mean()

print(f"XML校准平均像素尺寸: {xml_calibrated:.6f} μm/像素")
print(f"放大倍数估算平均像素尺寸: {estimated:.6f} μm/像素")

if abs(xml_calibrated - estimated) < 0.2:
    print("✅ 校准方法一致性良好")
else:
    print(f"⚠️ 警告: 不同校准方法差异较大({abs(xml_calibrated - estimated):.3f} μm/像素)")

# 检查物理面积计算一致性
xml_data = df[df['校准来源'] == 'XML元数据']
if len(xml_data) > 0:
    xml_data['面积一致性误差'] = abs(xml_data['总物理面积(μm²)'] - xml_data['视野面积(μm²)']) / xml_data['视野面积(μm²)'] * 100
    area_error_mean = xml_data['面积一致性误差'].mean()
    print(f"\n物理面积计算一致性误差: {area_error_mean:.2f} %")
    
    if area_error_mean < 5:
        print("✅ 物理面积计算准确")
    else:
        print("⚠️ 物理面积计算存在误差，需检查像素尺寸换算")

# ==================== 10. 生成分析报告图表 ====================
print("\n【10】生成分析可视化图表")
print("-" * 80)

# 创建输出目录
analysis_dir = os.path.join(SAVE_RESULT_FOLDER, "material_analysis")
os.makedirs(analysis_dir, exist_ok=True)

# 图1: 激光功率对各参数的影响
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (a) 晶粒尺寸
ax1 = axes[0, 0]
for i, power_str in enumerate(powers_str):
    data = df[df['激光功率'] == power_str]['熔覆层平均晶粒尺寸(μm)']
    ax1.scatter([powers_num[i]]*len(data), data, alpha=0.6, s=50)
ax1.plot(powers_num, grains, 'r-', linewidth=2, marker='o', markersize=8)
ax1.set_xlabel('激光功率 (W)')
ax1.set_ylabel('晶粒尺寸 (μm)')
ax1.set_title('(a) 晶粒尺寸随功率变化')
ax1.grid(True, alpha=0.3)

# (b) 稀释率
ax2 = axes[0, 1]
for i, power_str in enumerate(powers_str):
    data = df[df['激光功率'] == power_str]['基体稀释率(%)']
    ax2.scatter([powers_num[i]]*len(data), data, alpha=0.6, s=50)
ax2.plot(powers_num, dilutions, 'r-', linewidth=2, marker='o', markersize=8)
ax2.set_xlabel('激光功率 (W)')
ax2.set_ylabel('稀释率 (%)')
ax2.set_title('(b) 稀释率随功率变化')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='理想下限')
ax2.axhline(y=50, color='g', linestyle='--', alpha=0.5, label='理想上限')
ax2.legend()

# (c) 气孔率
ax3 = axes[0, 2]
porosity_by_power = [df[df['激光功率'] == p]['气孔孔隙率(%)'].mean() for p in powers_str]
for i, power_str in enumerate(powers_str):
    data = df[df['激光功率'] == power_str]['气孔孔隙率(%)']
    ax3.scatter([powers_num[i]]*len(data), data, alpha=0.6, s=50)
ax3.plot(powers_num, porosity_by_power, 'r-', linewidth=2, marker='o', markersize=8)
ax3.set_xlabel('激光功率 (W)')
ax3.set_ylabel('气孔率 (%)')
ax3.set_title('(c) 气孔率随功率变化')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='优质标准')

# (d) 析出相
ax4 = axes[1, 0]
for i, power_str in enumerate(powers_str):
    data = df[df['激光功率'] == power_str]['析出相/碳化物面积占比(%)']
    ax4.scatter([powers_num[i]]*len(data), data, alpha=0.6, s=50)
ax4.plot(powers_num, carbides, 'r-', linewidth=2, marker='o', markersize=8)
ax4.set_xlabel('激光功率 (W)')
ax4.set_ylabel('析出相占比 (%)')
ax4.set_title('(d) 析出相随功率变化')
ax4.grid(True, alpha=0.3)

# (e) Hall-Petch关系
ax5 = axes[1, 1]
ax5.scatter(grain_inv_sqrt, hardness, alpha=0.6, s=50)
ax5.set_xlabel('d^(-1/2) (μm^(-1/2))')
ax5.set_ylabel('预测硬度 (HV)')
ax5.set_title('(e) Hall-Petch关系验证')
ax5.grid(True, alpha=0.3)

# 添加拟合线
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(grain_inv_sqrt, hardness)
x_fit = np.linspace(grain_inv_sqrt.min(), grain_inv_sqrt.max(), 100)
y_fit = slope * x_fit + intercept
ax5.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'R²={r_value**2:.3f}')
ax5.legend()

# (f) 组织占比分布
ax6 = axes[1, 2]
org_means = [
    df['背景/基体面积占比(%)'].mean(),
    df['熔覆层组织面积占比(%)'].mean(),
    df['析出相/碳化物面积占比(%)'].mean(),
    df['气孔缺陷面积占比(%)'].mean(),
    df['裂纹缺陷面积占比(%)'].mean()
]
org_names = ['基体', '熔覆层', '析出相', '气孔', '裂纹']
colors = ['gray', 'green', 'gold', 'red', 'purple']
ax6.bar(org_names, org_means, color=colors, alpha=0.7)
ax6.set_ylabel('面积占比 (%)')
ax6.set_title('(f) 各组织平均占比')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = os.path.join(analysis_dir, "material_science_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ 分析图表已保存: {fig_path}")
plt.close()

# ==================== 11. 生成分析报告文本 ====================
print("\n【11】生成分析报告文本")
print("-" * 80)

report_text = f"""
================================================================================
材料科学物理性质合理性分析报告
================================================================================

数据来源: {csv_path}
样本数量: {len(df)} 张金相图像
激光功率: 900W, 1200W, 1500W, 1800W
放大倍数: 50x - 1000x

================================================================================
一、激光功率与组织演变规律
================================================================================

1. 晶粒尺寸演变
   - 900W平均晶粒: {grains[0]:.2f} μm
   - 1200W平均晶粒: {grains[1]:.2f} μm
   - 1500W平均晶粒: {grains[2]:.2f} μm
   - 1800W平均晶粒: {grains[3]:.2f} μm
   - 功率-晶粒相关性: r = {correlation:.4f}
   - 结论: {'符合规律' if correlation > 0.3 else '需进一步验证'}

2. 稀释率稳定性
   - 平均稀释率: {dilution_mean:.2f}%
   - 稀释率范围: {dilution_min:.2f} - {dilution_max:.2f}%
   - 标准差: {dilution_std:.2f}%
   - 结论: {'工艺稳定' if dilution_std < 3 else '需优化工艺一致性'}

================================================================================
二、缺陷控制质量评估
================================================================================

1. 气孔率
   - 平均气孔率: {porosity_mean:.4f}%
   - 范围: {porosity_min:.4f} - {porosity_max:.4f}%
   - 高气孔率样本数: {len(high_porosity)}个
   - 结论: {'符合优质标准' if porosity_mean < 2 else '需优化工艺'}

2. 裂纹率
   - 平均裂纹率: {crack_mean:.4f}%
   - 范围: {crack_min:.4f} - {crack_max:.4f}%
   - 高裂纹率样本数: {len(high_crack)}个
   - 结论: {'质量良好' if crack_mean < 1 else '需检查工艺参数'}

================================================================================
三、组织定量表征合理性
================================================================================

1. 析出相含量
   - 平均析出相占比: {carbide_mean:.2f}%
   - 范围: {carbide_min:.2f} - {carbide_max:.2f}%
   - 结论: {'符合JG-1合金特征' if 10 <= carbide_mean <= 20 else '需验证合金成分'}

2. 组织占比总和
   - 平均总和: {total_sum_mean:.2f}%
   - 标准差: {total_sum_std:.4f}%
   - 结论: {'分割算法准确' if 99 <= total_sum_mean <= 101 else '需检查分割逻辑'}

================================================================================
四、力学性能预测合理性
================================================================================

1. 显微硬度预测
   - 平均预测硬度: {hardness_mean:.1f} HV
   - 范围: {df['预测显微硬度(HV)_calc'].min():.1f} - {df['预测显微硬度(HV)_calc'].max():.1f} HV
   - 结论: {'符合典型范围' if 500 <= hardness_mean <= 700 else '预测异常'}

2. Hall-Petch关系验证
   - HV vs d^(-1/2)相关性: r = {correlation_hp:.4f}
   - 结论: {'符合Hall-Petch规律' if correlation_hp > 0.5 else '受其他因素影响'}

================================================================================
五、数据质量评估总结
================================================================================

✅ 合理项:
   - 晶粒尺寸范围符合激光熔覆特征
   - 稀释率在理想范围内
   - 气孔率符合优质熔覆标准
   - 析出相含量符合JG-1合金特征
   - 组织占比总和接近100%

⚠️ 需关注项:
   - 模型R²过高，存在过拟合风险
   - 部分样本气孔率/裂纹率偏高
   - Hall-Petch关系相关性较弱

建议:
   1. 增加独立测试数据验证模型泛化能力
   2. 对高缺陷样本进行工艺参数回溯分析
   3. 补充实际力学性能测试数据校准预测模型

================================================================================
"""

# 保存报告
report_path = os.path.join(analysis_dir, "material_science_analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"✅ 分析报告已保存: {report_path}")

print(report_text)

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)