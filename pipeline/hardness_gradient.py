"""
显微硬度梯度处理模块
处理从Q355基体到熔覆层JG-1的硬度梯度数据
"""

import os
import re
import numpy as np
import pandas as pd
from docx import Document


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MH_DIR = os.path.join(DATA_DIR, "microhardness-data")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")


def read_hardness_gradient():
    """
    读取显微硬度梯度数据，返回每个功率的10个测量点的硬度值
    
    返回:
        dict: {power: [hv_01, hv_02, ..., hv_10]}
    """
    results = {}
    
    for power in ["900W", "1200W", "1500W", "1800W"]:
        docx_path = os.path.join(MH_DIR, f"{power}.docx")
        if not os.path.exists(docx_path):
            print(f"  [WARN] {power}.docx not found")
            continue
        
        doc = Document(docx_path)
        hv_values = []
        
        for p in doc.paragraphs:
            text = p.text.strip()
            # 匹配 HV= xxx.x 格式
            match = re.search(r'HV=\s*([\d.]+)', text)
            if match:
                hv_values.append(float(match.group(1)))
        
        if len(hv_values) == 10:
            results[power] = hv_values
            print(f"  {power}: 读取 {len(hv_values)} 个硬度点")
        else:
            print(f"  [WARN] {power}: 只读取到 {len(hv_values)} 个硬度点 (期望10个)")
    
    return results


def calculate_position_averages(gradient_data):
    """
    计算每个位置的平均硬度（跨4个功率）
    
    参数:
        gradient_data: {power: [hv_01, ..., hv_10]}
    
    返回:
        dict: {
            'position_stats': DataFrame (位置统计),
            'gradient_summary': DataFrame (梯度汇总)
        }
    """
    powers = sorted(gradient_data.keys())
    n_positions = 10
    
    # 创建位置硬度矩阵 (4功率 x 10位置)
    matrix = np.zeros((len(powers), n_positions))
    for i, power in enumerate(powers):
        matrix[i, :] = gradient_data[power]
    
    # 计算每个位置的统计量
    position_stats = []
    for pos in range(n_positions):
        pos_values = matrix[:, pos]
        stats = {
            '位置编号': f'{pos+1:02d}',
            '位置说明': get_position_description(pos),
            '900W': gradient_data.get('900W', [np.nan]*10)[pos],
            '1200W': gradient_data.get('1200W', [np.nan]*10)[pos],
            '1500W': gradient_data.get('1500W', [np.nan]*10)[pos],
            '1800W': gradient_data.get('1800W', [np.nan]*10)[pos],
            '平均硬度(HV)': np.mean(pos_values),
            '标准差(HV)': np.std(pos_values),
            '最小值(HV)': np.min(pos_values),
            '最大值(HV)': np.max(pos_values),
        }
        position_stats.append(stats)
    
    df_position = pd.DataFrame(position_stats)
    
    # 计算梯度汇总
    gradient_summary = {
        '区域': ['熔覆层(位置1-5)', '基体(位置6-10)', '整体(位置1-10)'],
        '平均硬度(HV)': [
            np.mean(matrix[:, :5]),
            np.mean(matrix[:, 5:]),
            np.mean(matrix)
        ],
        '标准差(HV)': [
            np.std(matrix[:, :5]),
            np.std(matrix[:, 5:]),
            np.std(matrix)
        ],
        '硬度范围(HV)': [
            f"{np.min(matrix[:, :5]):.1f}-{np.max(matrix[:, :5]):.1f}",
            f"{np.min(matrix[:, 5:]):.1f}-{np.max(matrix[:, 5:]):.1f}",
            f"{np.min(matrix):.1f}-{np.max(matrix):.1f}"
        ]
    }
    df_gradient = pd.DataFrame(gradient_summary)
    
    return df_position, df_gradient


def get_position_description(pos_index):
    """
    根据位置索引返回位置描述
    位置01-05: 熔覆层区域 (从熔覆层顶部到底部)
    位置06-10: 基体区域 (从界面到基体深处)
    """
    descriptions = {
        0: "熔覆层顶部",
        1: "熔覆层中上部",
        2: "熔覆层中部",
        3: "熔覆层中下部",
        4: "熔覆层底部/界面附近",
        5: "界面附近/基体顶部",
        6: "基体上部",
        7: "基体中部",
        8: "基体中下部",
        9: "基体底部"
    }
    return descriptions.get(pos_index, f"位置{pos_index+1}")


def create_gradient_csv(gradient_data):
    """
    创建梯度硬度CSV文件，包含所有功率的原始数据和位置统计
    """
    powers = sorted(gradient_data.keys())
    
    # 创建原始数据DataFrame
    rows = []
    for power in powers:
        hv_values = gradient_data[power]
        for pos_idx, hv in enumerate(hv_values):
            rows.append({
                '激光功率': power,
                '测量位置': f'{pos_idx+1:02d}',
                '位置说明': get_position_description(pos_idx),
                'HV值': hv,
                'D1': np.nan,  # 需要从Word文档重新读取
                'D2': np.nan,  # 需要从Word文档重新读取
            })
    
    df_raw = pd.DataFrame(rows)
    
    # 计算位置平均值
    position_avg = []
    for pos_idx in range(10):
        pos_values = [gradient_data[p][pos_idx] for p in powers if p in gradient_data]
        position_avg.append({
            '测量位置': f'{pos_idx+1:02d}',
            '位置说明': get_position_description(pos_idx),
            '平均硬度(HV)': np.mean(pos_values),
            '标准差(HV)': np.std(pos_values),
            '900W': gradient_data.get('900W', [np.nan]*10)[pos_idx],
            '1200W': gradient_data.get('1200W', [np.nan]*10)[pos_idx],
            '1500W': gradient_data.get('1500W', [np.nan]*10)[pos_idx],
            '1800W': gradient_data.get('1800W', [np.nan]*10)[pos_idx],
        })
    
    df_position_avg = pd.DataFrame(position_avg)
    
    return df_raw, df_position_avg


def save_gradient_data(df_raw, df_position_avg, df_gradient):
    """
    保存梯度硬度数据到CSV文件
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存原始梯度数据
    raw_path = os.path.join(OUTPUT_DIR, "显微硬度原始梯度数据.csv")
    df_raw.to_csv(raw_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] 原始梯度数据: {raw_path}")
    
    # 保存位置平均数据
    avg_path = os.path.join(OUTPUT_DIR, "显微硬度位置平均数据.csv")
    df_position_avg.to_csv(avg_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] 位置平均数据: {avg_path}")
    
    # 保存梯度汇总
    grad_path = os.path.join(OUTPUT_DIR, "显微硬度梯度汇总.csv")
    df_gradient.to_csv(grad_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] 梯度汇总: {grad_path}")
    
    return raw_path, avg_path, grad_path


def update_integrated_data(gradient_data):
    """
    更新整合后的实验数据，添加梯度硬度信息
    """
    csv_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] 完整实验数据汇总.csv 不存在")
        return
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 计算每个功率的梯度统计
    for power in ["900W", "1200W", "1500W", "1800W"]:
        if power in gradient_data:
            hv_values = gradient_data[power]
            
            # 熔覆层硬度 (位置1-5)
            cladding_hv = np.mean(hv_values[:5])
            cladding_std = np.std(hv_values[:5])
            
            # 基体硬度 (位置6-10)
            substrate_hv = np.mean(hv_values[5:])
            substrate_std = np.std(hv_values[5:])
            
            # 整体统计
            mean_hv = np.mean(hv_values)
            std_hv = np.std(hv_values)
            
            # 更新对应功率的行
            mask = df['激光功率'] == power
            if mask.any():
                df.loc[mask, 'mh_mean_hv'] = mean_hv
                df.loc[mask, 'mh_std_hv'] = std_hv
                df.loc[mask, 'mh_min_hv'] = np.min(hv_values)
                df.loc[mask, 'mh_max_hv'] = np.max(hv_values)
                df.loc[mask, 'mh_cladding_hv'] = cladding_hv
                df.loc[mask, 'mh_cladding_std'] = cladding_std
                df.loc[mask, 'mh_substrate_hv'] = substrate_hv
                df.loc[mask, 'mh_substrate_std'] = substrate_std
                df.loc[mask, 'mh_gradient_range'] = cladding_hv - substrate_hv
                
                print(f"  {power}: 整体={mean_hv:.1f}±{std_hv:.1f}, "
                      f"熔覆层={cladding_hv:.1f}±{cladding_std:.1f}, "
                      f"基体={substrate_hv:.1f}±{substrate_std:.1f}")
    
    # 保存更新后的数据
    updated_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df.to_csv(updated_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] 已更新整合数据: {updated_path}")
    
    return updated_path


def create_ml_training_data(gradient_data):
    """
    创建用于ML训练的梯度硬度数据
    每个功率的每个位置作为一个样本
    """
    rows = []
    
    for power, hv_values in gradient_data.items():
        power_num = int(power.replace('W', ''))
        
        for pos_idx, hv in enumerate(hv_values):
            # 位置归一化 (0-1, 从熔覆层到基体)
            position_normalized = pos_idx / 9.0
            
            # 区域标签
            if pos_idx < 5:
                region = "cladding"
                region_label = 1  # 熔覆层
            else:
                region = "substrate"
                region_label = 0  # 基体
            
            rows.append({
                '激光功率(W)': power_num,
                '测量位置': pos_idx + 1,
                '位置归一化': position_normalized,
                '区域': region,
                '区域标签': region_label,
                '显微硬度(HV)': hv,
            })
    
    df_ml = pd.DataFrame(rows)
    
    # 保存ML训练数据
    ml_path = os.path.join(OUTPUT_DIR, "梯度硬度ML训练数据.csv")
    df_ml.to_csv(ml_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] ML训练数据: {ml_path}")
    
    return ml_path


def main():
    """主函数"""
    print("=" * 60)
    print("显微硬度梯度处理")
    print("=" * 60)
    
    print("\n[1/6] 读取显微硬度梯度数据...")
    gradient_data = read_hardness_gradient()
    
    if not gradient_data:
        print("  [ERROR] 未读取到任何硬度数据")
        return
    
    print("\n[2/6] 计算位置统计...")
    df_position, df_gradient = calculate_position_averages(gradient_data)
    
    print("\n位置统计:")
    print(df_position[['位置编号', '位置说明', '平均硬度(HV)', '标准差(HV)']].to_string(index=False))
    
    print("\n梯度汇总:")
    print(df_gradient.to_string(index=False))
    
    print("\n[3/6] 保存梯度数据...")
    df_raw, df_position_avg = create_gradient_csv(gradient_data)
    save_gradient_data(df_raw, df_position_avg, df_gradient)
    
    print("\n[4/6] 更新整合实验数据...")
    update_integrated_data(gradient_data)
    
    print("\n[5/6] 创建ML训练数据...")
    create_ml_training_data(gradient_data)
    
    print("\n[6/6] 生成分析报告...")
    generate_gradient_report(df_position, df_gradient, gradient_data)
    
    print("\n" + "=" * 60)
    print("硬度梯度处理完成!")
    print("=" * 60)


def generate_gradient_report(df_position, df_gradient, gradient_data):
    """生成梯度分析报告"""
    report_path = os.path.join(OUTPUT_DIR, "显微硬度梯度分析报告.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 显微硬度梯度分析报告\n\n")
        
        f.write("## 1. 测量说明\n\n")
        f.write("- 测量方式: 从Q355基体到熔覆层JG-1从下往上测量\n")
        f.write("- 测量点数: 每个功率10个测量点\n")
        f.write("- 位置分布: 位置01-05为熔覆层区域，位置06-10为基体区域\n\n")
        
        f.write("## 2. 位置统计\n\n")
        f.write(df_position[['位置编号', '位置说明', '平均硬度(HV)', '标准差(HV)']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. 梯度汇总\n\n")
        f.write(df_gradient.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. 各功率硬度分布\n\n")
        for power in ["900W", "1200W", "1500W", "1800W"]:
            if power in gradient_data:
                hv = gradient_data[power]
                f.write(f"### {power}\n\n")
                f.write("| 位置 | 硬度(HV) | 区域 |\n")
                f.write("|------|----------|------|\n")
                for i, h in enumerate(hv):
                    region = "熔覆层" if i < 5 else "基体"
                    f.write(f"| {i+1:02d} | {h:.1f} | {region} |\n")
                f.write("\n")
        
        f.write("## 5. 硬度梯度特征\n\n")
        cladding_avg = np.mean([gradient_data[p][:5] for p in gradient_data])
        substrate_avg = np.mean([gradient_data[p][5:] for p in gradient_data])
        f.write(f"- 熔覆层平均硬度: {cladding_avg:.1f} HV\n")
        f.write(f"- 基体平均硬度: {substrate_avg:.1f} HV\n")
        f.write(f"- 硬度梯度差: {cladding_avg - substrate_avg:.1f} HV\n")
        f.write(f"- 硬度提升倍数: {cladding_avg/substrate_avg:.2f}x\n")
    
    print(f"  [OK] 分析报告: {report_path}")


if __name__ == "__main__":
    main()
