"""
清理analysis_output目录，统一数据格式
"""

import os
import shutil
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
BACKUP_DIR = os.path.join(OUTPUT_DIR, "_backup_old_files")


def analyze_files():
    """分析文件结构"""
    print("分析analysis_output目录结构...")
    
    files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    
    csv_files = [f for f in files if f.endswith('.csv')]
    xlsx_files = [f for f in files if f.endswith('.xlsx')]
    pkl_files = [f for f in files if f.endswith('.pkl')]
    md_files = [f for f in files if f.endswith('.md')]
    
    print(f"\n文件统计:")
    print(f"  CSV文件: {len(csv_files)}")
    print(f"  Excel文件: {len(xlsx_files)}")
    print(f"  模型文件: {len(pkl_files)}")
    print(f"  Markdown文件: {len(md_files)}")
    
    return csv_files, xlsx_files, pkl_files, md_files


def identify_redundant_files(csv_files):
    """识别冗余文件"""
    # 基础数据文件（需要保留）
    keep_files = [
        "完整实验数据汇总.csv",  # 主数据文件（79列，最完整）
        "梯度硬度ML训练数据.csv",  # 梯度硬度训练数据
        "显微硬度位置平均数据.csv",  # 位置平均数据
        "univariate_correlation_results.csv",  # 单变量分析结果
        "pearson_correlation_matrix.csv",  # 相关性矩阵
    ]
    
    # 可删除的冗余文件
    delete_files = [
        "金相定量表征数据汇总.csv",  # 被完整实验数据汇总包含
        "金相定量表征数据汇总_修正版.csv",  # 冗余
        "完整实验数据汇总_修正版.csv",  # 冗余
        "L25_模拟实验数据.csv",  # 模拟数据
        "L25_正交实验数据.csv",  # 模拟数据
        "simulated_orthogonal_data.csv",  # 模拟数据
        "显微硬度原始梯度数据.csv",  # 原始数据，已被位置平均替代
        "显微硬度梯度汇总.csv",  # 简单汇总
        "显微硬度梯度分析报告.md",  # 报告
    ]
    
    # 可选保留的文件
    optional_files = [
        "显微硬度位置平均数据.csv",  # 如果需要位置信息
    ]
    
    return keep_files, delete_files, optional_files


def cleanup_files(delete_files):
    """清理冗余文件"""
    print("\n清理冗余文件...")
    
    # 创建备份目录
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    deleted = []
    for f in delete_files:
        src = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(src):
            dst = os.path.join(BACKUP_DIR, f)
            shutil.move(src, dst)
            deleted.append(f)
            print(f"  移动: {f} -> _backup_old_files/")
    
    return deleted


def cleanup_xlsx_files():
    """清理Excel文件（保留CSV版本）"""
    print("\n清理Excel文件...")
    
    xlsx_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.xlsx')]
    
    moved = []
    for f in xlsx_files:
        src = os.path.join(OUTPUT_DIR, f)
        dst = os.path.join(BACKUP_DIR, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved.append(f)
            print(f"  移动: {f} -> _backup_old_files/")
    
    return moved


def rename_files():
    """重命名文件以统一格式"""
    print("\n重命名文件...")
    
    renames = {
        "完整实验数据汇总.csv": "data_full.csv",
        "梯度硬度ML训练数据.csv": "data_gradient_ml.csv",
        "显微硬度位置平均数据.csv": "data_hardness_position.csv",
        "pearson_correlation_matrix.csv": "corr_pearson.csv",
        "univariate_correlation_results.csv": "analysis_univariate.csv",
    }
    
    renamed = []
    for old_name, new_name in renames.items():
        src = os.path.join(OUTPUT_DIR, old_name)
        dst = os.path.join(OUTPUT_DIR, new_name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            renamed.append(f"{old_name} -> {new_name}")
            print(f"  {old_name} -> {new_name}")
    
    return renamed


def create_readme():
    """创建文件说明"""
    readme_path = os.path.join(OUTPUT_DIR, "README_DATA_FILES.md")
    
    content = """# analysis_output 数据文件说明

## 数据文件 (CSV)

| 文件名 | 说明 | 行数 | 列数 |
|--------|------|------|------|
| data_full.csv | 完整实验数据汇总（主数据文件） | 50 | 79 |
| data_gradient_ml.csv | 梯度硬度ML训练数据 | 40 | 6 |
| data_hardness_position.csv | 显微硬度位置平均数据 | 10 | 8 |
| corr_pearson.csv | Pearson相关性矩阵 | 8 | 9 |
| analysis_univariate.csv | 单变量分析结果 | 19 | 11 |

## 模型文件 (PKL)

| 文件名 | 说明 |
|--------|------|
| property_models.pkl | 性能预测模型（GBR/RFR/GPR） |
| power_response_models.pkl | 功率响应面模型 |
| hardness_gradient_model.pkl | 硬度梯度预测模型 |
| optimization_results.pkl | 优化结果 |

## 报告文件 (MD)

| 文件名 | 说明 |
|--------|------|
| FINAL_SUMMARY.md | 最终总结报告 |
| 实验验证方案.md | 验证实验设计 |
| 阶段2_响应面建模报告.md | 响应面分析报告 |
| 阶段3_性能模型报告.md | 性能模型报告 |
| 阶段4_优化汇总报告.md | 优化结果报告 |

## 使用说明

1. **训练ML模型**: 使用 `data_full.csv` 或 `data_gradient_ml.csv`
2. **分析特征关系**: 使用 `analysis_univariate.csv` 和 `corr_pearson.csv`
3. **预测新数据**: 加载对应的 `.pkl` 模型文件

## 备份文件

被移动到 `_backup_old_files/` 的文件是冗余或旧版本数据。
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n创建文件说明: {readme_path}")


def main():
    print("=" * 60)
    print("清理analysis_output目录")
    print("=" * 60)
    
    # 分析文件
    csv_files, xlsx_files, pkl_files, md_files = analyze_files()
    
    # 识别冗余文件
    keep_files, delete_files, optional_files = identify_redundant_files(csv_files)
    
    print(f"\n保留文件: {keep_files}")
    print(f"删除文件: {delete_files}")
    
    # 执行清理
    deleted = cleanup_files(delete_files)
    moved_xlsx = cleanup_xlsx_files()
    renamed = rename_files()
    create_readme()
    
    # 显示最终结构
    print("\n" + "=" * 60)
    print("清理后文件结构")
    print("=" * 60)
    
    files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    for f in sorted(files):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")
    
    print(f"\n删除/移动文件数: {len(deleted) + len(moved_xlsx)}")
    print(f"重命名文件数: {len(renamed)}")


if __name__ == "__main__":
    main()
