"""
正交实验设计脚本
生成L25(5³)正交表 + 极差分析 + 模拟数据生成

用法:
    python scripts/orthogonal_design.py              # 生成正交表
    python scripts/orthogonal_design.py --simulate   # 生成模拟数据
    python scripts/orthogonal_design.py --analyze    # 极差分析（需有结果数据）
"""

import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== L25(5³) 正交表 =====================
# 标准正交表（5水平3因素）
ORTHOGONAL_TABLE = [
    [1, 1, 1, 1], [1, 2, 2, 2], [1, 3, 3, 3], [1, 4, 4, 4], [1, 5, 5, 5],
    [2, 1, 2, 3], [2, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 1], [2, 5, 1, 2],
    [3, 1, 3, 5], [3, 2, 4, 1], [3, 3, 5, 2], [3, 4, 1, 3], [3, 5, 2, 4],
    [4, 1, 4, 2], [4, 2, 5, 3], [4, 3, 1, 4], [4, 4, 2, 5], [4, 5, 3, 1],
    [5, 1, 5, 4], [5, 2, 1, 5], [5, 3, 2, 1], [5, 4, 3, 2], [5, 5, 4, 3],
]

# 因素水平定义
FACTORS = {
    "激光功率 P(W)": [700, 900, 1100, 1500, 1800],
    "扫描速度 Vs(mm/min)": [200, 250, 300, 350, 400],
    "送粉速率 Vf(g/min)": [6, 8, 10, 12, 14],
}


def generate_orthogonal_table():
    """生成L25(5³)正交实验设计表"""
    print("=" * 70)
    print("L25(5³) 正交实验设计表")
    print("=" * 70)

    factor_names = list(FACTORS.keys())
    factor_values = list(FACTORS.values())

    rows = []
    for i, row in enumerate(ORTHOGONAL_TABLE):
        exp_id = i + 1
        p_level, vs_level, vf_level = row[0], row[1], row[2]
        p_val = factor_values[0][p_level - 1]
        vs_val = factor_values[1][vs_level - 1]
        vf_val = factor_values[2][vf_level - 1]

        rows.append({
            "实验号": exp_id,
            "P水平": p_level,
            "Vs水平": vs_level,
            "Vf水平": vf_level,
            "P(W)": p_val,
            "Vs(mm/min)": vs_val,
            "Vf(g/min)": vf_val,
        })

    df = pd.DataFrame(rows)

    print("\n正交表:")
    print(df.to_string(index=False))

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "L25_正交实验设计.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 已保存: {out_path}")

    # 打印水平分布验证
    print("\n水平分布验证:")
    for col in ["P水平", "Vs水平", "Vf水平"]:
        counts = df[col].value_counts().sort_index()
        print(f"  {col}: {dict(counts)} (每水平{counts.min()}次)")

    return df


def range_analysis(results_df,指标列):
    """极差分析（ANOVA简化版）

    参数:
        results_df: 包含正交表+实验结果的DataFrame
        指标列: 要分析的指标列名
    """
    print(f"\n{'='*70}")
    print(f"极差分析: {指标列}")
    print(f"{'='*70}")

    factor_cols = ["P水平", "Vs水平", "Vf水平"]
    factor_names = ["激光功率 P", "扫描速度 Vs", "送粉速率 Vf"]
    levels = [1, 2, 3, 4, 5]

    means = {}
    for fcol, fname in zip(factor_cols, factor_names):
        level_means = {}
        for lv in levels:
            mask = results_df[fcol] == lv
            if mask.sum() > 0:
                level_means[lv] = results_df.loc[mask, 指标列].mean()
            else:
                level_means[lv] = 0
        means[fname] = level_means

        k_max = max(level_means.values())
        k_min = min(level_means.values())
        R = k_max - k_min
        print(f"\n  {fname}:")
        for lv, val in level_means.items():
            print(f"    水平{lv}: {val:.4f}")
        print(f"    极差R = {k_max:.4f} - {k_min:.4f} = {R:.4f}")

    # 排序
    R_values = {}
    for fname in factor_names:
        vals = list(means[fname].values())
        R_values[fname] = max(vals) - min(vals)

    sorted_factors = sorted(R_values.items(), key=lambda x: -x[1])
    print(f"\n  影响主次: {' > '.join([f[0] for f in sorted_factors])}")
    print(f"  极差值:   {' > '.join([f'{f[0]}={f[1]:.4f}' for f in sorted_factors])}")

    return means, R_values


def simulate_and_analyze():
    """生成模拟数据并做极差分析"""
    np.random.seed(42)

    df = generate_orthogonal_table()

    # 模拟三个指标
    df["稀释率(%)"] = 0.0
    df["宽高比(W/H)"] = 0.0
    df["硬度(HV)"] = 0.0

    for idx, row in df.iterrows():
        p, vs, vf = row["P(W)"], row["Vs(mm/min)"], row["Vf(g/min)"]

        # 模拟稀释率：功率正相关，速度负相关
        eta = 0.25 + 0.0001 * (p - 700) - 0.0003 * (vs - 200) + 0.02 * (vf - 6)
        eta += np.random.normal(0, 0.03)
        df.at[idx, "稀释率(%)"] = round(max(0.1, min(0.6, eta)), 3)

        # 模拟宽高比：速度负相关，送粉正相关
        wh = 2.0 - 0.005 * (vs - 200) + 0.3 * (vf - 6) + 0.001 * (p - 700)
        wh += np.random.normal(0, 0.2)
        df.at[idx, "宽高比(W/H)"] = round(max(1.0, min(5.0, wh)), 2)

        # 模拟硬度：功率负相关（与论文一致），速度正相关
        hv = 650 - 0.15 * (p - 700) + 0.3 * (vs - 200) - 5 * (vf - 6)
        hv += np.random.normal(0, 20)
        df.at[idx, "硬度(HV)"] = round(max(400, min(750, hv)), 1)

    # 保存模拟数据
    out_path = os.path.join(OUTPUT_DIR, "L25_模拟实验结果.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 模拟数据已保存: {out_path}")

    # 极差分析
    for indicator in ["稀释率(%)", "宽高比(W/H)", "硬度(HV)"]:
        range_analysis(df, indicator)

    return df


def analyze_real_data():
    """对真实实验数据做极差分析"""
    csv_path = os.path.join(OUTPUT_DIR, "L25_实验结果.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] 未找到 {csv_path}")
        print("请先完成实验并将结果保存为CSV格式")
        print("CSV需包含列: P水平, Vs水平, Vf水平, 稀释率(%), 宽高比(W/H), 硬度(HV)")
        return

    df = pd.read_csv(csv_path)
    print(f"加载数据: {len(df)} 行")

    for indicator in ["稀释率(%)", "宽高比(W/H)", "硬度(HV)"]:
        if indicator in df.columns:
            range_analysis(df, indicator)
        else:
            print(f"[WARN] 列 {indicator} 不存在")


def main():
    if "--simulate" in sys.argv:
        simulate_and_analyze()
    elif "--analyze" in sys.argv:
        analyze_real_data()
    else:
        generate_orthogonal_table()


if __name__ == "__main__":
    main()
