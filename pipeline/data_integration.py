"""
阶段1: 实测数据整合脚本
读取显微硬度、磨损、EIS、XRD实测数据，与金相定量表征CSV合并
输出: analysis_output/完整实验数据汇总.csv
"""

import os
import re
import numpy as np
import pandas as pd
from docx import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 1. 显微硬度数据读取 =====================
def read_microhardness():
    """从docx文件读取显微硬度实测数据"""
    mh_dir = os.path.join(DATA_DIR, "microhardness-data")
    results = {}

    for power in ["900W", "1200W", "1500W", "1800W"]:
        docx_path = os.path.join(mh_dir, f"{power}.docx")
        if not os.path.exists(docx_path):
            print(f"  [WARN] {power}.docx not found")
            continue

        doc = Document(docx_path)
        hv_values = []
        for p in doc.paragraphs:
            text = p.text.strip()
            match = re.search(r'HV=\s*([\d.]+)', text)
            if match:
                hv_values.append(float(match.group(1)))

        if hv_values:
            results[power] = {
                "mh_mean_hv": np.mean(hv_values),
                "mh_std_hv": np.std(hv_values),
                "mh_min_hv": np.min(hv_values),
                "mh_max_hv": np.max(hv_values),
                "mh_count": len(hv_values),
                "mh_cv_pct": np.std(hv_values) / np.mean(hv_values) * 100,
            }
            print(f"  {power}: HV = {np.mean(hv_values):.1f} ± {np.std(hv_values):.1f} "
                  f"(n={len(hv_values)}, CV={np.std(hv_values)/np.mean(hv_values)*100:.1f}%)")

    return results


# ===================== 2. 磨损数据读取 =====================
def read_wear_data():
    """从txt文件读取摩擦磨损数据"""
    wear_dir = os.path.join(DATA_DIR, "wear-data")
    results = {}

    file_map = {
        "900W": "900W.txt",
        "1200W": "1200w.txt",
        "1500W": "1500w.txt",
        "1800W": "1800w.txt",
    }

    for power, filename in file_map.items():
        fpath = os.path.join(wear_dir, filename)
        if not os.path.exists(fpath):
            print(f"  [WARN] {filename} not found")
            continue

        try:
            with open(fpath, "r", encoding="gbk", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

        friction_values = []
        for line in lines:
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    fc = float(parts[1])
                    if t > 0 and fc > 0:
                        friction_values.append(fc)
                except ValueError:
                    continue

        if friction_values:
            n = len(friction_values)
            steady_start = int(n * 0.3)
            steady_friction = friction_values[steady_start:]

            results[power] = {
                "wear_friction_mean": np.mean(friction_values),
                "wear_friction_steady": np.mean(steady_friction),
                "wear_friction_std": np.std(steady_friction),
                "wear_friction_max": np.max(friction_values),
                "wear_data_points": n,
            }
            print(f"  {power}: friction = {np.mean(steady_friction):.4f} ± {np.std(steady_friction):.4f} "
                  f"(steady, n={len(steady_friction)})")

    return results


# ===================== 3. EIS数据读取 =====================
def read_eis_data():
    """从拟合数据txt文件读取电化学阻抗参数"""
    eis_dir = os.path.join(DATA_DIR, "electrochemical-impedance")
    results = {}

    for power in ["900", "1200", "1500", "1800"]:
        candidates = [
            f"{power}-EIS-拟合数据.txt",
            f"{power}-拟合数据.txt",
        ]
        fit_path = None
        for c in candidates:
            p = os.path.join(eis_dir, c)
            if os.path.exists(p):
                fit_path = p
                break
        if fit_path is None:
            print(f"  [WARN] {power} EIS拟合数据 not found")
            continue

        try:
            with open(fit_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            with open(fit_path, "r", encoding="gbk", errors="ignore") as f:
                lines = f.readlines()

        freq, z_real, z_imag, z_mag, theta = [], [], [], [], []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('"') or line.startswith("Pt"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                try:
                    freq.append(float(parts[4]))
                    z_real.append(float(parts[2]))
                    z_imag.append(float(parts[3]))
                    z_mag.append(float(parts[5]))
                    theta.append(float(parts[6]))
                except ValueError:
                    continue

        if freq:
            z_real_arr = np.array(z_real)
            z_imag_arr = np.array(z_imag)

            rct = np.max(z_real) - np.min(z_real)
            rs = np.min(z_real)
            z_max = np.max(z_mag)
            theta_min = np.min(theta)

            results[power] = {
                "eis_Rs_ohm": rs,
                "eis_Rct_ohm": rct,
                "eis_Z_max_ohm": z_max,
                "eis_theta_min_deg": theta_min,
                "eis_freq_range": f"{np.min(freq):.1e}-{np.max(freq):.1e}",
                "eis_data_points": len(freq),
            }
            print(f"  {power}: Rs={rs:.1f}Ω, Rct={rct:.1f}Ω, |Z|max={z_max:.1f}Ω")

    return results


# ===================== 4. XRD数据读取 =====================
def read_xrd_data():
    """从txt文件读取XRD数据，计算峰强度比"""
    xrd_dir = os.path.join(DATA_DIR, "xrd-data")
    results = {}

    file_map = {
        "900W": "1-900.txt",
        "1200W": "2-1200.txt",
        "1500W": "3-1500.txt",
        "1800W": "4-1800.txt",
    }

    for power, filename in file_map.items():
        fpath = os.path.join(xrd_dir, filename)
        if not os.path.exists(fpath):
            print(f"  [WARN] {filename} not found")
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            with open(fpath, "r", encoding="gbk", errors="ignore") as f:
                lines = f.readlines()

        two_theta, intensity = [], []
        for line in lines:
            line = line.strip()
            if not line or line == "?":
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue

        if two_theta:
            theta_arr = np.array(two_theta)
            int_arr = np.array(intensity)

            main_peak_idx = np.argmax(int_arr)
            main_peak_angle = theta_arr[main_peak_idx]
            main_peak_intensity = int_arr[main_peak_idx]

            peak_44_idx = np.argmin(np.abs(theta_arr - 44.0))
            peak_44_range = int_arr[max(0, peak_44_idx - 20):peak_44_idx + 20]
            peak_44_area = np.sum(peak_44_range) if len(peak_44_range) > 0 else 0

            results[power] = {
                "xrd_main_peak_2theta": main_peak_angle,
                "xrd_main_peak_intensity": main_peak_intensity,
                "xrd_peak_44_area": peak_44_area,
                "xrd_total_intensity": np.sum(int_arr),
                "xrd_data_points": len(two_theta),
            }
            print(f"  {power}: main peak={main_peak_angle:.2f}°, I={main_peak_intensity:.0f}")

    return results


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("阶段1: 实测数据整合")
    print("=" * 60)

    print("\n[1/4] 读取显微硬度数据...")
    mh_data = read_microhardness()

    print("\n[2/4] 读取磨损数据...")
    wear_data = read_wear_data()

    print("\n[3/4] 读取EIS数据...")
    eis_data = read_eis_data()

    print("\n[4/4] 读取XRD数据...")
    xrd_data = read_xrd_data()

    # 读取现有金相定量CSV
    csv_path = os.path.join(OUTPUT_DIR, "金相定量表征数据汇总.csv")
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] 未找到 {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"\n现有金相数据: {len(df)} 行, {len(df.columns)} 列")

    # 按功率匹配实测数据
    power_map = {"900": "900W", "1200": "1200W", "1500": "1500W", "1800": "1800W"}

    for col_name in ["mh_mean_hv", "mh_std_hv", "mh_min_hv", "mh_max_hv", "mh_count", "mh_cv_pct"]:
        df[col_name] = np.nan
    for col_name in ["wear_friction_mean", "wear_friction_steady", "wear_friction_std",
                      "wear_friction_max", "wear_data_points"]:
        df[col_name] = np.nan
    for col_name in ["eis_Rs_ohm", "eis_Rct_ohm", "eis_Z_max_ohm", "eis_theta_min_deg",
                      "eis_data_points"]:
        df[col_name] = np.nan
    df["eis_freq_range"] = ""
    for col_name in ["xrd_main_peak_2theta", "xrd_main_peak_intensity",
                      "xrd_peak_44_area", "xrd_total_intensity", "xrd_data_points"]:
        df[col_name] = np.nan

    for idx, row in df.iterrows():
        power_str = str(row["激光功率"]).replace("W", "")
        power_key = power_map.get(power_str, f"{power_str}W")

        if power_key in mh_data:
            for k, v in mh_data[power_key].items():
                df.at[idx, k] = v

        if power_key in wear_data:
            for k, v in wear_data[power_key].items():
                df.at[idx, k] = v

        if power_str in eis_data:
            for k, v in eis_data[power_str].items():
                df.at[idx, k] = v

        if power_key in xrd_data:
            for k, v in xrd_data[power_key].items():
                df.at[idx, k] = v

    # 整合论文中的工艺参数和化学成分数据
    print("\n[5/5] 整合论文中的工艺参数和化学成分数据...")
    
    # 添加工艺参数
    df['送粉速度(g/min)'] = 10
    df['扫描速度(mm/s)'] = 10
    
    # 添加JG-1铁基合金粉末化学成分
    df['JG-1_C_含量(wt.%)'] = 1.0
    df['JG-1_Cr_含量(wt.%)'] = 18.0
    df['JG-1_Si_含量(wt.%)'] = 0.8
    df['JG-1_Fe_含量(wt.%)'] = 63.2
    df['JG-1_Ni_含量(wt.%)'] = 12.0
    df['JG-1_Others_含量(wt.%)'] = 5.0
    
    # 添加Q355钢化学成分
    df['Q355_C_含量(wt.%)'] = 0.24
    df['Q355_Si_含量(wt.%)'] = 0.55
    df['Q355_Mn_含量(wt.%)'] = 1.60
    df['Q355_Cr_含量(wt.%)'] = 0.30
    df['Q355_Ni_含量(wt.%)'] = 0.30
    df['Q355_Cu_含量(wt.%)'] = 0.40
    df['Q355_S_含量(wt.%)'] = 0.035
    df['Q355_P_含量(wt.%)'] = 0.035
    
    # 添加Q355钢力学性能
    df['Q355_下屈服强度(MPa)'] = 355
    df['Q355_抗拉强度(MPa)'] = 550
    df['Q355_断后伸长率(%)'] = 21

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "完整实验数据汇总.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 已保存: {out_path}")
    print(f"     行数: {len(df)}, 列数: {len(df.columns)}")

    # 生成汇总报告
    report_path = os.path.join(OUTPUT_DIR, "数据整合报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 实测数据整合报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. 显微硬度实测数据\n\n")
        f.write("| 功率 | 均值(HV) | 标准差 | 范围 | 样本数 | CV(%) |\n")
        f.write("|------|---------|--------|------|--------|-------|\n")
        for power in ["900W", "1200W", "1500W", "1800W"]:
            if power in mh_data:
                d = mh_data[power]
                f.write(f"| {power} | {d['mh_mean_hv']:.1f} | {d['mh_std_hv']:.1f} | "
                        f"{d['mh_min_hv']:.1f}-{d['mh_max_hv']:.1f} | {d['mh_count']} | "
                        f"{d['mh_cv_pct']:.1f} |\n")

        f.write("\n## 2. 磨损数据\n\n")
        f.write("| 功率 | 稳态摩擦系数 | 摩擦系数标准差 | 数据点 |\n")
        f.write("|------|------------|--------------|--------|\n")
        for power in ["900W", "1200W", "1500W", "1800W"]:
            if power in wear_data:
                d = wear_data[power]
                f.write(f"| {power} | {d['wear_friction_steady']:.4f} | "
                        f"{d['wear_friction_std']:.4f} | {d['wear_data_points']} |\n")

        f.write("\n## 3. 电化学阻抗数据\n\n")
        f.write("| 功率 | Rs(Ω) | Rct(Ω) | |Z|max(Ω) | θmin(°) |\n")
        f.write("|------|-------|---------|----------|--------|\n")
        for power in ["900", "1200", "1500", "1800"]:
            if power in eis_data:
                d = eis_data[power]
                f.write(f"| {power}W | {d['eis_Rs_ohm']:.1f} | {d['eis_Rct_ohm']:.1f} | "
                        f"{d['eis_Z_max_ohm']:.1f} | {d['eis_theta_min_deg']:.1f} |\n")

        f.write("\n## 4. XRD数据\n\n")
        f.write("| 功率 | 主峰2θ(°) | 主峰强度 | 44°峰面积 |\n")
        f.write("|------|----------|---------|----------|\n")
        for power in ["900W", "1200W", "1500W", "1800W"]:
            if power in xrd_data:
                d = xrd_data[power]
                f.write(f"| {power} | {d['xrd_main_peak_2theta']:.2f} | "
                        f"{d['xrd_main_peak_intensity']:.0f} | {d['xrd_peak_44_area']:.0f} |\n")

        f.write("\n## 5. 与物理公式预测对比\n\n")
        f.write("| 功率 | 实测硬度(HV) | 物理公式预测 | 偏差 |\n")
        f.write("|------|------------|------------|------|\n")

        physics_predictions = {
            "900W": 224.2,
            "1200W": 243.4,
            "1500W": 288.6,
            "1800W": 385.2,
        }
        for power in ["900W", "1200W", "1500W", "1800W"]:
            if power in mh_data:
                actual = mh_data[power]["mh_mean_hv"]
                f.write(f"| {power} | {actual:.1f} | (见fix_issues.py) | — |\n")

    print(f"[OK] 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
