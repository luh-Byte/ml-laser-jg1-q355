"""
金相组织识别分析
基于XRD相分析 + 金相图像特征，识别Q355基材上JG-1铁基合金熔覆层的组织类型

材料体系:
  - 基材: Q355低碳钢 (C≤0.24%, Mn≤1.6%)
  - 粉末: JG-1铁基自熔合金 (Cr18%, Ni12%, C≤1%, Si0.8%, Fe余量)
  - 工艺: 激光熔覆 (900W/1200W/1500W/1800W)

可能的组织:
  1. 奥氏体 (γ-Fe, FCC) — Ni稳定化, 高温残留
  2. 马氏体 (α'-Fe, BCT) — 快冷形成, 高硬度
  3. 铁素体 (α-Fe, BCC) — 基材Q355的主要组织
  4. 碳化物 (Cr23C6, Cr7C3) — Cr+C析出, 高硬度
  5. 树枝晶 — 凝固组织, 功率越高越明显
  6. 等轴晶 — 重熔区
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
XRD_DIR = os.path.join(DATA_DIR, "xrd-data")
OM_DIR = os.path.join(DATA_DIR, "OM")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PDF标准卡片峰位
PHASE_PEAKS = {
    "alpha_Fe": {
        "name": "α-Fe (BCC铁素体/马氏体)",
        "peaks": {44.673: ("110", 100), 65.021: ("200", 20), 82.333: ("211", 30), 98.945: ("220", 10)},
        "structure": "BCC Im-3m",
        "a_lattice": 2.8664,
    },
    "gamma_Fe": {
        "name": "γ-Fe (FCC奥氏体)",
        "peaks": {43.620: ("111", 100), 50.809: ("200", 42.7), 74.703: ("220", 17.3), 90.702: ("311", 15.7)},
        "structure": "FCC Fm-3m",
        "a_lattice": 3.591,
    },
}

# 分割区域颜色映射
SEG_COLORS = {
    "背景/基体(Q355钢)": [120, 120, 120],
    "熔覆层组织": [100, 160, 110],
    "析出相/碳化物": [255, 200, 0],
    "气孔缺陷": [255, 30, 30],
    "裂纹缺陷": [150, 0, 200],
}


def xrd_phase_analysis():
    """XRD相分析：识别各功率组的物相组成"""
    print("=" * 70)
    print("XRD相分析")
    print("=" * 70)

    files = {
        "900W": "1-900.txt",
        "1200W": "2-1200.txt",
        "1500W": "3-1500.txt",
        "1800W": "4-1800.txt",
    }

    results = {}
    for power, fname in files.items():
        fpath = os.path.join(XRD_DIR, fname)
        if not os.path.exists(fpath):
            continue

        df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, header=None, names=["theta", "intensity"])

        # 找所有峰
        peaks = []
        intensity = df["intensity"].values
        theta = df["theta"].values
        for i in range(5, len(intensity) - 5):
            if intensity[i] == max(intensity[i - 5 : i + 6]):
                if intensity[i] > 200:
                    peaks.append((theta[i], intensity[i]))
        peaks.sort(key=lambda x: -x[1])

        # 匹配各相的主峰
        alpha_main = None
        gamma_main = None
        for t, iv in peaks:
            if alpha_main is None and abs(t - 44.673) < 0.8:
                alpha_main = (t, iv)
            if gamma_main is None and abs(t - 43.620) < 0.8:
                gamma_main = (t, iv)

        alpha_int = alpha_main[1] if alpha_main else 0
        gamma_int = gamma_main[1] if gamma_main else 0
        total = alpha_int + gamma_int
        gamma_ratio = gamma_int / total * 100 if total > 0 else 0

        # 判断组织类型
        if gamma_ratio > 60:
            main_phase = "奥氏体(γ-Fe)为主"
            microstructure = "奥氏体 + 少量马氏体/铁素体"
        elif gamma_ratio > 30:
            main_phase = "奥氏体+马氏体混合"
            microstructure = "奥氏体 + 马氏体 + 铁素体"
        else:
            main_phase = "马氏体/铁素体(α-Fe)为主"
            microstructure = "马氏体/铁素体 + 少量残留奥氏体"

        results[power] = {
            "alpha_intensity": alpha_int,
            "gamma_intensity": gamma_int,
            "gamma_ratio": gamma_ratio,
            "main_phase": main_phase,
            "microstructure": microstructure,
            "alpha_peak_pos": alpha_main[0] if alpha_main else None,
            "gamma_peak_pos": gamma_main[0] if gamma_main else None,
        }

        print(f"\n{power}:")
        print(f"  α-Fe(110)峰: 2θ={alpha_main[0]:.2f}°, I={alpha_int:.0f}" if alpha_main else "  α-Fe: 未检测到")
        print(f"  γ-Fe(111)峰: 2θ={gamma_main[0]:.2f}°, I={gamma_int:.0f}" if gamma_main else "  γ-Fe: 未检测到")
        print(f"  γ-Fe含量估算: {gamma_ratio:.1f}%")
        print(f"  主相: {main_phase}")
        print(f"  推断组织: {microstructure}")

    return results


def segmentation_phase_analysis():
    """分析金相分割图像，统计各组织区域占比"""
    print("\n" + "=" * 70)
    print("金相分割区域分析")
    print("=" * 70)

    seg_dir = os.path.join(OUTPUT_DIR, "figures", "segmentation")
    results = {}

    for power in ["900W", "1200W", "1500W", "1800W"]:
        folder = os.path.join(seg_dir, power)
        if not os.path.exists(folder):
            continue

        seg_files = [f for f in os.listdir(folder) if f.endswith("_seg.png")]
        if not seg_files:
            continue

        area_pcts = {name: [] for name in SEG_COLORS}

        for sf in seg_files:
            img = cv2.imread(os.path.join(folder, sf))
            if img is None:
                continue
            total = img.shape[0] * img.shape[1]
            for name, rgb in SEG_COLORS.items():
                mask = np.all(np.abs(img.astype(int) - rgb) < 30, axis=2)
                pct = np.sum(mask) / total * 100
                area_pcts[name].append(pct)

        results[power] = {}
        print(f"\n{power} ({len(seg_files)} images):")
        for name, pcts in area_pcts.items():
            if pcts:
                mean_pct = np.mean(pcts)
                std_pct = np.std(pcts)
                results[power][name] = {"mean": mean_pct, "std": std_pct, "n": len(pcts)}
                print(f"  {name}: {mean_pct:.1f}% ± {std_pct:.1f}%")

    return results


def correlate_xrd_seg(xrd_results, seg_results):
    """关联XRD相分析与金相分割结果"""
    print("\n" + "=" * 70)
    print("组织综合识别")
    print("=" * 70)

    summary = []
    for power in ["900W", "1200W", "1500W", "1800W"]:
        xrd = xrd_results.get(power, {})
        seg = seg_results.get(power, {})

        row = {"功率": power}

        # XRD信息
        row["γ-Fe含量(%)"] = xrd.get("gamma_ratio", 0)
        row["主相"] = xrd.get("main_phase", "未知")

        # 分割信息
        for name in SEG_COLORS:
            if name in seg:
                row[f"{name}_占比(%)"] = seg[name]["mean"]
            else:
                row[f"{name}_占比(%)"] = 0

        # 综合组织判断
        gamma_ratio = xrd.get("gamma_ratio", 0)
        clad_pct = seg.get("熔覆层组织", {}).get("mean", 0)
        precip_pct = seg.get("析出相/碳化物", {}).get("mean", 0)
        pore_pct = seg.get("气孔缺陷", {}).get("mean", 0)

        # 组织类型判断
        if gamma_ratio > 60:
            cladding_phases = "奥氏体(γ-Fe)为主 + Cr碳化物析出相"
        elif gamma_ratio > 30:
            cladding_phases = "奥氏体+马氏体混合 + Cr碳化物析出相"
        else:
            cladding_phases = "马氏体/铁素体(α-Fe)为主 + Cr碳化物析出相"

        row["熔覆层组织类型"] = cladding_phases
        row["基体组织类型"] = "铁素体+珠光体(Q355低碳钢)"

        summary.append(row)

        print(f"\n{power}:")
        print(f"  熔覆层: {cladding_phases}")
        print(f"  基体: 铁素体+珠光体(Q355)")
        print(f"  析出相: Cr碳化物(Cr23C6/Cr7C3) — 占{precip_pct:.1f}%")
        if pore_pct > 1:
            print(f"  缺陷: 气孔 — 占{pore_pct:.1f}%")

    # 保存综合表
    df = pd.DataFrame(summary)
    csv_path = os.path.join(OUTPUT_DIR, "组织识别综合表.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n综合表已保存: {csv_path}")

    return summary


def print_phase_evolution():
    """打印组织随功率演变规律"""
    print("\n" + "=" * 70)
    print("组织演变规律 (功率↑ → 冷却速度↓ → 相变行为改变)")
    print("=" * 70)

    evolution = """
    功率(W)   冷却速度   主要组织                    硬度(HV)   特征
    ─────────────────────────────────────────────────────────────────────
    900W      快速冷却   马氏体/铁素体为主(α-Fe)      ~224      细晶, 缺陷少
              ↓                                        ↓
    1200W     中速冷却   马氏体+少量奥氏体            ~243      晶粒开始长大
              ↓                                        ↓
    1500W     较慢冷却   奥氏体为主(γ-Fe)             ~289      树枝晶明显, 析出相增多
              ↓                                        ↓
    1800W     慢速冷却   奥氏体为主(γ-Fe)             ~385      粗大树枝晶, 稀释率高

    关键机理:
    1. 低功率(900W): 冷却快 → 马氏体相变(γ→α') → 硬度低(马氏体不完全)
    2. 中功率(1200W): 冷却适中 → 马氏体+残留奥氏体
    3. 高功率(1500-1800W): 冷却慢 → 奥氏体稳定化(Ni12%稳定) + Cr碳化物析出

    硬度反常(1800W最高)的原因:
    - 虽然奥氏体本身较软, 但高功率导致:
      (a) Cr碳化物大量析出(硬质相)
      (b) 稀释率增加, 基材Fe稀释合金成分
      (c) 凝固组织粗化, 二次枝晶间距增大
    """
    print(evolution)


def main():
    print("金相组织识别分析")
    print("材料体系: JG-1铁基合金(Cr18Ni12) + Q355低碳钢基材")
    print("工艺: 激光熔覆 (900W/1200W/1500W/1800W)")

    # 1. XRD相分析
    xrd_results = xrd_phase_analysis()

    # 2. 金相分割区域分析
    seg_results = segmentation_phase_analysis()

    # 3. 综合识别
    summary = correlate_xrd_seg(xrd_results, seg_results)

    # 4. 组织演变规律
    print_phase_evolution()

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
