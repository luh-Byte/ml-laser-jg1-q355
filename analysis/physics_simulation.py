"""
物理仿真数据生成器 — 基于4组真实数据校准
方法: 物理模型 + 实验数据插值 + 物理约束噪声

真实数据 (v=10mm/s, feed=10g/min):
  900W:  HV=224.2, grain=16.9, dilution=60.4, friction=0.257
  1200W: HV=243.4, grain=18.6, dilution=60.4, friction=0.210
  1500W: HV=288.6, grain=23.0, dilution=60.5, friction=0.300
  1800W: HV=385.2, grain=23.8, dilution=60.3, friction=0.104

扩展方法:
  1. 插值: 在4个校准点之间/之外用物理约束插值
  2. 扫描速度修正: v偏离10mm/s时用热输入缩放
  3. 物理噪声: 模拟测量/工艺波动
"""

import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "analysis_output")
os.makedirs(OUTPUT, exist_ok=True)

# ===== 4组真实实验数据 (v=10mm/s, feed=10g/min基准) =====
REAL_CALIBRATION = {
    900:  {"hv": 224.2, "grain": 16.9, "dilution": 60.4, "friction": 0.2567, "std_hv": 62.7},
    1200: {"hv": 243.4, "grain": 18.6, "dilution": 60.4, "friction": 0.2095, "std_hv": 96.0},
    1500: {"hv": 288.6, "grain": 23.0, "dilution": 60.5, "friction": 0.3003, "std_hv": 96.6},
    1800: {"hv": 385.2, "grain": 23.8, "dilution": 60.3, "friction": 0.1044, "std_hv": 111.7},
}

CAL_POWER = sorted(REAL_CALIBRATION.keys())


def interp_property(P, prop, v_ratio=1.0):
    """
    基于4个校准点的物理约束插值
    P: 功率 [W]
    prop: 属性名 ('hv', 'grain', 'dilution', 'friction')
    v_ratio: v/10 (扫描速度偏离基准时的修正系数)
    """
    real_P = np.array(CAL_POWER, dtype=float)
    real_vals = np.array([REAL_CALIBRATION[p][prop] for p in CAL_POWER])

    # 功率插值
    val = np.interp(P, real_P, real_vals)

    # 外推: 低于900W时用线性外推, 高于1800W时用二次外推
    if P < 900:
        slope = (real_vals[1] - real_vals[0]) / (real_P[1] - real_P[0])
        val = real_vals[0] + slope * (P - 900)
    elif P > 1800:
        # 用最后3个点做二次外推
        coeffs = np.polyfit(real_P[-3:], real_vals[-3:], 2)
        val = np.polyval(coeffs, P)

    # 扫描速度修正: 快冷->细晶+高硬度, 慢冷->粗晶+低硬度
    if v_ratio != 1.0:
        if prop == "grain":
            val *= v_ratio ** (-0.3)  # 快冷->细晶
        elif prop == "hv":
            val *= v_ratio ** (0.15)  # 快冷->高硬度
        elif prop == "friction":
            val *= v_ratio ** (-0.1)

    return val


def generate_single(P, v, feed_rate=10.0, noise=True, seed=None):
    """生成单条仿真数据"""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    v_ratio = v / 10.0
    feed_ratio = feed_rate / 10.0

    # 热输入
    beam = 3.0
    t_layer = 1.5
    Q_vol = P / (v * beam * t_layer)
    Q_lin = P / v

    # 基础属性 (校准插值)
    hv = interp_property(P, "hv", v_ratio)
    grain = interp_property(P, "grain", v_ratio)
    dilution = interp_property(P, "dilution")
    friction = interp_property(P, "friction", v_ratio)

    # 送粉速率修正
    hv *= feed_ratio ** (-0.05)
    grain *= feed_ratio ** (0.05)

    # 物理噪声 (基于真实实验的标准差比例)
    if noise:
        hv_std = REAL_CALIBRATION[min(CAL_POWER, key=lambda p: abs(p - P))]["std_hv"]
        hv += rng.normal(0, hv_std * 0.15)
        grain += rng.normal(0, grain * 0.08)
        dilution += rng.normal(0, 1.5)
        friction += rng.normal(0, 0.02)

    # 物理约束
    hv = np.clip(hv, 80, 600)
    grain = np.clip(grain, 2, 80)
    dilution = np.clip(dilution, 10, 95)
    friction = np.clip(friction, 0.01, 0.6)

    # 温度场估算
    k = 15.0
    rho = 7800
    cp = 500
    alpha = k / (rho * cp)
    T_center = P / (2 * np.pi * k * 0.001) * 0.001 + 25
    T_center = min(T_center, 3000)

    # 标签
    if hv > 350:
        hv_class = "high"
    elif hv > 250:
        hv_class = "medium"
    else:
        hv_class = "low"

    if grain < 15:
        grain_class = "fine"
    elif grain < 25:
        grain_class = "medium"
    else:
        grain_class = "coarse"

    return {
        "power_w": round(P, 1),
        "scan_speed_mm_s": round(v, 2),
        "feed_rate_g_min": round(feed_rate, 2),
        "beam_diameter_mm": beam,
        "layer_height_mm": t_layer,
        "heat_input_vol_W_mm3": round(Q_vol, 4),
        "linear_energy_J_mm": round(Q_lin, 2),
        "T_center_C": round(T_center, 1),
        "grain_size_um": round(grain, 2),
        "dilution_pct": round(dilution, 2),
        "porosity_pct": round(np.clip(3.0 + 0.02 * (Q_lin - 150) ** 2 / 1000, 0.5, 20.0), 2),
        "hardness_HV": round(hv, 1),
        "friction_coeff": round(friction, 4),
        "hv_class": hv_class,
        "grain_class": grain_class,
    }


def validate():
    """验证4组真实数据"""
    print("=== Validation (v=10, feed=10, no noise) ===")
    print()
    print(f"{'Power':<8s} {'Real HV':<10s} {'Sim HV':<10s} {'Err%':<8s} {'Real Grain':<12s} {'Sim Grain':<12s}")
    print("-" * 65)

    errors = []
    for P in CAL_POWER:
        real = REAL_CALIBRATION[P]
        sim = generate_single(P, 10.0, 10.0, noise=False, seed=P)
        err_pct = (sim["hardness_HV"] - real["hv"]) / real["hv"] * 100
        errors.append(abs(err_pct))
        print(f"{P}W     {real['hv']:<10.1f} {sim['hardness_HV']:<10.1f} {err_pct:<+8.1f}% {real['grain']:<12.1f} {sim['grain_size_um']:<12.1f}")

    print(f"\nMAPE: {np.mean(errors):.1f}%")


def generate_batch(n_total=10000):
    """批量生成"""
    print(f"\n=== Generating {n_total} samples ===")

    rng = np.random.RandomState(42)

    # 功率: 600-2400W (覆盖+外推)
    P_samples = rng.uniform(600, 2400, n_total)
    # 扫描速度: 5-20 mm/s
    v_samples = rng.uniform(5.0, 20.0, n_total)
    # 送粉速率: 5-20 g/min
    feed_samples = rng.uniform(5.0, 20.0, n_total)

    # 加入4组真实实验点
    for P in CAL_POWER:
        P_samples = np.append(P_samples, P)
        v_samples = np.append(v_samples, 10.0)
        feed_samples = np.append(feed_samples, 10.0)

    n_total = len(P_samples)

    results = []
    for i in range(n_total):
        row = generate_single(P_samples[i], v_samples[i], feed_samples[i],
                              noise=True, seed=i)
        row["sample_id"] = i
        results.append(row)

    df = pd.DataFrame(results)

    csv_path = os.path.join(OUTPUT, "simulation_batch.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Generated: {len(df)} samples")
    print(f"Saved: {csv_path}")
    print(f"\nRanges:")
    print(f"  Power:    [{df['power_w'].min():.0f}, {df['power_w'].max():.0f}] W")
    print(f"  Speed:    [{df['scan_speed_mm_s'].min():.1f}, {df['scan_speed_mm_s'].max():.1f}] mm/s")
    print(f"  Hardness: [{df['hardness_HV'].min():.1f}, {df['hardness_HV'].max():.1f}] HV")
    print(f"  Grain:    [{df['grain_size_um'].min():.1f}, {df['grain_size_um'].max():.1f}] um")
    print(f"  Dilution: [{df['dilution_pct'].min():.1f}, {df['dilution_pct'].max():.1f}] %")
    print(f"\nClass dist: {df['hv_class'].value_counts().to_dict()}")
    print(f"Grain dist: {df['grain_class'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    validate()
    df = generate_batch(n_total=10000)
    print("\nDone.")
