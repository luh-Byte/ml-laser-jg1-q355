"""
FEniCSx 全耦合相场法 — 热传导 + 相演化

耦合PDE系统:
  1. 热传导: rho*cp*dT/dt = div(k*grad(T)) + Q
  2. 相演化: dphi_i/dt = -M_i * df/dphi_i

相体系 (JG-1铁基合金):
  phi_a: 奥氏体 (γ-Fe, FCC) — Ni12%稳定化
  phi_m: 马氏体 (α'-Fe, BCT) — 快冷转变
  phi_f: 铁素体 (α-Fe, BCC) = 1 - phi_a - phi_m
  phi_c: 碳化物 (Cr23C6) — 慢冷析出

自由能: Landau-Ginzburg双势阱
  f(phi, T) = W * phi^2 * (1-phi)^2 + g(T) * phi

时间步进: 隐式Euler
"""
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
import pandas as pd
import json, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
OUTPUT_DIR = os.path.expanduser("~/fenics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== 物理参数 (JG-1 / Q355体系) =====
PARAMS = {
    # 热物性
    "k_clad": 15.0,          # W/(m*K)
    "k_sub": 48.0,
    "rho": 7800,             # kg/m^3
    "cp": 500,               # J/(kg*K)
    "T_amb": 298.15,         # K

    # 相场参数
    "M_aust": 1e-4,          # 奥氏体迁移率
    "M_mart": 5e-5,          # 马氏体迁移率
    "W_interface": 1e5,      # 界面能密度 (J/m^3)
    "kappa": 1e-3,           # 梯度能系数 (J/m)

    # 相变温度
    "T_aust_start": 1073,    # K (800C)
    "T_aust_end": 1423,      # K (1150C, 奥氏体完全形成)
    "T_ms": 623,             # K (350C, 马氏体开始)
    "T_mf": 423,             # K (150C, 马氏体结束)

    # 碳化物析出
    "M_carb": 1e-6,          # 碳化物迁移率 (慢)
    "T_carb_start": 873,     # K (600C)
    "T_carb_end": 1273,      # K (1000C)

    # 热源
    "beam_R": 0.0015,        # m
    "calib_factor": 0.0482,
}


def create_domain(Lx=0.01, Ly=0.005, nx=80, ny=40):
    return mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([Lx, Ly])], [nx, ny],
        cell_type=mesh.CellType.triangle)


def solve_coupled(P, speed_mm_s=10.0, dt=0.001, n_steps=200, Lx=0.01, Ly=0.005, nx=60, ny=30):
    """
    求解耦合热-相场系统
    P: 功率 [W]
    speed_mm_s: 扫描速度 [mm/s]
    dt: 时间步长 [s]
    n_steps: 时间步数
    """
    p = PARAMS
    domain = create_domain(Lx, Ly, nx, ny)
    V = fem.functionspace(domain, ("CG", 1))
    Vc = fem.functionspace(domain, ("DG", 0))
    geom = domain.geometry.x

    # 单元形心
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    conn = domain.topology.connectivity(domain.topology.dim, 0)
    centroids = np.zeros((n_cells, 3))
    for i in range(n_cells):
        centroids[i] = geom[conn.links(i)].mean(axis=0)

    # 导热系数
    k_arr = np.where(centroids[:, 1] >= 0.003, p["k_clad"], p["k_sub"]).astype(np.float64)

    # 高斯热源
    cx, cy = Lx / 2.0, Ly - 0.0005
    r2 = (centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2
    speed_factor = (10.0 / speed_mm_s) ** 0.5
    Q0 = p["calib_factor"] * speed_factor * 3.0 * P / (np.pi * p["beam_R"]**2 * 0.001)

    # 初始条件
    T_prev = fem.Function(V)
    T_prev.x.array[:] = p["T_amb"]

    phi_a_prev = fem.Function(Vc)  # 奥氏体
    phi_m_prev = fem.Function(Vc)  # 马氏体
    phi_c_prev = fem.Function(Vc)  # 碳化物
    phi_a_prev.x.array[:] = 0.0
    phi_m_prev.x.array[:] = 0.0
    phi_c_prev.x.array[:] = 0.0

    # 边界条件
    geom_nodes = geom[:V.dofmap.index_map.size_local]
    bottom_dofs = np.where(geom_nodes[:, 1] < 1e-6)[0].astype(np.int32)

    T_history = np.zeros(n_steps + 1)
    T_history[0] = p["T_amb"]

    # 热源时间序列 (高斯脉冲, 模拟扫描)
    pulse_center = n_steps // 3
    pulse_width = n_steps // 6

    for step in range(1, n_steps + 1):
        t = step * dt

        # 时间依赖热源 (高斯脉冲)
        t_frac = (step - pulse_center) / pulse_width
        time_factor = np.exp(-0.5 * t_frac**2) if abs(t_frac) < 3 else 0.0

        # 当前相分数
        phi_a_arr = phi_a_prev.x.array
        phi_m_arr = phi_m_prev.x.array
        phi_c_arr = phi_c_prev.x.array

        # 相变潜热修正 (奥氏体→马氏体放热)
        latent_heat = 0.0
        if np.any(phi_m_arr > 0.1):
            latent_heat = 50000.0  # J/kg (近似)

        # --- 求解温度场 ---
        T_trial = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # 热源 DG0
        Q_func = fem.Function(Vc)
        Q_func.x.array[:] = Q0 * time_factor * np.exp(-2.0 * r2 / p["beam_R"]**2)

        # k with phase correction (austenite has lower k)
        k_func = fem.Function(Vc)
        k_func.x.array[:] = k_arr * (1.0 - 0.3 * phi_a_arr)  # 奥氏体降低导热

        # 隐式Euler: rho*cp*(T_new - T_old)/dt = div(k*grad(T_new)) + Q
        a = (p["rho"] * p["cp"] / dt) * T_trial * v * ufl.dx + \
            ufl.dot(k_func * ufl.grad(T_trial), ufl.grad(v)) * ufl.dx
        L_form = (p["rho"] * p["cp"] / dt) * T_prev * v * ufl.dx + \
                 Q_func * v * ufl.dx

        bc = fem.dirichletbc(fem.Constant(domain, p["T_amb"]), bottom_dofs, V)

        problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix=f"heat_{step}_",
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                               "pc_factor_mat_solver_type": "mumps"})
        T_new = problem.solve()
        T_arr = T_new.x.array

        # --- 求解相场 (Allen-Cahn) ---
        # 奥氏体: dphi_a/dt = -M_a * df_a/dphi_a
        #   df_a/dphi_a = W*2*phi_a*(2*phi_a-1) + g_a(T)*(1-2*phi_a)
        #   g_a(T) = 0 if T < T_aust_start, >0 if T > T_aust_end

        # 马氏体: dphi_m/dt = -M_m * df_m/dphi_m (仅在冷却时)
        # 碳化物: dphi_c/dt = -M_c * df_c/dphi_c (仅在T_carb范围内)

        T_arr_c = T_arr[:Vc.dofmap.index_map.size_local]

        # 奥氏体驱动力
        g_a = np.zeros_like(T_arr_c)
        mask_aust = (T_arr_c >= p["T_aust_start"]) & (T_arr_c <= p["T_aust_end"])
        g_a[mask_aust] = (T_arr_c[mask_aust] - p["T_aust_start"]) / (p["T_aust_end"] - p["T_aust_start"])

        # 马氏体驱动力 (冷却时, T < T_ms)
        g_m = np.zeros_like(T_arr_c)
        mask_mart = (T_arr_c <= p["T_ms"]) & (phi_a_arr[:len(T_arr_c)] > 0.01)
        g_m[mask_mart] = (p["T_ms"] - T_arr_c[mask_mart]) / (p["T_ms"] - p["T_mf"])

        # 碳化物驱动力
        g_c = np.zeros_like(T_arr_c)
        mask_carb = (T_arr_c >= p["T_carb_start"]) & (T_arr_c <= p["T_carb_end"])
        g_c[mask_carb] = 0.5

        # Allen-Cahn更新 (显式Euler)
        W = p["W_interface"]
        phi_a_new = phi_a_prev.x.array[:len(T_arr_c)].copy()
        phi_m_new = phi_m_prev.x.array[:len(T_arr_c)].copy()
        phi_c_new = phi_c_prev.x.array[:len(T_arr_c)].copy()

        # 奥氏体演化
        df_a = 2 * W * phi_a_new * (2 * phi_a_new - 1) + g_a * (1 - 2 * phi_a_new)
        phi_a_new -= p["M_aust"] * dt * df_a * 1e6  # 缩放因子

        # 马氏体演化 (仅冷却时, 从奥氏体转变)
        df_m = 2 * W * phi_m_new * (2 * phi_m_new - 1) + g_m * (1 - 2 * phi_m_new)
        phi_m_new -= p["M_mart"] * dt * df_m * 1e6

        # 碳化物析出 (从铁素体/奥氏体析出)
        df_c = 2 * W * phi_c_new * (2 * phi_c_new - 1) + g_c * (1 - 2 * phi_c_new)
        phi_c_new -= p["M_carb"] * dt * df_c * 1e6

        # 物理约束: 0 <= phi <= 1, phi_a + phi_m <= 1
        phi_a_new = np.clip(phi_a_new, 0, 1)
        phi_m_new = np.clip(phi_m_new, 0, 1)
        phi_c_new = np.clip(phi_c_new, 0, 0.3)  # 碳化物最多30%
        total = phi_a_new + phi_m_new
        mask_over = total > 1.0
        phi_a_new[mask_over] /= total[mask_over]
        phi_m_new[mask_over] /= total[mask_over]

        # 更新
        phi_a_prev.x.array[:len(phi_a_new)] = phi_a_new
        phi_m_prev.x.array[:len(phi_m_new)] = phi_m_new
        phi_c_prev.x.array[:len(phi_c_new)] = phi_c_new
        T_prev.x.array[:] = T_arr

        T_history[step] = np.max(T_arr) - 273.15

    # 提取最终结果
    T_final = T_prev.x.array - 273.15
    phi_a_final = phi_a_prev.x.array
    phi_m_final = phi_m_prev.x.array
    phi_c_final = phi_c_prev.x.array

    T_max = float(np.max(T_final))
    phi_a_mean = float(np.mean(phi_a_final))
    phi_m_mean = float(np.mean(phi_m_final))
    phi_c_mean = float(np.mean(phi_c_final))

    # 晶粒尺寸 (基于冷却速率)
    cooling_rate = abs(T_history[-1] - T_history[-50]) / (50 * dt) if n_steps > 50 else 100
    grain = 120.0 * max(cooling_rate, 1) ** (-0.38)
    grain = np.clip(grain, 5, 60)

    # 硬度 (相加权 + Hall-Petch + 碳化物)
    HV_base = phi_a_mean * 180 + phi_m_mean * 450 + (1 - phi_a_mean - phi_m_mean) * 120
    HV_hp = 600 / np.sqrt(grain * 1e-6) * 1e-3
    HV_carb = phi_c_mean * 800
    HV = HV_base + HV_hp + HV_carb

    # 熔池
    node_coords = geom[:V.dofmap.index_map.size_local]
    melt_mask = T_final[:len(node_coords)] >= 1420
    if np.any(melt_mask):
        melt_pts = node_coords[melt_mask]
        pool_W = float(np.ptp(melt_pts[:, 0])) * 1000
        pool_D = float(np.ptp(melt_pts[:, 1])) * 1000
    else:
        pool_W = pool_D = 0.0

    return {
        "power_w": int(P),
        "scan_speed_mm_s": round(speed_mm_s, 2),
        "T_max_C": round(T_max, 1),
        "pool_width_mm": round(pool_W, 3),
        "pool_depth_mm": round(pool_D, 3),
        "cooling_rate_K_s": round(cooling_rate, 1),
        "grain_size_um": round(grain, 2),
        "phi_austenite": round(phi_a_mean, 4),
        "phi_martensite": round(phi_m_mean, 4),
        "phi_carbide": round(phi_c_mean, 4),
        "hardness_HV": round(HV, 1),
    }


def validate():
    print("=" * 60)
    print("FEniCSx Coupled Phase-Field Validation")
    print("=" * 60)

    real = {
        900:  {"hv": 224.2, "grain": 16.9},
        1200: {"hv": 243.4, "grain": 18.6},
        1500: {"hv": 288.6, "grain": 23.0},
        1800: {"hv": 385.2, "grain": 23.8},
    }

    results = []
    for P in [900, 1200, 1500, 1800]:
        r = solve_coupled(P, n_steps=150, nx=40, ny=20)
        r["real_hv"] = real[P]["hv"]
        r["real_grain"] = real[P]["grain"]
        r["err_hv"] = round(abs(r["hardness_HV"] - real[P]["hv"]), 1)
        r["err_grain"] = round(abs(r["grain_size_um"] - real[P]["grain"]), 1)
        results.append(r)

    if rank == 0:
        print("\n%-6s %-8s %-10s %-10s %-8s %-10s %-10s %-8s" % (
            "Power", "G(K/s)", "phi_a", "phi_m", "phi_c", "Sim HV", "Real HV", "Err HV"))
        print("-" * 80)
        for r in results:
            print("%dW  %-8.1f %-10.3f %-10.3f %-8.3f %-10.1f %-10.1f %-8.1f" % (
                r["power_w"], r["cooling_rate_K_s"], r["phi_austenite"],
                r["phi_martensite"], r["phi_carbide"], r["hardness_HV"],
                r["real_hv"], r["err_hv"]))

        hv_errs = [r["err_hv"] / r["real_hv"] * 100 for r in results]
        grain_errs = [r["err_grain"] / r["real_grain"] * 100 for r in results]
        print("\nHV MAPE: %.1f%%" % np.mean(hv_errs))
        print("Grain MAPE: %.1f%%" % np.mean(grain_errs))

        with open(os.path.join(OUTPUT_DIR, "coupled_phasefield_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


def batch_generate():
    if rank == 0:
        print("\n=== Batch: 10 powers x 5 speeds = 50 samples ===")

    powers = np.linspace(600, 2400, 10).astype(int)
    speeds = [10.0, 12.5, 15.0, 17.5, 20.0]
    results = []

    for P in powers:
        for spd in speeds:
            r = solve_coupled(P, speed_mm_s=spd, n_steps=100, nx=30, ny=15)
            results.append(r)
            if rank == 0:
                print("  P=%dW spd=%.1f -> HV=%.1f phi_a=%.3f phi_m=%.3f" % (
                    P, spd, r["hardness_HV"], r["phi_austenite"], r["phi_martensite"]))

    # 加入4组真实数据
    real_points = [
        {"power_w": 900, "scan_speed_mm_s": 10.0, "T_max_C": 1650, "pool_width_mm": 1.2, "pool_depth_mm": 0.4,
         "cooling_rate_K_s": 174, "grain_size_um": 16.9, "phi_austenite": 0.21, "phi_martensite": 0.65,
         "phi_carbide": 0.05, "hardness_HV": 224.2, "source": "real"},
        {"power_w": 1200, "scan_speed_mm_s": 10.0, "T_max_C": 1850, "pool_width_mm": 1.5, "pool_depth_mm": 0.6,
         "cooling_rate_K_s": 135, "grain_size_um": 18.6, "phi_austenite": 0.18, "phi_martensite": 0.68,
         "phi_carbide": 0.04, "hardness_HV": 243.4, "source": "real"},
        {"power_w": 1500, "scan_speed_mm_s": 10.0, "T_max_C": 2100, "pool_width_mm": 1.8, "pool_depth_mm": 0.8,
         "cooling_rate_K_s": 77, "grain_size_um": 23.0, "phi_austenite": 0.79, "phi_martensite": 0.10,
         "phi_carbide": 0.08, "hardness_HV": 288.6, "source": "real"},
        {"power_w": 1800, "scan_speed_mm_s": 10.0, "T_max_C": 2400, "pool_width_mm": 2.1, "pool_depth_mm": 1.0,
         "cooling_rate_K_s": 71, "grain_size_um": 23.8, "phi_austenite": 0.75, "phi_martensite": 0.12,
         "phi_carbide": 0.10, "hardness_HV": 385.2, "source": "real"},
    ]
    results.extend(real_points)

    if rank == 0:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "coupled_phasefield_training.csv")
        df.to_csv(csv_path, index=False)
        csv_win = "/mnt/d/ML-Laser-JG1-Q355/ml-laser-jg1-q355/analysis_output/coupled_phasefield_training.csv"
        df.to_csv(csv_win, index=False)
        print("\nTotal: %d samples" % len(df))
        print("Saved: %s" % csv_path)
        print("Saved: %s" % csv_win)


if __name__ == "__main__":
    validate()
    batch_generate()
    if rank == 0:
        print("\nDone.")
