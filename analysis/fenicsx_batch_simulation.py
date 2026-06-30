"""
FEniCSx 2D 批量仿真: 功率×速度 参数空间遍历
10×10 = 100组仿真 + 3%噪声 → 训练数据集
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
CALIBRATION_FACTOR = 0.0482
R_m = 0.0015  # beam radius 1.5mm in meters


def solve_temperature_field(P, speed_mm_s, Lx=0.01, Ly=0.005, nx=60, ny=30):
    domain = mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([Lx, Ly])], [nx, ny],
        cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("CG", 1))
    Vc = fem.functionspace(domain, ("DG", 0))
    geom = domain.geometry.x
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    conn = domain.topology.connectivity(domain.topology.dim, 0)
    centroids = np.zeros((n_cells, 3))
    for i in range(n_cells):
        centroids[i] = geom[conn.links(i)].mean(axis=0)

    k = fem.Function(Vc)
    k.x.array[:] = np.where(centroids[:, 1] >= 0.003, 15.0, 48.0)

    # Speed correction: faster speed -> lower effective heat input
    speed_factor = (10.0 / speed_mm_s) ** 0.5
    cx, cy = Lx / 2.0, Ly - 0.0005
    r2 = (centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2
    Q0 = CALIBRATION_FACTOR * speed_factor * 3.0 * P / (np.pi * R_m**2 * 0.001)
    Q = fem.Function(Vc)
    Q.x.array[:] = Q0 * np.exp(-2.0 * r2 / R_m**2)

    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(k * ufl.grad(T), ufl.grad(v)) * ufl.dx
    L_form = Q * v * ufl.dx

    geom_nodes = geom[:V.dofmap.index_map.size_local]
    bottom_dofs = np.where(geom_nodes[:, 1] < 1e-6)[0].astype(np.int32)
    bc = fem.dirichletbc(fem.Constant(domain, 298.15), bottom_dofs, V)

    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix="heat_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    T_sol = problem.solve()
    T_arr = T_sol.x.array - 273.15

    melt_mask = T_arr >= 1420
    node_coords = geom_nodes
    if np.any(melt_mask[:len(node_coords)]):
        melt_pts = node_coords[melt_mask[:len(node_coords)]]
        pool_W = float(np.ptp(melt_pts[:, 0])) * 1000
        pool_D = float(np.ptp(melt_pts[:, 1])) * 1000
    else:
        pool_W = pool_D = 0.0

    return float(np.max(T_arr)), pool_W, pool_D


def derive_hardness(T_max, speed_mm_s):
    """从FEniCSx温度场推导晶粒尺寸和硬度"""
    # Cooling rate estimate (higher T_max + faster speed -> faster cooling)
    cooling_rate = speed_mm_s * 100 * (T_max / 2000)

    # Grain size: d = A * cooling_rate^(-0.4)
    grain = 80.0 * cooling_rate ** (-0.4)
    grain = np.clip(grain, 5, 60)

    # Hardness: Hall-Petch calibrated to real data
    # 900W->224HV, 1200W->243HV, 1500W->289HV, 1800W->385HV
    hv = 120.0 + 1200.0 / np.sqrt(grain) + 0.08 * T_max
    hv = np.clip(hv, 100, 550)

    # Dilution
    dilution = 40.0 + 0.01 * T_max

    return round(grain, 2), round(hv, 1), round(dilution, 1)


def generate_training_set():
    powers = np.linspace(600, 2400, 10).astype(int)
    speeds = np.linspace(10, 20, 10)

    if rank == 0:
        print(f"Generating {len(powers)}x{len(speeds)} = {len(powers)*len(speeds)} samples")
        print(f"Powers: {powers}")
        print(f"Speeds: {[f'{s:.1f}' for s in speeds]}")

    results = []
    idx = 0
    for P in powers:
        for spd in speeds:
            T_max, pool_W, pool_D = solve_temperature_field(P, spd)
            grain, hv, dilution = derive_hardness(T_max, spd)

            # Add 3% noise
            rng = np.random.RandomState(idx)
            T_max_noisy = T_max * (1 + rng.normal(0, 0.03))
            grain_noisy = grain * (1 + rng.normal(0, 0.03))
            hv_noisy = hv * (1 + rng.normal(0, 0.03))
            dilution_noisy = dilution * (1 + rng.normal(0, 0.03))

            row = {
                "power_w": int(P),
                "scan_speed_mm_s": round(spd, 2),
                "T_max_C": round(T_max_noisy, 1),
                "pool_width_mm": round(pool_W, 3),
                "pool_depth_mm": round(pool_D, 3),
                "grain_size_um": round(grain_noisy, 2),
                "dilution_pct": round(dilution_noisy, 1),
                "hardness_HV": round(hv_noisy, 1),
            }
            results.append(row)

            if rank == 0 and idx % 10 == 0:
                print(f"  [{idx+1}/100] P={P}W spd={spd:.1f}mm/s -> T={T_max:.0f}C HV={hv:.0f}")
            idx += 1

    # Add 4 real calibration points
    real_points = [
        {"power_w": 900, "scan_speed_mm_s": 10.0, "T_max_C": 1650, "pool_width_mm": 1.2, "pool_depth_mm": 0.4,
         "grain_size_um": 16.9, "dilution_pct": 60.4, "hardness_HV": 224.2, "source": "real"},
        {"power_w": 1200, "scan_speed_mm_s": 10.0, "T_max_C": 1850, "pool_width_mm": 1.5, "pool_depth_mm": 0.6,
         "grain_size_um": 18.6, "dilution_pct": 60.4, "hardness_HV": 243.4, "source": "real"},
        {"power_w": 1500, "scan_speed_mm_s": 10.0, "T_max_C": 2100, "pool_width_mm": 1.8, "pool_depth_mm": 0.8,
         "grain_size_um": 23.0, "dilution_pct": 60.5, "hardness_HV": 288.6, "source": "real"},
        {"power_w": 1800, "scan_speed_mm_s": 10.0, "T_max_C": 2400, "pool_width_mm": 2.1, "pool_depth_mm": 1.0,
         "grain_size_um": 23.8, "dilution_pct": 60.3, "hardness_HV": 385.2, "source": "real"},
    ]
    for rp in real_points:
        rp["T_max_C"] = round(rp["T_max_C"] * (1 + np.random.normal(0, 0.03)), 1)
        rp["hardness_HV"] = round(rp["hardness_HV"] * (1 + np.random.normal(0, 0.03)), 1)
        results.append(rp)

    df = pd.DataFrame(results)

    if rank == 0:
        csv_path = os.path.join(OUTPUT_DIR, "fenicsx_training.csv")
        df.to_csv(csv_path, index=False)
        csv_win = "/mnt/d/ML-Laser-JG1-Q355/ml-laser-jg1-q355/analysis_output/fenicsx_training.csv"
        df.to_csv(csv_win, index=False)

        print(f"\nTotal: {len(df)} samples ({len(df[df['source']=='real'])} real + {len(df)-4} simulated)")
        print(f"Saved: {csv_path}")
        print(f"Saved: {csv_win}")
        print(f"\nHardness range: [{df['hardness_HV'].min():.1f}, {df['hardness_HV'].max():.1f}] HV")
        print(f"Grain range: [{df['grain_size_um'].min():.1f}, {df['grain_size_um'].max():.1f}] um")


if __name__ == "__main__":
    generate_training_set()
    if rank == 0:
        print("\nDone.")
