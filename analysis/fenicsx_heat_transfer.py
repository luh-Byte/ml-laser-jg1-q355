"""
FEniCSx 2D Laser Cladding Heat Transfer - v5
Units: ALL in SI (meters, W, K)
Boundary: bottom fixed T=25C (Dirichlet), rest natural BC
"""
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
import json, os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
OUTPUT_DIR = os.path.expanduser("~/fenics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def solve_temperature_field(P, R_m=0.0015, Lx=0.01, Ly=0.005, nx=100, ny=50):
    """
    P: power [W]
    R_m: beam radius [m] (1.5mm = 0.0015m)
    Lx, Ly: domain size [m] (10mm x 5mm)
    """
    domain = mesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([Lx, Ly])], [nx, ny],
        cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("CG", 1))
    Vc = fem.functionspace(domain, ("DG", 0))

    geom = domain.geometry.x
    n_cells = domain.topology.index_map(domain.topology.dim).size_local
    conn = domain.topology.connectivity(domain.topology.dim, 0)

    # Cell centroids
    centroids = np.zeros((n_cells, 3))
    for i in range(n_cells):
        centroids[i] = geom[conn.links(i)].mean(axis=0)

    # Thermal conductivity [W/(m*K)]
    k = fem.Function(Vc)
    k.x.array[:] = np.where(centroids[:, 1] >= 0.003, 15.0, 48.0)

    # Gaussian heat source [W/m^3], calibrated from 4 real experiments
    CALIBRATION_FACTOR = 0.0482
    cx, cy = Lx / 2.0, Ly - 0.0005
    r2 = (centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2
    Q0 = CALIBRATION_FACTOR * 3.0 * P / (np.pi * R_m**2 * 0.001)
    Q = fem.Function(Vc)
    Q.x.array[:] = Q0 * np.exp(-2.0 * r2 / R_m**2)

    # Bilinear form: -div(k grad T) = Q
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(k * ufl.grad(T), ufl.grad(v)) * ufl.dx
    L_form = Q * v * ufl.dx

    # Dirichlet BC: bottom edge T = 25C = 298.15K
    geom = domain.geometry.x
    bottom_dofs = np.where(geom[:, 1] < 1e-6)[0].astype(np.int32)
    bc = fem.dirichletbc(fem.Constant(domain, 298.15), bottom_dofs, V)

    problem = LinearProblem(a, L_form, bcs=[bc], petsc_options_prefix="heat_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})
    T_sol = problem.solve()

    T_arr = T_sol.x.array - 273.15  # Convert K -> C
    T_max = float(np.max(T_arr))
    T_min = float(np.min(T_arr))

    # Melt pool (T > 1420C = JG-1 melting point)
    node_coords = geom[:V.dofmap.index_map.size_local]
    melt_mask = T_arr[:len(node_coords)] >= 1420
    if np.any(melt_mask):
        melt_pts = node_coords[melt_mask]
        pool_W = float(np.ptp(melt_pts[:, 0])) * 1000  # m -> mm
        pool_D = float(np.ptp(melt_pts[:, 1])) * 1000
    else:
        pool_W = pool_D = 0.0

    result = {"power_w": int(P), "T_max_C": round(T_max, 1), "T_min_C": round(T_min, 1),
              "pool_width_mm": round(pool_W, 3), "pool_depth_mm": round(pool_D, 3)}
    return result


def validate():
    print("=" * 60)
    print("FEniCSx 2D Heat Transfer Validation (SI units)")
    print("=" * 60)

    real = {900: 1650, 1200: 1850, 1500: 2100, 1800: 2400}
    results = []
    for P in [900, 1200, 1500, 1800]:
        r = solve_temperature_field(P)
        r["real_Tmax"] = real[P]
        r["error_pct"] = round(abs(r["T_max_C"] - real[P]) / real[P] * 100, 1)
        results.append(r)
        if rank == 0:
            print(f"  P={P}W  Sim={r['T_max_C']:.0f}C  Real={real[P]}C  Err={r['error_pct']:.1f}%")

    if rank == 0:
        mape = np.mean([r["error_pct"] for r in results])
        print(f"\nMAPE: {mape:.1f}%")
        with open(os.path.join(OUTPUT_DIR, "fenicsx_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results


def batch_generate():
    if rank == 0:
        print("\n=== Batch Generation (20 power levels) ===")
    powers = np.linspace(600, 2400, 20)
    results = []
    for P in powers:
        r = solve_temperature_field(P)
        results.append(r)
        if rank == 0:
            print(f"  P={int(P)}W  Tmax={r['T_max_C']:.0f}C  Pool={r['pool_width_mm']:.2f}x{r['pool_depth_mm']:.2f}mm")
    if rank == 0:
        import pandas as pd
        csv_path = os.path.join(OUTPUT_DIR, "fenicsx_batch.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    validate()
    batch_generate()
    if rank == 0:
        print("\nDone.")
