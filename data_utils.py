import torch
import numpy as np
from scipy.stats import qmc
from matplotlib.path import Path
import matplotlib.pyplot as plt

# ==== Normalization utilities ====
def minmax_normalize(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0

def minmax_denormalize(val, vmin, vmax):
    return 0.5 * (val + 1.0) * (vmax - vmin) + vmin

def aoa_normalize(aoa_deg, aoa_min, aoa_max):
    return 2.0 * (aoa_deg - aoa_min) / (aoa_max - aoa_min) - 1.0
# ================================

def get_airfoil_points(filename):
    """
    Reads airfoil surface points from a file.
    The file format:
      - First line: integer number of points
      - Following lines: x y z (z is ignored)
    Returns:
      x_airfoil, y_airfoil (each a 1D numpy array)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_points = int(lines[0].strip())
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            data.append((x, y))
    data = np.array(data)
    if data.shape[0] != n_points:
        print(f"Warning: File says {n_points} points, but read {data.shape[0]}.")
    x_airfoil = data[:, 0]
    y_airfoil = data[:, 1]
    return x_airfoil, y_airfoil

def read_structured_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    data = [line.strip().split() for line in lines[1:] if line.strip()]
    mesh_data = np.array([[float(val) for val in line] for line in data])
    x_mesh = mesh_data[:, 0]
    y_mesh = mesh_data[:, 1]
    return x_mesh, y_mesh, nx, ny, mesh_data

def generate_training_data(aoa_deg, mesh_info, aoa_min, aoa_max, airfoil_file="airfoil-points.dat"):
    aoa_norm = aoa_normalize(aoa_deg, aoa_min, aoa_max)
    x_mesh = mesh_info["x_mesh"]
    y_mesh = mesh_info["y_mesh"]
    lb = mesh_info["lb"]
    ub = mesh_info["ub"]

    # Normalize mesh collocation points
    x_mesh_norm = minmax_normalize(x_mesh, lb[0], ub[0])
    y_mesh_norm = minmax_normalize(y_mesh, lb[1], ub[1])
    aoa_mesh = np.full_like(x_mesh_norm, aoa_norm)
    x_f_np = np.stack([x_mesh_norm, y_mesh_norm, aoa_mesh], axis=1)
    x_f = torch.tensor(x_f_np, dtype=torch.float32, requires_grad=True)

    # Airfoil boundary points
    x_airfoil, y_airfoil = get_airfoil_points(airfoil_file)
    x_airfoil_norm = minmax_normalize(x_airfoil, lb[0], ub[0])
    y_airfoil_norm = minmax_normalize(y_airfoil, lb[1], ub[1])
    x_b_np = np.stack((x_airfoil_norm, y_airfoil_norm), axis=1)
    aoa_b = np.full((x_b_np.shape[0], 1), aoa_norm)
    x_b_np = np.concatenate([x_b_np, aoa_b], axis=1)
    x_b = torch.tensor(x_b_np, dtype=torch.float32)

    # Boundary points from mesh
    tol = 1e-5
    min_x = x_mesh.min()
    max_x = x_mesh.max()
    min_y = y_mesh.min()
    max_y = y_mesh.max()
    inlet_mask = np.abs(x_mesh - min_x) < tol
    outlet_mask = np.abs(x_mesh - max_x) < tol
    bottom_mask = np.abs(y_mesh - min_y) < tol
    top_mask = np.abs(y_mesh - max_y) < tol

    x_inlet_np = np.stack([
        minmax_normalize(x_mesh[inlet_mask], lb[0], ub[0]),
        minmax_normalize(y_mesh[inlet_mask], lb[1], ub[1]),
        aoa_mesh[inlet_mask]
    ], axis=1)
    x_inlet = torch.tensor(x_inlet_np, dtype=torch.float32)

    x_outlet_np = np.stack([
        minmax_normalize(x_mesh[outlet_mask], lb[0], ub[0]),
        minmax_normalize(y_mesh[outlet_mask], lb[1], ub[1]),
        aoa_mesh[outlet_mask]
    ], axis=1)
    x_outlet = torch.tensor(x_outlet_np, dtype=torch.float32)

    x_bottom_np = np.stack([
        minmax_normalize(x_mesh[bottom_mask], lb[0], ub[0]),
        minmax_normalize(y_mesh[bottom_mask], lb[1], ub[1]),
        aoa_mesh[bottom_mask]
    ], axis=1)
    x_bottom = torch.tensor(x_bottom_np, dtype=torch.float32)

    x_top_np = np.stack([
        minmax_normalize(x_mesh[top_mask], lb[0], ub[0]),
        minmax_normalize(y_mesh[top_mask], lb[1], ub[1]),
        aoa_mesh[top_mask]
    ], axis=1)
    x_top = torch.tensor(x_top_np, dtype=torch.float32)

    # Visualization (for normalized points)
    def to_np(t): return t.detach().cpu().numpy()
    plt.figure(figsize=(12, 12))
    plt.scatter(x_mesh_norm, y_mesh_norm, s=1, label='Mesh Points (Collocation)', alpha=0.2)
    plt.scatter(to_np(x_b)[:, 0], to_np(x_b)[:, 1], s=2, label='Airfoil Boundary', alpha=0.7)
    plt.scatter(x_inlet_np[:, 0], x_inlet_np[:, 1], s=8, label='Inlet', alpha=0.7, c='green')
    plt.scatter(x_outlet_np[:, 0], x_outlet_np[:, 1], s=8, label='Outlet', alpha=0.7, c='red')
    plt.scatter(x_top_np[:, 0], x_top_np[:, 1], s=8, label='Top', alpha=0.7, c='purple')
    plt.scatter(x_bottom_np[:, 0], x_bottom_np[:, 1], s=8, label='Bottom', alpha=0.7, c='brown')
    plt.legend()
    plt.xlabel('x (normalized)')
    plt.ylabel('y (normalized)')
    plt.title('Training Points Distribution (normalized)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'training_points_aoa{aoa_deg}.png', dpi=300)
    plt.close()

    return x_f, x_b, x_inlet, x_outlet, x_top, x_bottom, x_b

def load_steady_data(filename, L, U_inf, rho, aoa_deg, aoa_min, aoa_max, lb=None, ub=None):
    data = np.loadtxt(filename, skiprows=1)
    x_data = data[:, 1]
    y_data = data[:, 2]
    if lb is not None and ub is not None:
        x_data_norm = minmax_normalize(x_data, lb[0], ub[0])
        y_data_norm = minmax_normalize(y_data, lb[1], ub[1])
    else:
        x_data_norm = x_data / L
        y_data_norm = y_data / L
    aoa_norm = aoa_normalize(aoa_deg, aoa_min, aoa_max)
    aoa_col = np.full_like(x_data_norm, aoa_norm)
    coords = torch.tensor(np.stack((x_data_norm, y_data_norm, aoa_col), axis=1), dtype=torch.float32)
    p_data = data[:, 3] / (rho * U_inf ** 2)
    u_data = data[:, 4] / U_inf
    v_data = data[:, 5] / U_inf
    values = torch.tensor(np.stack((u_data, v_data, p_data), axis=1), dtype=torch.float32)
    return coords, values