import numpy as np
import torch
import matplotlib.pyplot as plt
from data_utils import minmax_normalize, minmax_denormalize, aoa_normalize


def write_cp_tecplot(model, airfoil_points_aoa, rho, U_inf, p_inf, lb, ub, chord=1.0,
                     output_file="cp_distribution.dat"):
    """
    ✅ FIXED: Coordinates are now properly denormalized from [-1, 1] to physical space
    """
    with torch.no_grad():
        pred = model(airfoil_points_aoa)
    p_star = pred[:, 2].cpu().numpy()
    p = p_star * rho * U_inf ** 2
    Cp = (p - p_inf) / (0.5 * rho * U_inf ** 2)

    # ✅ FIXED: Denormalize coordinates from [-1, 1] to physical space
    x_surface_norm = airfoil_points_aoa[:, 0].cpu().numpy()
    y_surface_norm = airfoil_points_aoa[:, 1].cpu().numpy()
    x_surface = minmax_denormalize(x_surface_norm, lb[0], ub[0])
    y_surface = minmax_denormalize(y_surface_norm, lb[1], ub[1])

    with open(output_file, "w") as f:
        f.write('TITLE="Cp distribution"\n')
        f.write('VARIABLES="X","Y","Cp"\n')
        f.write(f'ZONE I={len(x_surface)}, F=POINT\n')
        for xi, yi, cpi in zip(x_surface, y_surface, Cp):
            f.write(f"{xi:.6f}\t{yi:.6f}\t{cpi:.6f}\n")
    print(f"Cp distribution (Tecplot) saved to {output_file}")


def write_results_to_file(model, chord, U_inf, rho, output_file='results.dat', aoa=0, mesh_info=None, aoa_min=0,
                          aoa_max=3):
    """
    Write flow field results to Tecplot format file.
    ✅ FIXED: Mesh is now correctly reshaped as (ny, nx) for proper coordinate mapping.
    """
    if mesh_info is None:
        raise ValueError("mesh_info must be provided for mesh-based output.")

    x_mesh = mesh_info["x_mesh"]
    y_mesh = mesh_info["y_mesh"]
    nx = mesh_info["nx"]
    ny = mesh_info["ny"]
    lb = mesh_info["lb"]
    ub = mesh_info["ub"]
    aoa_norm = aoa_normalize(aoa, aoa_min, aoa_max)
    x_mesh_norm = minmax_normalize(x_mesh, lb[0], ub[0])
    y_mesh_norm = minmax_normalize(y_mesh, lb[1], ub[1])
    pts = np.stack([x_mesh_norm, y_mesh_norm, np.full_like(x_mesh_norm, aoa_norm)], axis=1)

    # Reshape mesh coordinates: data is stored row-wise (ny rows × nx columns)
    X = x_mesh.reshape(ny, nx)
    Y = y_mesh.reshape(ny, nx)

    pts_torch = torch.tensor(pts, dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        pred = model(pts_torch)
    u = pred[:, 0].cpu().numpy() * U_inf
    v = pred[:, 1].cpu().numpy() * U_inf
    p_star = pred[:, 2].cpu().numpy()
    p = p_star * rho * U_inf ** 2
    # Reshape solution fields: same order as mesh (ny rows × nx columns)
    u = u.reshape(ny, nx)
    v = v.reshape(ny, nx)
    p = p.reshape(ny, nx)

    with open(output_file, 'w', newline='\n') as f:
        f.write(f'# AoA (deg) = {aoa}\n')
        f.write('VARIABLES = "x", "y", "z", "u", "v", "p", "aoa_deg"\n')
        f.write(f'ZONE I={nx}, J={ny}, F=POINT\n')
        for j in range(ny):
            for i in range(nx):
                f.write(f"{X[j, i]:.15e} {Y[j, i]:.15e} 0.0 {u[j, i]:.8e} {v[j, i]:.8e} {p[j, i]:.8e} {aoa:.2f}\n")
    print(f"Results (Tecplot format, AoA={aoa} deg) saved to {output_file}")


def CDCL(model, airfoil_points_aoa, rho, U_inf, p_inf, aoa_rad, chord, lb, ub):
    """
    ✅ FIXED: Coordinates are now properly denormalized before computing gradients
    """
    with torch.no_grad():
        pred = model(airfoil_points_aoa)
    p_star = pred[:, 2].cpu().numpy()
    p = p_star * rho * U_inf ** 2

    # ✅ FIXED: Denormalize airfoil coordinates from [-1, 1] to physical space
    x_norm = airfoil_points_aoa[:, 0].cpu().numpy()
    y_norm = airfoil_points_aoa[:, 1].cpu().numpy()
    x_phys = minmax_denormalize(x_norm, lb[0], ub[0])
    y_phys = minmax_denormalize(y_norm, lb[1], ub[1])

    # Compute gradients in physical space
    dx = np.gradient(x_phys)
    dy = np.gradient(y_phys)
    dl = np.sqrt(dx ** 2 + dy ** 2)
    nx = dy / (dl + 1e-12)
    ny = -dx / (dl + 1e-12)
    fx = -p * nx * dl
    fy = -p * ny * dl

    # Rotate to lift/drag
    drag = np.sum(fx * np.cos(aoa_rad) + fy * np.sin(aoa_rad))
    lift = np.sum(-fx * np.sin(aoa_rad) + fy * np.cos(aoa_rad))
    q_inf = 0.5 * rho * U_inf ** 2
    c_d = drag / (q_inf * chord)
    c_l = lift / (q_inf * chord)
    print(f"CD (drag coefficient): {c_d:.5f}, CL (lift coefficient): {c_l:.5f}")