import torch
from torch.autograd import grad
import numpy as np

def physics_loss(model, x_f, Re, lb, ub):
    """
    Compute PDE residuals with proper scaling for normalized coordinates.
    """
    x_f = x_f.detach().requires_grad_(True)
    outputs = model(x_f)
    u_star = outputs[:, 0:1]
    v_star = outputs[:, 1:2]
    p_star = outputs[:, 2:3]

    u_grads = grad(u_star.sum(), x_f, create_graph=True)[0]
    v_grads = grad(v_star.sum(), x_f, create_graph=True)[0]
    p_grads = grad(p_star.sum(), x_f, create_graph=True)[0]
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    p_x, p_y = p_grads[:, 0:1], p_grads[:, 1:2]

    u_xx = grad(u_x.sum(), x_f, create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y.sum(), x_f, create_graph=True)[0][:, 1:2]
    v_xx = grad(v_x.sum(), x_f, create_graph=True)[0][:, 0:1]
    v_yy = grad(v_y.sum(), x_f, create_graph=True)[0][:, 1:2]

    # <<< MODIFIED: Apply scaling for normalized coordinates >>>
    scale_x = 2.0 / (ub[0] - lb[0])
    scale_y = 2.0 / (ub[1] - lb[1])

    u_x_phys = u_x * scale_x
    u_y_phys = u_y * scale_y
    v_x_phys = v_x * scale_x
    v_y_phys = v_y * scale_y
    p_x_phys = p_x * scale_x
    p_y_phys = p_y * scale_y

    u_xx_phys = u_xx * scale_x**2
    u_yy_phys = u_yy * scale_y**2
    v_xx_phys = v_xx * scale_x**2
    v_yy_phys = v_yy * scale_y**2

    continuity = u_x_phys + v_y_phys
    momentum_x = u_star * u_x_phys + v_star * u_y_phys + p_x_phys - (1 / Re) * (u_xx_phys + u_yy_phys)
    momentum_y = u_star * v_x_phys + v_star * v_y_phys + p_y_phys - (1 / Re) * (v_xx_phys + v_yy_phys)

    loss_continuity = torch.mean(continuity ** 2)
    loss_momentum_x = torch.mean(momentum_x ** 2)
    loss_momentum_y = torch.mean(momentum_y ** 2)
    total_physics_loss = loss_continuity + loss_momentum_x + loss_momentum_y
    return total_physics_loss

def boundary_loss(model, x_b, x_inlet, x_outlet, x_top, x_bottom, U_inf, p_inf, rho, aoa_min, aoa_max):
    """
    ✅ FIXED: AOA is now properly denormalized from [-1, 1] to degrees, then converted to radians
    """
    def get_u_v_star(x):
        aoa_norm = x[:, 2]  # Normalized AOA in [-1, 1]
        # Denormalize to degrees
        aoa_deg = 0.5 * (aoa_norm + 1.0) * (aoa_max - aoa_min) + aoa_min
        # Convert to radians
        aoa_rad = aoa_deg * np.pi / 180.0
        u_inf = torch.cos(aoa_rad)
        v_inf = torch.sin(aoa_rad)
        return u_inf, v_inf

    # Airfoil (no-slip)
    pred_b = model(x_b)
    loss_b = torch.mean(pred_b[:, 0] ** 2 + pred_b[:, 1] ** 2)

    # Inlet
    pred_inlet = model(x_inlet)
    u_inf_star, v_inf_star = get_u_v_star(x_inlet)
    loss_inlet = torch.mean((pred_inlet[:, 0] - u_inf_star) ** 2 + (pred_inlet[:, 1] - v_inf_star) ** 2)

    # Outlet (p*=p_inf_star)
    pred_outlet = model(x_outlet)
    p_inf_star = p_inf / (rho * U_inf ** 2)
    loss_outlet = torch.mean((pred_outlet[:, 2] - p_inf_star) ** 2)

    # Top
    pred_top = model(x_top)
    u_inf_star, v_inf_star = get_u_v_star(x_top)
    loss_top = torch.mean((pred_top[:, 0] - u_inf_star) ** 2 + (pred_top[:, 1] - v_inf_star) ** 2)

    # Bottom
    pred_bottom = model(x_bottom)
    u_inf_star, v_inf_star = get_u_v_star(x_bottom)
    loss_bottom = torch.mean((pred_bottom[:, 0] - u_inf_star) ** 2 + (pred_bottom[:, 1] - v_inf_star) ** 2)

    N_b, N_fs = x_b.shape[0], x_inlet.shape[0]
    w_b = 100.0 / N_b
    w_inlet = 100.0 / N_fs
    w_outlet = 1.0 / N_fs
    w_top = 1.0 / N_fs
    w_bottom = 1.0 / N_fs
    loss = w_b * loss_b + w_inlet * loss_inlet + w_outlet * loss_outlet + w_top * loss_top + w_bottom * loss_bottom

    return loss