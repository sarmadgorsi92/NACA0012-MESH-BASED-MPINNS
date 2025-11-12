import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from model import MultiScalePINN
from losses import physics_loss, boundary_loss
from data_utils import generate_training_data, load_steady_data, read_structured_mesh, aoa_normalize
from viz_utils import write_results_to_file, CDCL, write_cp_tecplot
import multiprocessing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(123)
np.random.seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s) available.")
    if num_gpus > 1:
        print("Multiple GPUs detected. Enabling DataParallel for multi-GPU training.")
    else:
        print("Single GPU detected.")
else:
    print("No GPU available. Using CPU.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AOA-parameterized PINN for steady laminar flow around a NACA0012 airfoil')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of CPU threads to use (0 to use all available)')
    parser.add_argument('--aoa_min', type=float, default=0, help='Minimum angle of attack (deg)')
    parser.add_argument('--aoa_max', type=float, default=1, help='Maximum angle of attack (deg)')
    parser.add_argument('--aoa_step', type=float, default=1, help='AOA step (deg)')
    parser.add_argument('--mesh_file', type=str, default="MESH(C27K).dat", help='Path to structured mesh file')
    args = parser.parse_args()


    # Configure CPU parallelization
    num_threads = multiprocessing.cpu_count() if args.num_threads == 0 else args.num_threads
    
    # Set environment variables for thread control (must be done before importing heavy libraries)
    # This ensures consistent threading across all parallel backends
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    
    # Set PyTorch thread count
    torch.set_num_threads(num_threads)
    
    # Verify and report threading configuration
    actual_threads = torch.get_num_threads()
    available_cpus = multiprocessing.cpu_count()
    print(f"CPU Parallelization Configuration:")
    print(f"  Requested threads: {num_threads}")
    print(f"  PyTorch threads: {actual_threads}")
    print(f"  Available CPU cores: {available_cpus}")
    if num_threads > available_cpus:
        print(f"  ⚠️  WARNING: Requested threads ({num_threads}) exceeds available cores ({available_cpus})")
        print(f"     This may lead to oversubscription and reduced performance.")
    print()

 
    # --- READ MESH ONCE, ACCESS IN MAIN AND PASS TO DATA UTILS ---
    x_mesh, y_mesh, nx, ny, mesh_data = read_structured_mesh(args.mesh_file)
    lb = np.array([x_mesh.min(), y_mesh.min()])
    ub = np.array([x_mesh.max(), y_mesh.max()])
    mesh_info = {
        "x_mesh": x_mesh,
        "y_mesh": y_mesh,
        "nx": nx,
        "ny": ny,
        "mesh_data": mesh_data,
        "lb": lb,
        "ub": ub
    }
    chord = 1.0
    layers = [3,128,128, 128,128,128,128,3]
    activation = 'tanh'
    rho = 0.000544567
    nu = 0.0343
    U_inf = 171.5
    p_inf = 45.8

    Re = (U_inf * chord) / nu
    print(f"Reynolds Number: {Re}")

    num_epochs = 15000
    initial_lr = 1e-5
    min_lr = 1e-6
    patience = 500

    aoa_min = args.aoa_min
    aoa_max = args.aoa_max

    aoa_list = np.arange(aoa_min, aoa_max + 0.1, args.aoa_step)
    all_x_f, all_x_b, all_x_inlet, all_x_outlet, all_x_top, all_x_bottom = [], [], [], [], [], []
    all_data_coords, all_data_values = [], []
    for aoa in aoa_list:
        print(f"Preparing data for AOA={aoa}")
        x_f, x_b, x_inlet, x_outlet, x_top, x_bottom, airfoil_points = generate_training_data(
            aoa, mesh_info, aoa_min, aoa_max, airfoil_file="BD-POINTS.dat"
        )
        all_x_f.append(x_f.to(device))
        all_x_b.append(x_b.to(device))
        all_x_inlet.append(x_inlet.to(device))
        all_x_outlet.append(x_outlet.to(device))
        all_x_top.append(x_top.to(device))
        all_x_bottom.append(x_bottom.to(device))
        data_filename = f'ICEM-N12-AOA{int(aoa)}'
        print(f"DATA LOADED= ICEM-N12-AOA{int(aoa)}")
        data_coords, data_values = load_steady_data(data_filename, L=chord, U_inf=U_inf, rho=rho, aoa_deg=aoa, aoa_min=aoa_min, aoa_max=aoa_max, lb=lb, ub=ub)
        all_data_coords.append(data_coords)
        all_data_values.append(data_values)
        if aoa == aoa_list[0]:
            airfoil_points_ref = airfoil_points

    x_f = torch.cat(all_x_f, dim=0).to(device)
    x_b = torch.cat(all_x_b, dim=0).to(device)
    x_inlet = torch.cat(all_x_inlet, dim=0).to(device)
    x_outlet = torch.cat(all_x_outlet, dim=0).to(device)
    x_top = torch.cat(all_x_top, dim=0).to(device)
    x_bottom = torch.cat(all_x_bottom, dim=0).to(device)
    data_coords = torch.cat(all_data_coords, dim=0).to(device)
    data_values = torch.cat(all_data_values, dim=0).to(device)
    airfoil_points = airfoil_points_ref.to(device)

    model = MultiScalePINN(layers, activation=activation).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, min_lr=min_lr
    )

    losses = {'total': [], 'pde': [], 'bc': [], 'data': []}
    w_pde, w_bc, w_data = 10.0, 50.0, 100.0
    print("Starting Training: AdamW Optimization")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_pde = physics_loss(model, x_f, Re, lb, ub)
        loss_bc = boundary_loss(model, x_b, x_inlet, x_outlet, x_top, x_bottom, U_inf, p_inf, rho, aoa_min, aoa_max)
        pred_data = model(data_coords)
        loss_data = torch.mean((pred_data - data_values) ** 2)
        total_loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())
        losses['total'].append(total_loss.item())
        losses['pde'].append(loss_pde.item())
        losses['bc'].append(loss_bc.item())
        losses['data'].append(loss_data.item())
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, "
                f"BC Loss: {loss_bc.item():.6f}, Data Loss: {loss_data.item():.6f}")

    plt.figure(figsize=(12, 8))
    plt.semilogy(losses['total'], label='Total Loss')
    plt.semilogy(losses['pde'], label='PDE Loss')
    plt.semilogy(losses['bc'], label='BC Loss')
    plt.semilogy(losses['data'], label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History (AOA-param)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_history.png')
    plt.close()

    torch.save(model.state_dict(), f'FINAL-PINNS-MODEL.pth')

    user_aoas = [0,1,2,3]
    for aoa in user_aoas:
        print(f"Saving results for user AOA={aoa}")
        aoa_norm = aoa_normalize(aoa, aoa_min, aoa_max)
        aoa_tensor = torch.full((airfoil_points.shape[0], 1), aoa_norm, dtype=torch.float32, device=device)
        airfoil_points_aoa = torch.cat([airfoil_points[:, :2], aoa_tensor], dim=1)
        write_results_to_file(model, chord, U_inf, rho, output_file=f'results_AOA{aoa}.dat', aoa=aoa, mesh_info=mesh_info, aoa_min=aoa_min, aoa_max=aoa_max)
        write_cp_tecplot(model, airfoil_points_aoa, rho, U_inf, p_inf, lb, ub, chord=chord, output_file=f'cp_AOA{aoa}.dat')
        CDCL(model, airfoil_points_aoa, rho, U_inf, p_inf, np.radians(aoa), chord, lb, ub)