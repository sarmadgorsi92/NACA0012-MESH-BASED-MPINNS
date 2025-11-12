#!/usr/bin/env python
"""
Visualization test to demonstrate the fix for mesh/airfoil alignment.
Shows the difference between the old (incorrect) and new (correct) reshape methods.
"""

import numpy as np
import matplotlib.pyplot as plt

def minmax_normalize(val, vmin, vmax):
    return 2.0 * (val - vmin) / (vmax - vmin) - 1.0

def read_structured_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    data = [line.strip().split() for line in lines[1:] if line.strip()]
    mesh_data = np.array([[float(val) for val in line] for line in data])
    x_mesh = mesh_data[:, 0]
    y_mesh = mesh_data[:, 1]
    return x_mesh, y_mesh, nx, ny, mesh_data

def get_airfoil_points(filename):
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
    return data[:, 0], data[:, 1]

# Read data
x_mesh, y_mesh, nx, ny, mesh_data = read_structured_mesh('MESH-POINTS.dat')
x_airfoil, y_airfoil = get_airfoil_points('BD-POINTS.dat')

lb = np.array([x_mesh.min(), y_mesh.min()])
ub = np.array([x_mesh.max(), y_mesh.max()])

# OLD (incorrect) reshape
X_old = x_mesh.reshape(nx, ny).T
Y_old = y_mesh.reshape(nx, ny).T

# NEW (correct) reshape
X_new = x_mesh.reshape(ny, nx)
Y_new = y_mesh.reshape(ny, nx)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot OLD method
ax = axes[0]
# Sample every 10th point for clarity
sample = 10
ax.scatter(X_old[::sample, ::sample], Y_old[::sample, ::sample], 
          s=1, alpha=0.3, c='blue', label='Mesh points')
ax.scatter(x_airfoil, y_airfoil, s=10, c='red', marker='o', 
          label='Airfoil boundary', zorder=5)
ax.set_xlabel('X (physical)')
ax.set_ylabel('Y (physical)')
ax.set_title('OLD (INCORRECT) METHOD\nreshape(nx, ny).T - Grid is scrambled', fontweight='bold')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim([-1, 2])
ax.set_ylim([-0.5, 0.5])

# Plot NEW method
ax = axes[1]
ax.scatter(X_new[::sample, ::sample], Y_new[::sample, ::sample], 
          s=1, alpha=0.3, c='blue', label='Mesh points')
ax.scatter(x_airfoil, y_airfoil, s=10, c='red', marker='o', 
          label='Airfoil boundary', zorder=5)
ax.set_xlabel('X (physical)')
ax.set_ylabel('Y (physical)')
ax.set_title('NEW (CORRECT) METHOD\nreshape(ny, nx) - Grid properly aligned', fontweight='bold')
ax.legend()
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim([-1, 2])
ax.set_ylim([-0.5, 0.5])

plt.tight_layout()
plt.savefig('reshape_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved comparison visualization to 'reshape_comparison.png'")
print("\nThe visualization shows:")
print("  LEFT (OLD): Mesh points are scrambled, don't form proper grid")
print("  RIGHT (NEW): Mesh points properly form structured grid around airfoil")
print("\nThis fix ensures that output files (results_AOA*.dat, cp_AOA*.dat)")
print("have correct (x,y) coordinate pairs, resolving the alignment issue.")
