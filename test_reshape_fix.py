#!/usr/bin/env python
"""
Test script to verify that the mesh reshape fix works correctly.
This script tests that the mesh coordinates are properly structured after the fix.
"""

import numpy as np

def read_structured_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        lines = f.readlines()
    nx, ny = map(int, lines[0].split())
    data = [line.strip().split() for line in lines[1:] if line.strip()]
    mesh_data = np.array([[float(val) for val in line] for line in data])
    x_mesh = mesh_data[:, 0]
    y_mesh = mesh_data[:, 1]
    return x_mesh, y_mesh, nx, ny, mesh_data

def test_reshape():
    """Test that the mesh is reshaped correctly."""
    print("Testing mesh reshape fix...")
    print("=" * 60)
    
    # Read mesh
    x_mesh, y_mesh, nx, ny, mesh_data = read_structured_mesh('MESH-POINTS.dat')
    print(f"\nMesh dimensions: nx={nx}, ny={ny}, total={len(x_mesh)} points")
    
    # Test the OLD (incorrect) way
    print("\n1. OLD (INCORRECT) METHOD: X = x_mesh.reshape(nx, ny).T")
    X_old = x_mesh.reshape(nx, ny).T
    Y_old = y_mesh.reshape(nx, ny).T
    print(f"   Shape: {X_old.shape}")
    print(f"   First row X values - should vary, but got:")
    print(f"     min={X_old[0, :].min():.6f}, max={X_old[0, :].max():.6f}, unique={len(np.unique(X_old[0, :]))}")
    print(f"   First row Y values - should be constant:")
    print(f"     min={Y_old[0, :].min():.6f}, max={Y_old[0, :].max():.6f}, unique={len(np.unique(Y_old[0, :]))}")
    
    if len(np.unique(X_old[0, :])) == 1:
        print("   ❌ FAIL: First row has constant X (should vary)!")
    else:
        print("   ✓ PASS: First row X varies")
        
    # Test the NEW (correct) way
    print("\n2. NEW (CORRECT) METHOD: X = x_mesh.reshape(ny, nx)")
    X_new = x_mesh.reshape(ny, nx)
    Y_new = y_mesh.reshape(ny, nx)
    print(f"   Shape: {X_new.shape}")
    print(f"   First row X values - should vary:")
    print(f"     min={X_new[0, :].min():.6f}, max={X_new[0, :].max():.6f}, unique={len(np.unique(X_new[0, :]))}")
    print(f"   First row Y values - should be constant:")
    print(f"     min={Y_new[0, :].min():.6f}, max={Y_new[0, :].max():.6f}, unique={len(np.unique(Y_new[0, :]))}")
    
    first_row_x_varies = len(np.unique(X_new[0, :])) > 1
    first_row_y_constant = len(np.unique(Y_new[0, :])) == 1
    first_col_y_varies = len(np.unique(Y_new[:, 0])) > 1
    # Note: first_col_x does not need to be constant for body-fitted grids
    
    print("\n3. VALIDATION (for structured body-fitted mesh):")
    print(f"   First row X varies: {first_row_x_varies} (should be True)")
    print(f"   First row Y constant: {first_row_y_constant} (should be True)")
    print(f"   First column Y varies: {first_col_y_varies} (should be True)")
    print(f"   Note: X in a column can vary for body-fitted grids (this is normal)")
    
    if first_row_x_varies and first_row_y_constant and first_col_y_varies:
        print("\n✅ ALL TESTS PASSED! The new reshape method is correct.")
        return True
    else:
        print("\n❌ TESTS FAILED! The reshape method needs further investigation.")
        return False

if __name__ == "__main__":
    success = test_reshape()
    exit(0 if success else 1)
