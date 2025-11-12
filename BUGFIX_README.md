# Mesh Reshape Bug Fix

## Problem

The mesh points and airfoil boundary points appeared misaligned in the output visualization files (`results_AOA*.dat`, `cp_AOA*.dat`). The user reported that "these points are not on actual positions" as shown in their figures.

## Root Cause

The MESH-POINTS.dat file stores structured mesh data in **row-major format**:
- 456 rows (ny) × 80 columns (nx) = 36,480 total points
- Data is linearized with x varying fastest within each row
- Each row has 80 points with the SAME y-coordinate but VARYING x-coordinates

The bug was in `viz_utils.py` in the `write_results_to_file` function:

```python
# OLD (INCORRECT) CODE:
X = x_mesh.reshape(nx, ny).T  # Reshape to (80, 456) then transpose
Y = y_mesh.reshape(nx, ny).T
```

This incorrect reshape caused:
1. The first row to have constant X values instead of varying X
2. Coordinate pairs (x, y) to be completely scrambled
3. Mesh points to appear misaligned with airfoil boundary points

## Solution

Changed the reshape to correctly match the data storage format:

```python
# NEW (CORRECT) CODE:
X = x_mesh.reshape(ny, nx)  # Reshape to (456, 80) - no transpose needed
Y = y_mesh.reshape(ny, nx)
```

Also fixed the reshape of solution fields (u, v, p) to match:

```python
# OLD (INCORRECT):
u = u.reshape(nx, ny).T
v = v.reshape(nx, ny).T
p = p.reshape(nx, ny).T

# NEW (CORRECT):
u = u.reshape(ny, nx)
v = v.reshape(ny, nx)
p = p.reshape(ny, nx)
```

## Additional Fixes

1. **Default mesh file**: Changed default `--mesh_file` parameter from non-existent `MESH(C27K).dat` to `MESH-POINTS.dat`

2. **Added .gitignore**: Created .gitignore file to exclude:
   - Python cache files (`__pycache__/`, `*.pyc`)
   - Temporary diagnostic files
   - Generated output files
   - Model files (`*.pth`)

## Testing

Created two test scripts to verify the fix:

1. **test_reshape_fix.py**: Validates that the new reshape method produces correct grid structure
   - Run: `python test_reshape_fix.py`
   - Expected: All tests pass ✅

2. **visualize_fix.py**: Creates side-by-side comparison visualization
   - Run: `python visualize_fix.py`
   - Output: `reshape_comparison.png` showing old (scrambled) vs new (correct) mesh

## Impact

After this fix:
- Output files (`results_AOA*.dat`, `cp_AOA*.dat`) will have correct (x, y) coordinate pairs
- Mesh points will be properly aligned with airfoil boundary points
- Flow field visualizations in Tecplot/ParaView will show correct spatial distribution
- Lift and drag coefficient calculations will use proper surface normal vectors

## Files Modified

1. `viz_utils.py`: Fixed mesh and solution field reshape (lines 53-54, 64-66)
2. `N12-BP-MP.py`: Updated default mesh_file parameter (line 37)
3. `.gitignore`: Created new file to exclude build artifacts

## Files Added

1. `test_reshape_fix.py`: Automated test for reshape correctness
2. `visualize_fix.py`: Visualization script showing before/after comparison
3. `reshape_comparison.png`: Visual demonstration of the fix
4. `.gitignore`: Git ignore rules
