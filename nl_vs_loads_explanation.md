# Understanding `nl` vs Number of Loads

## Question
> "Is current simulation running on multiple force loads (nl = 4)? If yes are the other 3 dummies or real?"

## Answer: **No, there is only ONE load per sample**

## What is `nl`?

**`nl` = Number of Multigrid Levels** (NOT number of loads!)

This is a parameter for the **numerical solver algorithm**, specifically for the multigrid method used to solve the finite element equations.

### Multigrid Hierarchy

When `nl = 4`, the solver creates a hierarchy of 4 mesh refinement levels:

```
Level 4 (finest):   64 × 32 × 32 elements  ← Your actual problem
Level 3:            32 × 16 × 16 elements
Level 2:            16 × 8  × 8  elements
Level 1 (coarsest):  8 × 4  × 4  elements  ← This is what you pass to the solver
```

The relationship is: **fine_grid = coarse_grid × 2^(nl-1)**

For `nl=4`: `fine_grid = coarse_grid × 2^3 = coarse_grid × 8`

### Why Use Multigrid?

Multigrid is a numerical technique to **solve linear systems more efficiently**:
1. Solves on coarse grid first (fast but inaccurate)
2. Refines solution on progressively finer grids
3. Each level corrects errors from the previous level
4. Result: Much faster convergence than solving directly on fine grid

## How Many Loads Are There?

**Exactly ONE load per sample**, defined by:

```python
# From run_factory.py - generate_random_load()
# Creates a SINGLE circular patch with random:
# - Center: (cx, cy) at 20-80% of domain
# - Radius: 3-5 elements
# - Total force: -1000 N (default)
```

### Evidence from Code

**1. Load Generation (Python):**
```python
# run_factory.py, line ~180
cx = np.random.randint(int(nelx * 0.2), int(nelx * 0.8))
cy = np.random.randint(int(nely * 0.2), int(nely * 0.8))
radius = np.random.uniform(3.0, 5.0)

# Creates ONE circular patch of nodes
dist = np.sqrt((ix - cx)**2 + (iy - cy)**2)
if dist <= radius:
    load_nodes.append((ix, iy, nelz))  # Top surface
```

**2. Force File (C++):**
```c
// stencil_optimization.c
// Reads a SINGLE force vector from binary file
fread(temp_buffer, sizeof(double), ndof, load_file);
memcpy(F, temp_buffer, ndof * sizeof(double));
```

**3. Dataset Metadata:**
```json
{
  "sample_id": "0001",
  "load_center": [27, 24],        // ONE center point
  "load_radius": 4.46,            // ONE circular region
  "num_load_nodes": 61,           // Nodes in this ONE patch
  "solve_time": 1.514
}
```

## Verification

You can verify this by checking any sample:

```python
import numpy as np

X = np.load("paired_dataset/sample_0001_inputs.npy")
fz = X[:, :, :, 3]  # Force Z channel

# Count non-zero force locations
loaded_elements = (np.abs(fz) > 1e-6).sum()
print(f"Loaded elements: {loaded_elements}")  # ~80 elements (one patch)

# Total force
total_force = fz.sum()
print(f"Total force: {total_force:.2f} N")  # -1000.00 N (one load)
```

## Summary

| Parameter | Meaning | Value | Purpose |
|-----------|---------|-------|---------|
| **nl** | Multigrid levels | 4 | Solver algorithm parameter |
| **Number of loads** | Load cases | **1** | Single randomized patch per sample |
| **Load center** | Patch location | Random (20-80%) | Varies per sample |
| **Load radius** | Patch size | 3-5 elements | Random per sample |
| **Total force** | Magnitude | -1000 N | Constant across samples |

## Why `nl=4` Was Chosen

From earlier debugging, we found:
- `nl=4` is the default and most stable configuration
- `nl=2` caused segmentation faults (pre-existing solver bug)
- `nl=4` means 8× grid refinement (2³ = 8)

This was **not** about number of loads, but about numerical solver stability.
