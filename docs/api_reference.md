# API Reference

## Core Module (`topopt_ml.core`)

### GridCalculator

```python
from topopt_ml.core import GridCalculator

calc = GridCalculator(
    nelx=64, nely=32, nelz=32,
    stencil_x=1, stencil_y=8, stencil_z=1
)

# Properties
calc.wrapx, calc.wrapy, calc.wrapz  # Wrapped dimensions
calc.ndof  # Total degrees of freedom

# Methods
flat_idx = calc.node_index_to_flat(i, j, k)
info = calc.get_info()
```

### LoadFactory

```python
from topopt_ml.core import LoadFactory

factory = LoadFactory(grid_calc)

filename, input_tensor, metadata = factory.generate_random_load(
    filename="load.bin",
    total_force=-1000.0,
    center_min_frac=0.2,
    center_max_frac=0.8,
    radius_min=3.0,
    radius_max=5.0,
    random_seed=42
)
```

### BCFactory

Generate configurable boundary conditions for diverse structural problems.

```python
from topopt_ml.core import BCFactory, BC_TYPES, get_bc_types

# Available BC types
print(BC_TYPES)  # ['cantilever_left', 'cantilever_right', 'simply_supported', 'corner_fixed', 'bridge']

bc_factory = BCFactory(grid_calc)

# Generate specific BC type
filename, metadata = bc_factory.generate_bc('simply_supported', 'bc.bin')

# Generate random BC with weights
weights = {
    'cantilever_left': 1.0,
    'cantilever_right': 1.0,
    'simply_supported': 1.0,
    'corner_fixed': 0.5,
    'bridge': 0.5
}
filename, metadata = bc_factory.generate_random_bc(
    filename='bc.bin',
    weights=weights,
    random_seed=42
)

# Metadata includes:
# - bc_type: str
# - num_fixed_dofs: int  
# - num_fixed_nodes: int
# - fixed_node_locations: dict
```

**BC Types:**
- `cantilever_left`: Left face fixed (classic cantilever beam)
- `cantilever_right`: Right face fixed (inverted cantilever)
- `simply_supported`: Z-displacement fixed on left/right edges
- `corner_fixed`: All DOFs fixed at 4 bottom corners
- `bridge`: Z-fixed along two parallel edge lines

### SolverInterface

```python
from topopt_ml.core import SolverInterface

solver = SolverInterface(solver_path="solver/top3d")

result = solver.run(
    grid_calc, load_file,
    vol_frac=0.2, rmin=1.5, max_iter=20, nl=4,
    bc_file="bc.bin"  # Optional: custom BC file
)

density = solver.read_density_output(grid_calc)
solver.cleanup_density_file()
```

## I/O Module (`topopt_ml.io`)

### Binary I/O

```python
from topopt_ml.io import read_density_field

density = read_density_field(nelx, nely, nelz, "density.bin")
```

### Dataset I/O

```python
from topopt_ml.io import load_sample, save_sample, iterate_samples

X, Y = load_sample("dataset_dir", "0001")
save_sample("output_dir", "0001", input_tensor, target_density)

for sample_id, X, Y in iterate_samples("dataset_dir"):
    # Process samples
    pass
```

## Config Module (`topopt_ml.config`)

```python
from topopt_ml.config import load_config

config = load_config("config/default.yaml")

config.grid  # Grid configuration dict
config.solver  # Solver configuration dict
config.ml  # ML generation settings
```

## Experiments Module (`topopt_ml.experiments`)

```python
from topopt_ml.experiments import ExperimentManager

manager = ExperimentManager("config/default.yaml")
results = manager.run_batch(num_samples=10)
```

## Visualization Module (`topopt_ml.visualization`)

```python
from topopt_ml.visualization import render_density, compare_input_output

# Render structure
render_density(density, threshold=0.3, screenshot="output.png")

# Side-by-side comparison
compare_input_output(input_tensor, target_density)
```
