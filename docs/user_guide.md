# HPC Experiment Manager - Usage Guide

## Overview

Centralized experiment management system for HPC topology optimization with:
- YAML-based configuration
- Versioned experiment tracking  
- Accurate C memory layout matching
- Automated workflow execution

## Files

### Configuration: `config.yaml`

```yaml
grid:
  nx: 64              # Elements in X
  ny: 32              # Elements in Y  
  nz: 32              # Elements in Z
  vol_fraction: 0.2   # Volume constraint

solver:
  nl: 4               # Multigrid levels (4 for 64-width grids)
  rmin: 1.5           # Filter radius
  penal: 3.0          # SIMP penalization
  max_iter: 20        # Iterations

hpc:
  stencil_size_y: 8   # AVX-512 alignment
  num_threads: 10     # OpenMP threads

ml:
  num_samples: 10     # Dataset size
  base_seed: 1000     # Random seed base
```

### Manager: `experiment_manager.py`

**Key Classes:**
- `ExperimentConfig`: YAML loader with validation
- `GridCalculator`: Accurate ndof calculation matching C
- `LoadFactory`: Randomized load generation
- `ExperimentManager`: Main orchestration

## Usage

### Basic Run
```bash
python experiment_manager.py
```

### Custom Configuration
```bash
python experiment_manager.py --config my_config.yaml
```

### Override Samples
```bash
python experiment_manager.py --samples 5
```

## Output Structure

```
experiments/
└── EXP_20260129_143758_64x32x32/
    ├── config.yaml                  # Configuration snapshot
    ├── dataset_index.json           # Sample metadata
    ├── sample_0001_inputs.npy       # Input tensor (64,32,32,4)
    ├── sample_0001_target.npy       # Density field (64,32,32)
    ├── sample_0002_inputs.npy
    └── sample_0002_target.npy
```

**Experiment Naming:** `EXP_{timestamp}_{nx}x{ny}x{nz}`

## Key Features

### 1. Accurate Memory Layout
Matches C `grid_utilities.c`:
```python
# Padding for SIMD alignment
padding_y = (stencil_y - ((ny + 1) % stencil_y)) % stencil_y

# Add halo cells
wrapy = ny + padding_y + 3

# Calculate NDOF
ndof = wrapx * wrapy * wrapz * 3
```

### 2. Randomized Load Generation
- **Shape**: Circular patch
- **Location**: Random (20-80% of domain)
- **Radius**: Random 3-5 elements
- **Force**: Distributed evenly over nodes

### 3. Versioned Experiments
- Timestamp-based folders
- Configuration snapshots
- Metadata tracking

### 4. ML-Ready Output
- Input: 4-channel tensor (solid + forces)
- Target: Density field
- NumPy `.npy` format

## Validation

```bash
# Check grid divisibility
nx % 2^(nl-1) == 0  # Must be true

# Verify NDOF
Expected: wrapx × wrapy × wrapz × 3

# Check load
sum(force_z) == -1000.0  # Total force
```

## Integration with Existing Code

The manager reuses proven components from `run_factory.py`:
- `read_density_field()` - Binary density reader
- Grid calculation logic
- Load generation patterns

## Logging

Set log level in `config.yaml`:
```yaml
output:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Best Practices

1. **Grid Sizing**: Always ensure divisibility by `2^(nl-1)`
2. **Multigrid**: Use `nl=4` for 64-width grids (stable)
3. **Stencil**: Set `stencil_size_y=8` for AVX-512
4. **Seeds**: Use different `base_seed` for train/val/test splits

## Example: Generate Training Dataset

```bash
# Edit config.yaml
ml:
  num_samples: 1000
  base_seed: 1000

# Run
python experiment_manager.py

# Results in:
# experiments/EXP_20260129_HHMMSS_64x32x32/
#   - 1000 paired samples
#   - dataset_index.json
```

## Performance

**Per Sample (64×32×32):**
- Grid calculation: <0.01s
- Load generation: ~0.1s
- Solver execution: ~1.5s
- Tensor export: ~0.2s
- **Total**: ~2s per sample

**1000 samples**: ~33 minutes
