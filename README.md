# TopOpt-ML: Topology Optimization for Machine Learning

A production-ready pipeline for generating training data for ML-based 3D topology optimization.

---

## Architecture Overview

This project follows a **single-flow architecture** designed around three pillars:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  scripts/                        CLI Entry Points                       │
│  ├── generate_dataset.py         → Dataset generation pipeline          │
│  └── validate_dataset.py         → Validation & visualization           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  configs/                        Configuration (YAML)                   │
│  └── experiment.yaml             → All parameters in one place          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  topopt_ml/                      Python Library                         │
│  ├── config/                     → Configuration loading                │
│  ├── core/                       → Grid, loads, solver interface        │
│  ├── io/                         → Data I/O (binary, numpy)             │
│  ├── experiments/                → Experiment management                │
│  └── visualization/              → Rendering & validation               │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  solver/                         C Solver                               │
│  └── top3d                       → High-performance topology optimizer  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  outputs/                        Data Storage                           │
│  └── experiment_name/            → Standardized experiment folder       │
│      ├── sample_XXX_inputs.npy   → ML input tensors (4 channels)        │
│      ├── sample_XXX_target.npy   → Optimized density (labels)           │
│      └── dataset_index.json      → Metadata for all samples             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Three Pillars

### 1. Decoupling (Python ↔ C)

- **Python** handles configuration, data orchestration, ML I/O, and visualization
- **C** handles the computationally intensive topology optimization solver
- Communication via **binary files** with guaranteed memory alignment
- Neither component knows about the other's internals

### 2. Configuration Management

- **Single YAML file** defines all experiment parameters
- Loaded once, passed through the entire pipeline
- No hardcoded values scattered across scripts

```yaml
# configs/experiment.yaml
grid:
  nelx: 64
  nely: 32
  nelz: 32

solver:
  vol_frac: 0.2
  rmin: 1.5
  max_iter: 20

generation:
  num_samples: 100
  random_seed: 42
```

### 3. Memory Alignment

- Binary format matches C struct layout exactly
- Fortran-order arrays for solver compatibility
- Consistent coordinate system across Python and C++

---

## Quick Start

### 1. Build the Solver

```bash
cd solver
make clean && make
```

### 2. Generate Dataset

```bash
# Edit config first
nano configs/experiment.yaml

# Run generation
uv run python scripts/generate_dataset.py --config configs/experiment.yaml --output outputs/my_experiment
```

### 3. Validate Results

```bash
# Quick math validation
uv run python scripts/validate_dataset.py outputs/my_experiment

# Visual validation
uv run python scripts/validate_dataset.py outputs/my_experiment --visual
```

### 4. Use in ML Training

```python
from topopt_ml.io import load_sample, iterate_samples

# Load single sample
X, Y = load_sample("outputs/my_experiment", "001")

# Iterate all samples
for sample_id, X, Y in iterate_samples("outputs/my_experiment"):
    # X.shape = (64, 32, 32, 4)  - input channels
    # Y.shape = (64, 32, 32)     - target density
    pass
```

---

## Directory Structure

```
NewAM/
├── configs/              # Configuration files (YAML)
├── scripts/              # User-facing CLI tools
├── topopt_ml/            # Python library (the core)
│   ├── config/           # Config loading
│   ├── core/             # Grid, loads, solver interface
│   ├── io/               # File I/O (binary, numpy, VTK)
│   ├── experiments/      # Experiment management
│   └── visualization/    # PyVista rendering & validation
├── solver/               # C topology optimization solver
├── examples/             # API usage examples (not for production)
├── notebooks/            # Jupyter notebooks for exploration
└── outputs/              # Generated datasets (gitignored)
```

---

## Input/Output Format

### Input Tensor (4 channels)
| Channel | Description |
|---------|-------------|
| 0 | Solid domain mask (1 = element exists) |
| 1 | Fx - Force in x direction |
| 2 | Fy - Force in y direction |
| 3 | Fz - Force in z direction |

### Target Tensor
- Shape: `(nelx, nely, nelz)`
- Values: Continuous density ∈ [0, 1]
- 0 = void, 1 = solid material

---

## Examples vs Scripts

| Folder | Purpose | When to use |
|--------|---------|-------------|
| `scripts/` | Production pipelines | Generating datasets, training |
| `examples/` | API tutorials | Learning the library |
| `notebooks/` | Interactive exploration | Debugging, visualization |

---

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/

# Check imports
uv run python -c "from topopt_ml import core, io, config, visualization; print('OK')"
```

---

## License

MIT
