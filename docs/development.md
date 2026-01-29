# Developer Guide

## Setup

### Prerequisites
- Python 3.8+
- GCC 15 (for C++ solver)
- OpenBLAS and SuiteSparse

### Installation

```bash
# Clone repository
git clone https://github.com/yourname/TODataFactory.git
cd TODataFactory

# Install Python package
pip install -e ".[dev]"

# Build C++ solver
cd solver && make && cd ..
```

## Project Structure

```
topopt_ml/
├── core/           # Core algorithms
│   ├── grid.py     # Grid calculations
│   ├── loads.py    # Load generation
│   ├── solver.py   # C++ interface
│   └── tensors.py  # Tensor utilities
├── io/             # File I/O
├── config/         # Configuration
├── experiments/    # Experiment management
└── visualization/  # Rendering
```

## Adding New Features

### New Load Type

1. Add method to `LoadFactory` in `topopt_ml/core/loads.py`
2. Update `topopt_ml/core/__init__.py` exports
3. Add tests in `tests/test_loads.py`

### New Visualization

1. Add function to `topopt_ml/visualization/pyvista_renderer.py`
2. Update module `__init__.py`
3. Add example in `examples/`

## Code Style

- Follow PEP 8
- Use type hints
- Document with docstrings
- Run black for formatting

```bash
black topopt_ml/
flake8 topopt_ml/
```

## Testing

```bash
pytest tests/
pytest tests/test_grid.py -v
```

## Building Solver

```bash
cd solver
make clean
make
./top3d -h
```
