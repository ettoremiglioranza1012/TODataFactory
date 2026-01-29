# System Architecture

## Overview

TODataFactory is a production-grade ML pipeline for topology optimization, consisting of three main components:

1. **C Solver** - High-performance finite element solver
2. **Python Package** - ML data generation and management
3. **Configuration System** - Centralized parameter management

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Configuration                          │
│  config/default.yaml → ConfigLoader → Validation            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   ExperimentManager                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ GridCalculator │ │ LoadFactory │ │ SolverInterface │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌──────────────────────────────────────────────┐          │
│  │              Solver Execution                │          │
│  │   Python → Binary Load File → C top3d     │          │
│  │   C → Binary Density File → Python        │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Output                           │
│  sample_XXXX_inputs.npy (4-channel tensor)                 │
│  sample_XXXX_target.npy (density field)                    │
│  dataset_index.json (metadata)                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### GridCalculator
- Calculates wrapped grid dimensions matching C memory layout
- Handles SIMD alignment padding (stencil_size_y = 8 for AVX-512)
- Provides node-to-flat index conversion

### LoadFactory
- Generates randomized circular load patches
- Creates 4-channel ML input tensors
- Outputs binary force vectors for C solver

### SolverInterface
- Wraps C `top3d` executable
- Manages subprocess execution
- Reads binary density output

### ExperimentManager
- Orchestrates complete pipeline
- Creates versioned experiment directories
- Saves paired (input, target) samples

## Data Flow

1. **Config Loading** → YAML → Validated parameters
2. **Load Generation** → Random patch → Binary file + ML tensor
3. **Solver Execution** → C optimization → Binary density
4. **Post-Processing** → Read density → Save paired .npy files
5. **Tracking** → Metadata → dataset_index.json

## Memory Alignment

The grid calculator must match C exactly:

```
wrapped_dimension = element_count + padding + 3 (halo cells)
padding = (stencil_size - ((count + 1) % stencil_size)) % stencil_size
ndof = 3 * wrapx * wrapy * wrapz
```
