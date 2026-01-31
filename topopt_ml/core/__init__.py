"""Core module for topology optimization."""

from topopt_ml.core.grid import (
    GridCalculator,
    compute_wrap_dimensions,
    compute_ndof,
    STENCIL_SIZE_X,
    STENCIL_SIZE_Y,
    STENCIL_SIZE_Z,
)
from topopt_ml.core.loads import LoadFactory
from topopt_ml.core.boundary_conditions import BCFactory, BC_TYPES, get_bc_types
from topopt_ml.core.solver import SolverInterface
from topopt_ml.core.tensors import (
    create_input_tensor,
    normalize_density,
    threshold_density,
    compute_volume_fraction,
    get_tensor_stats,
)

__all__ = [
    'GridCalculator',
    'LoadFactory',
    'BCFactory',
    'BC_TYPES',
    'get_bc_types',
    'SolverInterface',
    'compute_wrap_dimensions',
    'compute_ndof',
    'create_input_tensor',
    'normalize_density',
    'threshold_density',
    'compute_volume_fraction',
    'get_tensor_stats',
    'STENCIL_SIZE_X',
    'STENCIL_SIZE_Y',
    'STENCIL_SIZE_Z',
]

