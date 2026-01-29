"""Visualization tools."""

from topopt_ml.visualization.pyvista_renderer import (
    render_density,
    render_loads,
    compare_input_output,
)
from topopt_ml.visualization.validation import (
    validate_sample,
    validate_sample_math,
    validate_sample_visual,
    check_physical_consistency,
    validate_dataset,
    print_validation_report,
)

__all__ = [
    # Rendering
    'render_density',
    'render_loads',
    'compare_input_output',
    # Validation
    'validate_sample',
    'validate_sample_math',
    'validate_sample_visual',
    'check_physical_consistency',
    'validate_dataset',
    'print_validation_report',
]

