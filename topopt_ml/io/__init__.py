"""I/O module for reading/writing files."""

from topopt_ml.io.binary import read_density_field
from topopt_ml.io.datasets import (
    load_sample,
    load_dataset_index,
    save_sample,
    save_dataset_index,
    iterate_samples,
)
from topopt_ml.io.vtk import density_to_vtk, create_mesh_from_density

__all__ = [
    'read_density_field',
    'load_sample',
    'load_dataset_index',
    'save_sample',
    'save_dataset_index',
    'iterate_samples',
    'density_to_vtk',
    'create_mesh_from_density',
]
