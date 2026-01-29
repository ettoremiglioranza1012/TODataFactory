"""
VTK export utilities for visualization.
"""

import numpy as np
from pathlib import Path


def density_to_vtk(
    density: np.ndarray,
    output_path: str,
    threshold: float = None
) -> str:
    """
    Export density field to VTK format.
    
    Args:
        density: Density array (nelx, nely, nelz)
        output_path: Output file path (.vtu or .vtk)
        threshold: Optional threshold to apply
    
    Returns:
        Path to created file
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required for VTK export: pip install pyvista")
    
    nelx, nely, nelz = density.shape
    
    # Create structured grid
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    grid.cell_data["density"] = density.flatten(order='F')
    
    # Apply threshold if specified
    if threshold is not None:
        grid = grid.threshold(value=threshold, scalars="density")
    
    # Save
    output = Path(output_path)
    grid.save(str(output))
    
    return str(output)


def create_mesh_from_density(density: np.ndarray, threshold: float = 0.3):
    """
    Create a mesh from density field.
    
    Args:
        density: Density array
        threshold: Density threshold
    
    Returns:
        PyVista mesh object
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    nelx, nely, nelz = density.shape
    
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    grid.cell_data["density"] = density.flatten(order='F')
    
    structure = grid.threshold(value=threshold, scalars="density")
    
    if structure.n_cells > 0:
        return structure.extract_geometry()
    
    return None
