"""
PyVista 3D rendering utilities.
"""

import numpy as np
from typing import Optional, Tuple


def render_density(
    density: np.ndarray,
    threshold: float = 0.3,
    smoothing: int = 50,
    color: str = 'steelblue',
    show: bool = True,
    screenshot: Optional[str] = None
):
    """
    Render density field as 3D mesh.
    
    Args:
        density: Density array (nelx, nely, nelz)
        threshold: Density threshold
        smoothing: Laplacian smoothing iterations
        color: Mesh color
        show: Display interactive plot
        screenshot: Optional screenshot path
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    nelx, nely, nelz = density.shape
    
    # Create grid
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    grid.cell_data["density"] = density.flatten(order='F')
    
    # Threshold
    structure = grid.threshold(value=threshold, scalars="density")
    
    if structure.n_cells == 0:
        print("Warning: No structure found at this threshold")
        return None
    
    # Extract and smooth surface
    surface = structure.extract_geometry()
    if smoothing > 0:
        surface = surface.smooth(n_iter=smoothing, relaxation_factor=0.1)
    
    # Create plotter
    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(window_size=(1200, 900), off_screen=off_screen)
    
    plotter.add_mesh(
        surface,
        color=color,
        smooth_shading=True,
        specular=0.5,
        specular_power=20,
        ambient=0.2
    )
    
    plotter.set_background('white')
    plotter.camera_position = 'iso'
    plotter.add_axes()
    
    if screenshot:
        plotter.screenshot(screenshot)
    
    if show:
        plotter.show()
    else:
        plotter.close()
    
    return surface


def render_loads(
    input_tensor: np.ndarray,
    force_channel: int = 3,
    show: bool = True,
    screenshot: Optional[str] = None
):
    """
    Render load visualization with arrows.
    
    Args:
        input_tensor: Input tensor (nelx, nely, nelz, 4)
        force_channel: Force channel index
        show: Display interactive plot
        screenshot: Optional screenshot path
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    fz = input_tensor[:, :, :, force_channel]
    nelx, nely, nelz = fz.shape
    
    # Create grid
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    grid.cell_data["Fz"] = fz.flatten(order='F')
    
    # Get loaded region
    loaded = grid.threshold(value=(-1000, -1e-6), scalars="Fz")
    
    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(window_size=(1200, 900), off_screen=off_screen)
    
    if loaded.n_cells > 0:
        plotter.add_mesh(loaded, scalars="Fz", cmap="Reds", opacity=0.6)
        
        # Add arrows
        centers = loaded.cell_centers()
        arrows = centers.glyph(
            orient=False,
            scale=False,
            factor=3.0,
            geom=pv.Arrow(direction=(0, 0, -1), scale=2)
        )
        plotter.add_mesh(arrows, color='crimson', opacity=0.9)
    
    plotter.set_background('white')
    plotter.camera_position = 'iso'
    plotter.add_axes()
    plotter.add_title("Applied Loads", font_size=14)
    
    if screenshot:
        plotter.screenshot(screenshot)
    
    if show:
        plotter.show()
    else:
        plotter.close()


def compare_input_output(
    input_tensor: np.ndarray,
    target_density: np.ndarray,
    threshold: float = 0.3,
    show: bool = True,
    screenshot: Optional[str] = None
):
    """
    Side-by-side comparison of input and output.
    
    Args:
        input_tensor: Input tensor
        target_density: Target density
        threshold: Density threshold
        show: Display plot
        screenshot: Optional screenshot path
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    fz = input_tensor[:, :, :, 3]
    nelx, nely, nelz = fz.shape
    
    off_screen = screenshot is not None and not show
    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 800), off_screen=off_screen)
    
    # Left: Loads
    plotter.subplot(0, 0)
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    grid.cell_data["Fz"] = fz.flatten(order='F')
    loaded = grid.threshold(value=(-1000, -1e-6), scalars="Fz")
    
    if loaded.n_cells > 0:
        plotter.add_mesh(loaded, scalars="Fz", cmap="coolwarm", opacity=0.7)
        centers = loaded.cell_centers()
        arrows = centers.glyph(orient=False, scale=False, factor=2.5,
                               geom=pv.Arrow(direction=(0, 0, -1)))
        plotter.add_mesh(arrows, color='red', opacity=0.8)
    
    plotter.add_title("INPUT: Applied Loads", font_size=12)
    plotter.set_background('white')
    plotter.camera_position = 'iso'
    
    # Right: Structure
    plotter.subplot(0, 1)
    grid2 = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid2.spacing = (1, 1, 1)
    grid2.cell_data["density"] = target_density.flatten(order='F')
    structure = grid2.threshold(value=threshold, scalars="density")
    
    if structure.n_cells > 0:
        surface = structure.extract_geometry()
        surface = surface.smooth(n_iter=30, relaxation_factor=0.1)
        plotter.add_mesh(surface, color='steelblue', smooth_shading=True, specular=0.5)
    
    plotter.add_title(f"OUTPUT: Structure (ρ > {threshold})", font_size=12)
    plotter.set_background('white')
    plotter.camera_position = 'iso'
    
    if screenshot:
        plotter.screenshot(screenshot)
    
    if show:
        plotter.show()
    else:
        plotter.close()
