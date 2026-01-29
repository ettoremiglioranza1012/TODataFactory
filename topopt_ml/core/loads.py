"""
Load generation for topology optimization.

Creates randomized load cases for ML dataset generation.
"""

import numpy as np
from typing import Tuple, Dict
from topopt_ml.core.grid import GridCalculator


class LoadFactory:
    """Generate randomized load cases for ML training."""
    
    def __init__(self, grid_calc: GridCalculator):
        """
        Initialize load factory.
        
        Args:
            grid_calc: GridCalculator instance for grid information
        """
        self.grid = grid_calc
    
    def generate_random_load(
        self,
        filename: str,
        total_force: float = -1000.0,
        center_min_frac: float = 0.2,
        center_max_frac: float = 0.8,
        radius_min: float = 3.0,
        radius_max: float = 5.0,
        random_seed: int = None
    ) -> Tuple[str, np.ndarray, Dict]:
        """
        Generate randomized distributed load case.
        
        Creates a random circular patch on the top face and generates
        ML input tensors for 3D U-Net training.
        
        Args:
            filename: Output filename for binary force vector
            total_force: Total force magnitude (negative = downward)
            center_min_frac: Min center position (fraction of domain)
            center_max_frac: Max center position (fraction of domain)
            radius_min: Min patch radius (elements)
            radius_max: Max patch radius (elements)
            random_seed: Random seed for reproducibility
        
        Returns:
            tuple: (binary_filename, input_tensor, metadata_dict)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        # Initialize force vector for C solver
        F_solver = np.zeros(self.grid.ndof, dtype=np.float64)
        
        # Number of nodes
        nx_nodes = nelx + 1
        ny_nodes = nely + 1
        nz_nodes = nelz + 1
        
        # Randomize load patch center
        cx_norm = np.random.uniform(center_min_frac, center_max_frac)
        cy_norm = np.random.uniform(center_min_frac, center_max_frac)
        
        cx = int(cx_norm * nelx)
        cy = int(cy_norm * nely)
        
        # Randomize patch radius
        radius = np.random.uniform(radius_min, radius_max)
        radius2 = radius * radius
        
        # Top face nodes (z = nelz)
        k = nz_nodes - 1
        
        # Find nodes in circular patch
        load_nodes = []
        for x in range(nx_nodes):
            for y in range(ny_nodes):
                dx = x - cx
                dy = y - cy
                dist2 = dx * dx + dy * dy
                
                if dist2 <= radius2:
                    # Convert to wrapped grid (add 1 for halo offset)
                    i = x + 1
                    j = y + 1
                    kk = k + 1
                    
                    nidx = self.grid.node_index_to_flat(i, j, kk)
                    load_nodes.append((nidx, x, y))
        
        num_load_nodes = len(load_nodes)
        if num_load_nodes == 0:
            raise ValueError(f"No nodes in load region! Center: ({cx}, {cy}), Radius: {radius}")
        
        # Distribute force evenly
        nodal_force = total_force / num_load_nodes
        
        # Apply force in z-direction
        for nidx, _, _ in load_nodes:
            dof_z = 3 * nidx + 2  # z-component
            F_solver[dof_z] = nodal_force
        
        # Save binary file
        F_solver.tofile(filename)
        
        # Generate ML input tensor (channels last)
        input_tensor = self._create_input_tensor(load_nodes, nodal_force, nelx, nely, nelz)
        
        # Metadata
        metadata = {
            'load_center': [int(cx), int(cy)],
            'load_radius': float(radius),
            'num_load_nodes': num_load_nodes,
            'nodal_force': float(nodal_force),
            'total_force': float(total_force),
            'random_seed': random_seed,
            'filename': filename
        }
        
        return (filename, input_tensor, metadata)
    
    def _create_input_tensor(
        self,
        load_nodes: list,
        nodal_force: float,
        nelx: int,
        nely: int,
        nelz: int
    ) -> np.ndarray:
        """
        Create 4-channel ML input tensor.
        
        Channels:
          0: Solid domain mask (all 1s)
          1: Force X (all 0s for vertical load)
          2: Force Y (all 0s for vertical load)
          3: Force Z (mapped from nodes to elements)
        """
        input_tensor = np.zeros((nelx, nely, nelz, 4), dtype=np.float32)
        
        # Channel 0: Solid mask
        input_tensor[:, :, :, 0] = 1.0
        
        # Channel 3: Force Z (map from nodes to elements)
        for _, node_x, node_y in load_nodes:
            # Each node is shared by up to 4 elements
            for elem_x in [node_x - 1, node_x]:
                for elem_y in [node_y - 1, node_y]:
                    if 0 <= elem_x < nelx and 0 <= elem_y < nely:
                        elem_z = nelz - 1  # Top element
                        # Divide by 4 since node is shared
                        input_tensor[elem_x, elem_y, elem_z, 3] += nodal_force / 4.0
        
        return input_tensor
