"""
Boundary condition generation for topology optimization.

Creates configurable boundary conditions (fixed DOF patterns) for ML dataset generation.
Supports various structural configurations: cantilever, simply-supported, corner-fixed, bridge.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from topopt_ml.core.grid import GridCalculator


# Supported boundary condition types
BC_TYPES = [
    'cantilever_left',
    'cantilever_right', 
    'simply_supported',
    'corner_fixed',
    'bridge'
]


class BCFactory:
    """Generate configurable boundary conditions for ML training.
    
    Creates binary files containing fixed DOF indices that the C solver reads
    to apply appropriate displacement constraints.
    
    Attributes:
        grid: GridCalculator instance for grid information and indexing
    
    Example:
        >>> grid_calc = GridCalculator(64, 32, 32)
        >>> bc_factory = BCFactory(grid_calc)
        >>> filename, metadata = bc_factory.generate_bc('cantilever_left', 'bc.bin')
    """
    
    def __init__(self, grid_calc: GridCalculator):
        """
        Initialize BC factory.
        
        Args:
            grid_calc: GridCalculator instance for grid information
        """
        self.grid = grid_calc
    
    def generate_bc(
        self,
        bc_type: str,
        filename: str
    ) -> Tuple[str, Dict]:
        """
        Generate specific boundary condition type.
        
        Args:
            bc_type: Type of BC ('cantilever_left', 'cantilever_right', 
                    'simply_supported', 'corner_fixed', 'bridge')
            filename: Output filename for binary BC file
        
        Returns:
            tuple: (filename, metadata_dict)
        
        Raises:
            ValueError: If bc_type is not recognized
        """
        if bc_type not in BC_TYPES:
            raise ValueError(
                f"Unknown BC type: '{bc_type}'. "
                f"Supported types: {BC_TYPES}"
            )
        
        # Generate fixed DOFs based on type
        if bc_type == 'cantilever_left':
            fixed_dofs = self._generate_cantilever_left()
        elif bc_type == 'cantilever_right':
            fixed_dofs = self._generate_cantilever_right()
        elif bc_type == 'simply_supported':
            fixed_dofs = self._generate_simply_supported()
        elif bc_type == 'corner_fixed':
            fixed_dofs = self._generate_corner_fixed()
        elif bc_type == 'bridge':
            fixed_dofs = self._generate_bridge()
        
        # Remove duplicates and sort
        fixed_dofs = np.unique(fixed_dofs)
        
        # Validate DOF indices
        if np.any(fixed_dofs < 0) or np.any(fixed_dofs >= self.grid.ndof):
            raise ValueError(
                f"Invalid DOF indices generated. "
                f"Valid range: [0, {self.grid.ndof - 1}]"
            )
        
        # Save binary file (int32 array)
        fixed_dofs_int32 = fixed_dofs.astype(np.int32)
        fixed_dofs_int32.tofile(filename)
        
        # Generate metadata
        fixed_nodes = self._get_fixed_node_locations(fixed_dofs)
        
        metadata = {
            'bc_type': bc_type,
            'num_fixed_dofs': len(fixed_dofs),
            'num_fixed_nodes': len(set([d // 3 for d in fixed_dofs])),
            'fixed_node_locations': fixed_nodes,
            'filename': filename
        }
        
        return (filename, metadata)
    
    def generate_random_bc(
        self,
        bc_type: Optional[str] = None,
        filename: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[str, Dict]:
        """
        Generate randomized boundary condition.
        
        If bc_type is None, randomly selects from available types.
        
        Args:
            bc_type: Specific BC type, or None for random selection
            filename: Output filename for binary BC file
            weights: Optional probability weights for random type selection
            random_seed: Random seed for reproducibility
        
        Returns:
            tuple: (filename, metadata_dict)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Select BC type if not specified
        if bc_type is None:
            if weights is None:
                # Uniform random selection
                bc_type = np.random.choice(BC_TYPES)
            else:
                # Weighted random selection
                types = list(weights.keys())
                probs = np.array([weights.get(t, 0.0) for t in types])
                probs = probs / probs.sum()  # Normalize
                bc_type = np.random.choice(types, p=probs)
        
        return self.generate_bc(bc_type, filename)
    
    def _generate_cantilever_left(self) -> np.ndarray:
        """
        Generate cantilever BC with left face fixed.
        
        Fixes all DOFs (x, y, z) on the i=1 plane for bottom quarter of nodes.
        This matches the hardcoded behavior in grid_utilities.c
        
        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        # Number of nodes
        nz_nodes = nelz + 1
        ny_nodes = nely + 1
        
        # Fix bottom quarter plus top quarter (symmetric cantilever)
        nodelimit = (nely // 4) + 1
        
        i = 1  # Left face (halo offset)
        
        for k in range(1, nz_nodes + 1):
            # Bottom quarter
            for j in range(1, nodelimit + 1):
                nidx = self.grid.node_index_to_flat(i, j, k)
                fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1, 3 * nidx + 2])
            
            # Top quarter
            for j in range(ny_nodes + 1 - nodelimit, ny_nodes + 1):
                nidx = self.grid.node_index_to_flat(i, j, k)
                fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1, 3 * nidx + 2])
        
        return np.array(fixed_dofs, dtype=np.int64)
    
    def _generate_cantilever_right(self) -> np.ndarray:
        """
        Generate cantilever BC with right face fixed.
        
        Fixes all DOFs (x, y, z) on the i=nelx+1 plane for bottom quarter of nodes.
        
        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        # Number of nodes
        nz_nodes = nelz + 1
        ny_nodes = nely + 1
        
        # Fix bottom quarter plus top quarter (symmetric cantilever)
        nodelimit = (nely // 4) + 1
        
        i = nelx + 1  # Right face (halo offset)
        
        for k in range(1, nz_nodes + 1):
            # Bottom quarter
            for j in range(1, nodelimit + 1):
                nidx = self.grid.node_index_to_flat(i, j, k)
                fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1, 3 * nidx + 2])
            
            # Top quarter
            for j in range(ny_nodes + 1 - nodelimit, ny_nodes + 1):
                nidx = self.grid.node_index_to_flat(i, j, k)
                fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1, 3 * nidx + 2])
        
        return np.array(fixed_dofs, dtype=np.int64)
    
    def _generate_simply_supported(self) -> np.ndarray:
        """
        Generate simply-supported BC.
        
        Fixes z-component DOFs along edges at i=1 and i=nelx+1 planes,
        with x and y constrained at corners to prevent rigid body motion.
        
        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        # Fix z-displacement on left edge (bottom face, i=1)
        i = 1
        k = 1  # Bottom face
        for j in range(1, nely + 2):
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.append(3 * nidx + 2)  # z-component only
        
        # Fix z-displacement on right edge (bottom face, i=nelx+1)
        i = nelx + 1
        k = 1  # Bottom face
        for j in range(1, nely + 2):
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.append(3 * nidx + 2)  # z-component only
        
        # Fix x and y at corner nodes to prevent rigid body motion
        # Left-front corner
        nidx = self.grid.node_index_to_flat(1, 1, 1)
        fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1])
        
        # Left-back corner
        nidx = self.grid.node_index_to_flat(1, nely + 1, 1)
        fixed_dofs.append(3 * nidx + 0)  # Fix x to prevent rotation
        
        return np.array(fixed_dofs, dtype=np.int64)
    
    def _generate_corner_fixed(self) -> np.ndarray:
        """
        Generate corner-fixed BC.
        
        Fixes all DOFs at the 4 bottom corners of the domain.
        
        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []
        
        nelx, nely = self.grid.nelx, self.grid.nely
        
        # All 4 bottom corners (k=1 for bottom face)
        k = 1
        corners = [
            (1, 1),           # Front-left
            (nelx + 1, 1),    # Front-right
            (1, nely + 1),    # Back-left
            (nelx + 1, nely + 1)  # Back-right
        ]
        
        for i, j in corners:
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.extend([3 * nidx + 0, 3 * nidx + 1, 3 * nidx + 2])
        
        return np.array(fixed_dofs, dtype=np.int64)
    
    def _generate_bridge(self) -> np.ndarray:
        """
        Generate bridge BC.
        
        Fixes z-component along two parallel edge lines (left and right edges)
        on the bottom face, simulating bridge supports.
        
        Returns:
            Array of fixed DOF indices
        """
        fixed_dofs = []
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        k = 1  # Bottom face
        
        # Left support line (i=1, all j)
        i = 1
        for j in range(1, nely + 2):
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.append(3 * nidx + 2)  # z-component only
        
        # Right support line (i=nelx+1, all j)
        i = nelx + 1
        for j in range(1, nely + 2):
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.append(3 * nidx + 2)  # z-component only
        
        # Fix x at left edge nodes to prevent sliding
        i = 1
        for j in range(1, nely + 2):
            nidx = self.grid.node_index_to_flat(i, j, k)
            fixed_dofs.append(3 * nidx + 0)  # x-component
        
        # Fix y at front edge to prevent rotation
        j = 1
        nidx = self.grid.node_index_to_flat(1, j, k)
        fixed_dofs.append(3 * nidx + 1)  # y-component
        
        return np.array(fixed_dofs, dtype=np.int64)
    
    def _get_fixed_node_locations(self, fixed_dofs: np.ndarray) -> Dict[str, List]:
        """
        Get summary of fixed node locations.
        
        Args:
            fixed_dofs: Array of fixed DOF indices
        
        Returns:
            Dictionary with location summary
        """
        # Get unique node indices
        node_indices = np.unique(fixed_dofs // 3)
        
        # Determine which faces contain fixed nodes
        locations = {
            'planes': [],
            'edges': [],
            'corners': []
        }
        
        nelx, nely, nelz = self.grid.nelx, self.grid.nely, self.grid.nelz
        
        # Count nodes on each face
        left_count = 0
        right_count = 0
        bottom_count = 0
        
        for nidx in node_indices:
            # Reverse the node_index_to_flat to get coordinates
            # nidx = i * wrapy * wrapz + wrapy * k + j
            i = nidx // (self.grid.wrapy * self.grid.wrapz)
            rem = nidx % (self.grid.wrapy * self.grid.wrapz)
            k = rem // self.grid.wrapy
            j = rem % self.grid.wrapy
            
            if i == 1:
                left_count += 1
            if i == nelx + 1:
                right_count += 1
            if k == 1:
                bottom_count += 1
        
        if left_count > 0:
            locations['planes'].append('left (i=1)')
        if right_count > 0:
            locations['planes'].append('right (i=nelx)')
        if bottom_count > 0:
            locations['planes'].append('bottom (k=1)')
        
        return locations


def get_bc_types() -> List[str]:
    """Get list of supported BC types."""
    return BC_TYPES.copy()
