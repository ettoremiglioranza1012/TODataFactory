"""
ML tensor utilities for topology optimization.

Functions for creating and manipulating tensors for ML training.
"""

import numpy as np
from typing import Tuple


def create_input_tensor(
    nelx: int,
    nely: int,
    nelz: int,
    load_nodes: list,
    nodal_force: float
) -> np.ndarray:
    """
    Create 4-channel ML input tensor from load nodes.
    
    Args:
        nelx, nely, nelz: Grid dimensions
        load_nodes: List of (nidx, node_x, node_y) tuples
        nodal_force: Force per node
    
    Returns:
        Tensor of shape (nelx, nely, nelz, 4)
    """
    input_tensor = np.zeros((nelx, nely, nelz, 4), dtype=np.float32)
    
    # Channel 0: Solid mask
    input_tensor[:, :, :, 0] = 1.0
    
    # Channel 3: Force Z
    for _, node_x, node_y in load_nodes:
        for elem_x in [node_x - 1, node_x]:
            for elem_y in [node_y - 1, node_y]:
                if 0 <= elem_x < nelx and 0 <= elem_y < nely:
                    elem_z = nelz - 1
                    input_tensor[elem_x, elem_y, elem_z, 3] += nodal_force / 4.0
    
    return input_tensor


def normalize_density(density: np.ndarray) -> np.ndarray:
    """Normalize density field to [0, 1] range."""
    return np.clip(density, 0.0, 1.0).astype(np.float32)


def threshold_density(density: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold density to binary solid/void."""
    return (density >= threshold).astype(np.float32)


def compute_volume_fraction(density: np.ndarray) -> float:
    """Compute volume fraction of structure."""
    return float(density.mean())


def get_tensor_stats(tensor: np.ndarray) -> dict:
    """Get statistics for a tensor."""
    return {
        'shape': tensor.shape,
        'dtype': str(tensor.dtype),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std())
    }
