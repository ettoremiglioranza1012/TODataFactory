"""
Grid utilities for topology optimization.

Handles grid dimension calculations, memory alignment, and node indexing
matching the C solver implementation.
"""

import numpy as np
from typing import Tuple


# Stencil sizes for SIMD alignment (from C definitions.h)
STENCIL_SIZE_X = 1  # must be 1
STENCIL_SIZE_Y = 8  # set to 4 for AVX2, or 8 for AVX512
STENCIL_SIZE_Z = 1  # must be 1


class GridCalculator:
    """
    Calculate grid dimensions and indexing matching C memory layout.
    
    This class handles the wrap dimensions, padding, and node indexing
    that must match the C solver's internal grid representation.
    """
    
    def __init__(self, nelx: int, nely: int, nelz: int,
                 stencil_x: int = STENCIL_SIZE_X,
                 stencil_y: int = STENCIL_SIZE_Y,
                 stencil_z: int = STENCIL_SIZE_Z):
        """
        Initialize grid calculator.
        
        Args:
            nelx, nely, nelz: Number of elements in each direction
            stencil_x, stencil_y, stencil_z: Stencil sizes for SIMD alignment
        """
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        self.stencil_z = stencil_z
        
        # Calculate wrap dimensions
        self.wrapx, self.wrapy, self.wrapz = self._compute_wrap_dimensions()
        self.ndof = 3 * self.wrapx * self.wrapy * self.wrapz
    
    def _compute_wrap_dimensions(self) -> Tuple[int, int, int]:
        """
        Compute wrapped dimensions matching C initializeGridContext.
        
        Returns node dimensions with padding and halo cells.
        """
        # Padding for SIMD alignment
        paddingx = (self.stencil_x - ((self.nelx + 1) % self.stencil_x)) % self.stencil_x
        paddingy = (self.stencil_y - ((self.nely + 1) % self.stencil_y)) % self.stencil_y
        paddingz = (self.stencil_z - ((self.nelz + 1) % self.stencil_z)) % self.stencil_z
        
        # Add padding and 3 halo cells
        wrapx = self.nelx + paddingx + 3
        wrapy = self.nely + paddingy + 3
        wrapz = self.nelz + paddingz + 3
        
        return wrapx, wrapy, wrapz
    
    def node_index_to_flat(self, i: int, j: int, k: int) -> int:
        """
        Convert 3D node index to flat index using C scheme.
        
        Args:
            i, j, k: Node indices (0-based)
        
        Returns:
            Flat index in C memory layout
        """
        return i * self.wrapy * self.wrapz + self.wrapy * k + j
    
    def get_info(self) -> dict:
        """Get grid information as dictionary."""
        return {
            'elements': (self.nelx, self.nely, self.nelz),
            'nodes': (self.nelx + 1, self.nely + 1, self.nelz + 1),
            'wrap': (self.wrapx, self.wrapy, self.wrapz),
            'ndof': self.ndof,
            'stencil': (self.stencil_x, self.stencil_y, self.stencil_z)
        }


# Helper functions for backward compatibility
def compute_wrap_dimensions(nelx: int, nely: int, nelz: int) -> Tuple[int, int, int]:
    """Backward-compatible wrapper."""
    calc = GridCalculator(nelx, nely, nelz)
    return calc.wrapx, calc.wrapy, calc.wrapz


def compute_ndof(nelx: int, nely: int, nelz: int) -> int:
    """Backward-compatible wrapper."""
    calc = GridCalculator(nelx, nely, nelz)
    return calc.ndof


def node_index_to_flat(i: int, j: int, k: int, wrapy: int, wrapz: int) -> int:
    """Backward-compatible wrapper."""
    return i * wrapy * wrapz + wrapy * k + j
