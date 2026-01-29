#!/usr/bin/env python3
"""Test with all-zero load file"""
import numpy as np

STENCIL_SIZE_X = 1
STENCIL_SIZE_Y = 8
STENCIL_SIZE_Z = 1

def compute_wrap_dimensions(nelx, nely, nelz):
    paddingx = (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X
    paddingy = (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y
    paddingz = (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z
    
    wrapx = nelx + paddingx + 3
    wrapy = nely + paddingy + 3
    wrapz = nelz + paddingz + 3
    
    return wrapx, wrapy, wrapz

# Fine elements (matching what C++ will compute with coarse=16,8,8 nl=2)
nelx, nely, nelz = 32, 16, 16
wrapx, wrapy, wrapz = compute_wrap_dimensions(nelx, nely, nelz)
ndof = 3 * wrapx * wrapy * wrapz

# Create all-zero load vector
F = np.zeros(ndof, dtype=np.float64)

# Save it
F.tofile("test_load_zeros.bin")
print(f"Created test file with {ndof} DOFs ({ndof*8} bytes), all zeros")
