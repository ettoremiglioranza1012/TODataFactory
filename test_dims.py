#!/usr/bin/env python3
"""Test to verify dimension calculations match C++"""

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

# Test case: coarse = 16, 8, 8 with nl=2
nelx_coarse = 16
nely_coarse = 8
nelz_coarse = 8
nl = 2

# Compute sizeIncr (matches C++ logic)
sizeIncr = 2
for i in range(2, nl):
    sizeIncr *= 2

print(f"nl={nl}, sizeIncr={sizeIncr}")

# Fine elements
nelx = nelx_coarse * sizeIncr
nely = nely_coarse * sizeIncr
nelz = nelz_coarse * sizeIncr

print(f"Fine elements: {nelx} × {nely} × {nelz}")

# Compute wrap dimensions
wrapx, wrapy, wrapz = compute_wrap_dimensions(nelx, nely, nelz)
ndof = 3 * wrapx * wrapy * wrapz

print(f"Padding for SIMD:")
print(f"  paddingx = (1 - ({nelx+1} % 1)) % 1 = {(STENCIL_SIZE_X - ((nelx+1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X}")
print(f"  paddingy = (8 - ({nely+1} % 8)) % 8 = {(STENCIL_SIZE_Y - ((nely+1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y}")
print(f"  paddingz = (1 - ({nelz+1} % 1)) % 1 = {(STENCIL_SIZE_Z - ((nelz+1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z}")

print(f"\nWrap dimensions: {wrapx} × {wrapy} × {wrapz}")
print(f"Total DOFs: {ndof}")
print(f"File size: {ndof * 8} bytes = {ndof * 8 / 1024:.2f} KB")

print(f"\nExpected by C++: 51870 DOFs, 414960 bytes")
print(f"Match: {ndof == 51870}")
