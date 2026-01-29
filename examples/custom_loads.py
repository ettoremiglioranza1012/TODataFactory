#!/usr/bin/env python3
"""
custom_loads.py - Example of creating custom load configurations

This example shows how to use the LoadFactory for different load patterns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from topopt_ml.core import GridCalculator, LoadFactory


def main():
    # Create grid calculator
    grid = GridCalculator(64, 32, 32)
    factory = LoadFactory(grid)
    
    print("Grid Configuration:")
    print(f"  Elements: {grid.nelx} × {grid.nely} × {grid.nelz}")
    print(f"  NDOF: {grid.ndof:,}")
    print()
    
    # Example 1: Center load
    print("Example 1: Centered load")
    _, tensor1, meta1 = factory.generate_random_load(
        "/tmp/load1.bin",
        center_min_frac=0.45,
        center_max_frac=0.55,
        radius_min=3.0,
        radius_max=3.0,
        random_seed=1
    )
    print(f"  Center: {meta1['load_center']}")
    print(f"  Radius: {meta1['load_radius']:.2f}")
    print()
    
    # Example 2: Off-center load
    print("Example 2: Off-center load")
    _, tensor2, meta2 = factory.generate_random_load(
        "/tmp/load2.bin",
        center_min_frac=0.1,
        center_max_frac=0.3,
        radius_min=4.0,
        radius_max=5.0,
        random_seed=2
    )
    print(f"  Center: {meta2['load_center']}")
    print(f"  Radius: {meta2['load_radius']:.2f}")
    print()
    
    # Example 3: Small concentrated load
    print("Example 3: Concentrated load")
    _, tensor3, meta3 = factory.generate_random_load(
        "/tmp/load3.bin",
        radius_min=1.5,
        radius_max=2.0,
        total_force=-500.0,
        random_seed=3
    )
    print(f"  Center: {meta3['load_center']}")
    print(f"  Nodes: {meta3['num_load_nodes']}")
    print(f"  Force/node: {meta3['nodal_force']:.2f} N")
    print()
    
    # Compare tensors
    print("Tensor Force Comparison:")
    for i, (t, m) in enumerate([(tensor1, meta1), (tensor2, meta2), (tensor3, meta3)], 1):
        fz = t[:, :, :, 3]
        print(f"  Load {i}: Total Fz = {fz.sum():.2f} N, "
              f"Max element = {fz.min():.4f} N")


if __name__ == "__main__":
    main()
