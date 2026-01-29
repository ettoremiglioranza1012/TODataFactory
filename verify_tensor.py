#!/usr/bin/env python3
"""
Test script to verify ML input tensor generation and batch processing.
"""

import numpy as np
from pathlib import Path

# Load and inspect the saved tensor
tensor_path = Path("ml_dataset/input_X_64x32x32_seed42.npy")

if tensor_path.exists():
    X = np.load(tensor_path)
    print("=" * 60)
    print("ML Input Tensor Verification")
    print("=" * 60)
    print(f"\nLoaded tensor from: {tensor_path}")
    print(f"Shape: {X.shape} (channels-last: nelx, nely, nelz, 4)")
    print(f"Dtype: {X.dtype}")
    print(f"Memory: {X.nbytes / 1024:.2f} KB")
    
    print("\nChannel Statistics:")
    for ch, name in enumerate(['Solid Mask', 'Force X', 'Force Y', 'Force Z']):
        data = X[:, :, :, ch]
        print(f"  {name} (ch {ch}):")
        print(f"    Min: {data.min():.6f}, Max: {data.max():.6f}")
        print(f"    Mean: {data.mean():.6f}, Std: {data.std():.6f}")
        print(f"    Sum: {data.sum():.2f}")
        print(f"    Non-zero: {np.count_nonzero(data)}/{data.size}")
    
    # Verify force is only on top face
    fz = X[:, :, :, 3]
    print(f"\nForce Z Distribution:")
    for z in range(X.shape[2]):
        layer_sum = fz[:, :, z].sum()
        if abs(layer_sum) > 1e-6:
            print(f"  z={z}: sum={layer_sum:.2f} N")
    
    # Visualize load patch location
    top_layer = fz[:, :, -1]
    load_x, load_y = np.where(top_layer != 0)
    if len(load_x) > 0:
        print(f"\nLoad Patch on Top Face (z={X.shape[2]-1}):")
        print(f"  X range: {load_x.min()} to {load_x.max()}")
        print(f"  Y range: {load_y.min()} to {load_y.max()}")
        print(f"  Center (approx): ({load_x.mean():.1f}, {load_y.mean():.1f})")
        print(f"  Loaded elements: {len(load_x)}")
    
    print("\n" + "=" * 60)
    print("✅ Tensor verification complete!")
    print("=" * 60)
else:
    print(f"❌ Tensor file not found: {tensor_path}")
