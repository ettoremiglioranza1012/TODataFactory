#!/usr/bin/env python3
"""
Example: Batch dataset generation with randomized loads.

Generates multiple samples with different random load configurations
for ML training dataset creation.
"""

import sys
sys.path.append('.')

from run_factory import run_solver
from pathlib import Path
import numpy as np

def generate_ml_dataset(num_samples: int = 5, output_dir: str = "ml_dataset"):
    """
    Generate a batch of ML training samples with randomized loads.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory for tensors
    """
    print("=" * 60)
    print(f"ML Dataset Generation: {num_samples} samples")
    print("=" * 60)
    
    results = []
    
    for i in range(num_samples):
        print(f"\n[Sample {i+1}/{num_samples}]")
        print("-" * 60)
        
        result = run_solver(
            nelx=64,
            nely=32,
            nelz=32,
            vol_frac=0.2,
            iters=3,  # Fewer iterations for dataset generation
            random_seed=100 + i,  # Different seed for each sample
            save_input_tensor=True,
            output_dir=output_dir
        )
        
        if result["success"]:
            results.append(result)
            print(f"✅ Sample {i+1} completed in {result['total_time']:.2f}s")
        else:
            print(f"❌ Sample {i+1} failed!")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Dataset Generation Complete: {len(results)}/{num_samples} successful")
    print("=" * 60)
    
    if results:
        # Display load diversity
        print("\nLoad Configuration Diversity:")
        for i, r in enumerate(results):
            meta = r['load_metadata']
            cx, cy = meta['load_center']
            print(f"  Sample {i+1}: center=({cx:2d}, {cy:2d}), "
                  f"radius={meta['load_radius']:.2f}, "
                  f"nodes={meta['num_load_nodes']}")
        
        # List saved files
        print(f"\nSaved Tensors:")
        output_path = Path(output_dir)
        tensor_files = sorted(output_path.glob("input_X_*.npy"))
        for f in tensor_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
    
    return results


if __name__ == "__main__":
    # Generate a small batch for demonstration
    results = generate_ml_dataset(num_samples=3, output_dir="ml_dataset")
    
    print("\n" + "=" * 60)
    print("🎉 Ready for ML Training!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load tensors: X = np.load('ml_dataset/input_X_64x32x32_seed100.npy')")
    print("2. Run solver to get outputs (density fields)")
    print("3. Train 3D U-Net: X (loads) → Y (optimized topology)")
