#!/usr/bin/env python3
"""
batch_generation.py - Example of batch dataset generation

This example demonstrates programmatic batch generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topopt_ml.experiments import ExperimentManager
from topopt_ml.io import load_dataset_index


def main():
    print("=" * 60)
    print("Batch Dataset Generation Example")
    print("=" * 60)
    
    # Option 1: Use ExperimentManager with config file
    print("\nMethod 1: Using ExperimentManager")
    print("-" * 40)
    
    manager = ExperimentManager("config/default.yaml")
    
    # Generate 3 samples
    results = manager.run_batch(num_samples=3)
    
    print(f"\nGenerated {len(results)} samples")
    print(f"Output directory: {manager.exp_dir}")
    
    # Load and inspect results
    print("\nGenerated samples:")
    for sample in results:
        print(f"  {sample['sample_id']}: "
              f"solve_time={sample['solve_time']:.2f}s, "
              f"load_center={sample['load_center']}")
    
    # Option 2: Programmatic access
    print("\n" + "-" * 40)
    print("Method 2: Programmatic dataset iteration")
    print("-" * 40)
    
    from topopt_ml.io import iterate_samples
    
    for sample_id, X, Y in iterate_samples(str(manager.exp_dir)):
        print(f"Sample {sample_id}:")
        print(f"  Input shape:  {X.shape}")
        print(f"  Target shape: {Y.shape}")
        print(f"  Force sum:    {X[:,:,:,3].sum():.2f} N")
        print(f"  Density mean: {Y.mean():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Batch generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
