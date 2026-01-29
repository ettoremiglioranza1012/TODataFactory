#!/usr/bin/env python3
"""
basic_workflow.py - Example of using the TopOpt ML library.

This example demonstrates how to use the SolverInterface to run
topology optimization simulations.
"""

from pathlib import Path

from topopt_ml.core import GridCalculator, SolverInterface, LoadFactory


def run_single_example():
    """Run a single topology optimization example."""
    print("=" * 60)
    print("Basic TopOpt Workflow Example")
    print("=" * 60)
    
    # 1. Create grid calculator
    grid = GridCalculator(64, 32, 32)
    print(f"\n📐 Grid Configuration:")
    print(f"   Elements: {grid.nelx} × {grid.nely} × {grid.nelz}")
    print(f"   NDOF: {grid.ndof:,}")
    
    # 2. Create load factory and generate random load
    factory = LoadFactory(grid)
    print(f"\n🔧 Generating random load...")
    
    load_file, input_tensor, load_meta = factory.generate_random_load(
        filename="/tmp/example_load.bin",
        center_min_frac=0.3,
        center_max_frac=0.7,
        radius_min=3.0,
        radius_max=5.0,
        total_force=-1000.0,
        random_seed=42
    )
    
    print(f"   Load center: {load_meta['load_center']}")
    print(f"   Load radius: {load_meta['load_radius']:.2f}")
    print(f"   Nodal force: {load_meta['nodal_force']:.2f} N")
    print(f"   Load file: {load_file}")
    
    # 3. Run the solver
    print(f"\n🏭 Running TopOpt solver...")
    solver = SolverInterface()
    
    result = solver.run(
        grid_calc=grid,
        load_file=load_file,
        vol_frac=0.2,
        rmin=1.5,
        penal=3.0,
        max_iter=20
    )
    
    if result['success']:
        print(f"   ✅ Solver completed!")
        print(f"   Solve time: {result['solve_time']:.2f}s")
        
        # 4. Read output density
        density = solver.read_density_output(grid)
        print(f"\n📊 Result Statistics:")
        print(f"   Density shape: {density.shape}")
        print(f"   Volume fraction: {density.mean():.4f}")
        print(f"   Density range: [{density.min():.4f}, {density.max():.4f}]")
        
        # 5. Cleanup
        solver.cleanup_density_file()
        print(f"\n🧹 Cleaned up temporary files")
        
    else:
        print(f"   ❌ Solver failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)


def batch_example():
    """Run a batch of simulations with different parameters."""
    print("\n" + "=" * 60)
    print("Batch TopOpt Example")
    print("=" * 60)
    
    grid = GridCalculator(64, 32, 32)
    factory = LoadFactory(grid)
    solver = SolverInterface()
    
    # Different volume fractions
    vol_fracs = [0.1, 0.2, 0.3]
    
    for i, vol_frac in enumerate(vol_fracs, 1):
        print(f"\n[{i}/{len(vol_fracs)}] Volume fraction: {vol_frac}")
        print("-" * 40)
        
        # Generate load
        load_file, _, meta = factory.generate_random_load(
            filename=f"/tmp/batch_load_{i}.bin",
            random_seed=i
        )
        
        # Run solver
        result = solver.run(
            grid_calc=grid,
            load_file=load_file,
            vol_frac=vol_frac,
            max_iter=10  # Fewer iterations for batch
        )
        
        if result['success']:
            density = solver.read_density_output(grid)
            print(f"   ✅ Completed in {result['solve_time']:.2f}s")
            print(f"   Actual volume: {density.mean():.4f}")
            solver.cleanup_density_file()
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Batch complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_example()
    else:
        run_single_example()
