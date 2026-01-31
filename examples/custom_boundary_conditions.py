#!/usr/bin/env python3
"""
Example: Custom Boundary Conditions for Topology Optimization

This example demonstrates how to use the BCFactory class to generate
different boundary condition configurations for ML dataset generation.
"""

import tempfile
import os
import numpy as np
from pathlib import Path

# Add package to path if running from examples directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from topopt_ml.core import GridCalculator, BCFactory, LoadFactory, SolverInterface
from topopt_ml.core.boundary_conditions import BC_TYPES, get_bc_types


def demonstrate_bc_types():
    """Demonstrate all available boundary condition types."""
    print("=" * 60)
    print("Available Boundary Condition Types")
    print("=" * 60)
    
    # Create grid calculator for 64x32x32 domain
    grid_calc = GridCalculator(64, 32, 32)
    bc_factory = BCFactory(grid_calc)
    
    print(f"\nGrid: {grid_calc.nelx} x {grid_calc.nely} x {grid_calc.nelz}")
    print(f"Total DOFs: {grid_calc.ndof:,}")
    print()
    
    for bc_type in get_bc_types():
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, meta = bc_factory.generate_bc(bc_type, bc_file)
            
            print(f"📍 {bc_type}:")
            print(f"   Fixed DOFs: {meta['num_fixed_dofs']}")
            print(f"   Fixed Nodes: {meta['num_fixed_nodes']}")
            print(f"   Locations: {meta['fixed_node_locations']}")
            print()
        finally:
            os.remove(bc_file)


def demonstrate_random_bc_generation():
    """Demonstrate random BC generation with weights."""
    print("=" * 60)
    print("Random BC Generation with Weights")
    print("=" * 60)
    
    grid_calc = GridCalculator(64, 32, 32)
    bc_factory = BCFactory(grid_calc)
    
    # Custom weights favoring cantilever configurations
    weights = {
        'cantilever_left': 2.0,    # Higher probability
        'cantilever_right': 2.0,   # Higher probability
        'simply_supported': 1.0,
        'corner_fixed': 0.5,       # Lower probability
        'bridge': 1.0
    }
    
    print("\nWeights:", weights)
    print("\nGenerating 10 random BCs:")
    
    bc_counts = {t: 0 for t in BC_TYPES}
    
    for i in range(10):
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, meta = bc_factory.generate_random_bc(
                filename=bc_file,
                weights=weights,
                random_seed=1000 + i
            )
            bc_counts[meta['bc_type']] += 1
            print(f"  Sample {i+1}: {meta['bc_type']}")
        finally:
            os.remove(bc_file)
    
    print("\nDistribution:")
    for bc_type, count in bc_counts.items():
        print(f"  {bc_type}: {count}/10")


def demonstrate_solver_integration():
    """Demonstrate BC file integration with the solver."""
    print("=" * 60)
    print("Solver Integration with Custom BC")
    print("=" * 60)
    
    # Check if solver exists
    solver_path = Path(__file__).parent.parent / "solver" / "top3d"
    if not solver_path.exists():
        print("\n⚠️  Solver not found. Compile with 'make' in solver/ directory.")
        print(f"   Expected: {solver_path}")
        return
    
    grid_calc = GridCalculator(16, 8, 8)  # Small grid for quick demo
    bc_factory = BCFactory(grid_calc)
    load_factory = LoadFactory(grid_calc)
    solver = SolverInterface(solver_path)
    
    print(f"\nGrid: {grid_calc.nelx} x {grid_calc.nely} x {grid_calc.nelz}")
    
    # Generate BC file
    fd_bc, bc_file = tempfile.mkstemp(suffix=".bin", prefix="demo_bc_")
    os.close(fd_bc)
    
    # Generate load file
    fd_load, load_file = tempfile.mkstemp(suffix=".bin", prefix="demo_load_")
    os.close(fd_load)
    
    try:
        # Generate simply-supported BC
        _, bc_meta = bc_factory.generate_bc('simply_supported', bc_file)
        print(f"\nBC Type: {bc_meta['bc_type']}")
        print(f"Fixed DOFs: {bc_meta['num_fixed_dofs']}")
        
        # Generate random load
        _, _, load_meta = load_factory.generate_random_load(
            load_file,
            total_force=-100.0,
            random_seed=42
        )
        print(f"Load Center: {load_meta['load_center']}")
        print(f"Load Radius: {load_meta['load_radius']:.2f}")
        
        # Run solver with custom BC
        print("\nRunning solver with custom BC...")
        result = solver.run(
            grid_calc,
            load_file,
            vol_frac=0.3,
            max_iter=5,
            nl=2,
            bc_file=bc_file  # Custom BC file
        )
        
        if result['success']:
            print(f"✅ Solver completed in {result['solve_time']:.2f}s")
        else:
            print(f"❌ Solver failed: {result.get('error', 'Unknown')}")
            
    finally:
        os.remove(bc_file)
        os.remove(load_file)
        solver.cleanup_density_file()


def compare_bc_configurations():
    """Compare different BC types visually."""
    print("=" * 60)
    print("BC Configuration Comparison")
    print("=" * 60)
    
    grid_calc = GridCalculator(16, 8, 8)  # Small grid for visualization
    bc_factory = BCFactory(grid_calc)
    
    print(f"\nGrid: {grid_calc.nelx} x {grid_calc.nely} x {grid_calc.nelz}")
    print(f"Node dimensions: {grid_calc.nelx+1} x {grid_calc.nely+1} x {grid_calc.nelz+1}")
    print()
    
    print("BC Type Descriptions:")
    print("-" * 50)
    
    descriptions = {
        'cantilever_left': "Left face fixed (classic beam)",
        'cantilever_right': "Right face fixed (inverted beam)",
        'simply_supported': "Z-fixed on left/right edges (beam support)",
        'corner_fixed': "All DOFs at 4 bottom corners (plate)",
        'bridge': "Z-fixed along left/right lines (bridge deck)"
    }
    
    for bc_type in BC_TYPES:
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, meta = bc_factory.generate_bc(bc_type, bc_file)
            
            print(f"\n{bc_type}:")
            print(f"  Description: {descriptions[bc_type]}")
            print(f"  Fixed DOFs: {meta['num_fixed_dofs']}")
            print(f"  Fixed Nodes: {meta['num_fixed_nodes']}")
            print(f"  Constraint ratio: {meta['num_fixed_dofs'] / grid_calc.ndof * 100:.2f}%")
        finally:
            os.remove(bc_file)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Custom Boundary Conditions Example")
    print("=" * 60 + "\n")
    
    demonstrate_bc_types()
    print()
    
    demonstrate_random_bc_generation()
    print()
    
    compare_bc_configurations()
    print()
    
    demonstrate_solver_integration()
    print()
    
    print("=" * 60)
    print("Example Complete!")
    print("=" * 60)
