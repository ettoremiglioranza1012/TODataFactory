#!/usr/bin/env python3
"""
benchmark_solver.py - Performance benchmarking for the solver

Usage:
    python scripts/benchmark_solver.py --grid 64,32,32 --iters 5
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from topopt_ml.core import GridCalculator, LoadFactory, SolverInterface
import tempfile
import os


def benchmark_single(grid_calc: GridCalculator, solver: SolverInterface, 
                     load_factory: LoadFactory, max_iter: int = 5) -> dict:
    """Run single benchmark."""
    fd, load_file = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    
    try:
        # Generate load
        t0 = time.perf_counter()
        load_factory.generate_random_load(load_file, random_seed=42)
        load_time = time.perf_counter() - t0
        
        # Run solver
        t0 = time.perf_counter()
        result = solver.run(
            grid_calc, load_file,
            vol_frac=0.2, max_iter=max_iter
        )
        solve_time = time.perf_counter() - t0
        
        return {
            'success': result['success'],
            'load_gen_time': load_time,
            'solve_time': solve_time,
            'total_time': load_time + solve_time,
            'solver_reported_time': result.get('solve_time')
        }
    finally:
        if os.path.exists(load_file):
            os.remove(load_file)
        solver.cleanup_density_file()


def main():
    parser = argparse.ArgumentParser(description="Solver Benchmark")
    parser.add_argument("--grid", default="64,32,32", help="Grid size (nx,ny,nz)")
    parser.add_argument("--iters", type=int, default=5, help="Solver iterations")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Parse grid
    nx, ny, nz = map(int, args.grid.split(','))
    
    print("=" * 60)
    print(f"Solver Benchmark: {nx}×{ny}×{nz}, {args.iters} iterations")
    print("=" * 60)
    
    # Initialize
    grid_calc = GridCalculator(nx, ny, nz)
    solver = SolverInterface()
    load_factory = LoadFactory(grid_calc)
    
    print(f"Grid elements: {nx * ny * nz:,}")
    print(f"NDOF: {grid_calc.ndof:,}")
    print()
    
    # Run benchmarks
    results = []
    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}...", end=" ", flush=True)
        result = benchmark_single(grid_calc, solver, load_factory, args.iters)
        results.append(result)
        print(f"{result['total_time']:.2f}s")
    
    # Summary
    print()
    print("Results:")
    times = [r['total_time'] for r in results if r['success']]
    if times:
        print(f"  Mean: {sum(times)/len(times):.2f}s")
        print(f"  Min:  {min(times):.2f}s")
        print(f"  Max:  {max(times):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
