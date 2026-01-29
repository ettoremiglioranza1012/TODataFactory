#!/usr/bin/env python3
"""
TopOpt Orchestrator - Python wrapper for the top3d topology optimization solver.

This module provides a high-level interface for running topology optimization
simulations and collecting results for the Data Factory pipeline.
"""

import subprocess
import os
import re
from pathlib import Path


def run_topopt(
    nx: int,
    ny: int,
    nz: int,
    vol_frac: float,
    penal: float = 3.0,
    rmin: float = 1.5,
    output_file: str = "density.bin",
    working_dir: str = None
) -> dict:
    """
    Wrapper to call the C TopOpt executable.
    
    Args:
        nx: Number of elements in x direction
        ny: Number of elements in y direction
        nz: Number of elements in z direction
        vol_frac: Volume fraction constraint (0.0 to 1.0)
        penal: Penalization factor for SIMP method (default: 3.0)
        rmin: Filter radius in elements (default: 1.5)
        output_file: Output filename for density field (default: "density.bin")
        working_dir: Working directory for execution (default: script directory)
    
    Returns:
        dict: Results containing convergence info and output paths
    """
    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    exe_path = script_dir / "TopOpt-in-OpenMP" / "top3d"
    
    if working_dir is None:
        working_dir = script_dir / "TopOpt-in-OpenMP"
    else:
        working_dir = Path(working_dir)
    
    # Validate executable exists
    if not exe_path.exists():
        raise FileNotFoundError(f"TopOpt executable not found at: {exe_path}")
    
    # Construct command
    # Usage: ./top3d [nx] [ny] [nz] [vol_frac] [penal] [rmin]
    cmd = [
        str(exe_path),
        str(nx),
        str(ny),
        str(nz),
        str(vol_frac),
        str(penal),
        str(rmin)
    ]
    
    print(f"🏭 Factory: Running {' '.join(cmd)}...")
    print(f"   Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} elements")
    print(f"   Volume fraction: {vol_frac:.1%}")
    print(f"   Penalization: {penal}, Filter radius: {rmin}")
    
    # Execute
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(working_dir)
        )
        
        # Parse output
        stdout_lines = result.stdout.strip().splitlines()
        
        # Extract final iteration info
        final_line = stdout_lines[-1] if stdout_lines else ""
        
        # Extract timing info
        timing_line = [l for l in stdout_lines if "End time:" in l]
        total_time = float(timing_line[0].split(":")[1].strip()) if timing_line else None
        
        # Extract thread info
        thread_line = [l for l in stdout_lines if "OpenMP enabled" in l]
        num_threads = None
        if thread_line:
            match = re.search(r"(\d+) threads", thread_line[0])
            if match:
                num_threads = int(match.group(1))
        
        print("✅ Solver Finished.")
        print(f"   {final_line}")
        if total_time:
            print(f"   Total time: {total_time:.2f}s")
        
        return {
            "success": True,
            "stdout": result.stdout,
            "final_iteration": final_line,
            "total_time": total_time,
            "num_threads": num_threads,
            "grid_size": (nx, ny, nz),
            "volume_fraction": vol_frac,
            "working_dir": str(working_dir)
        }
        
    except subprocess.CalledProcessError as e:
        print("❌ Solver Failed!")
        print(f"   Exit code: {e.returncode}")
        print(f"   Error: {e.stderr}")
        
        return {
            "success": False,
            "error": e.stderr,
            "returncode": e.returncode,
            "grid_size": (nx, ny, nz),
            "volume_fraction": vol_frac
        }


def batch_run(
    configurations: list,
    output_dir: str = "results"
) -> list:
    """
    Run multiple TopOpt simulations in batch.
    
    Args:
        configurations: List of dicts with keys (nx, ny, nz, vol_frac, ...)
        output_dir: Directory to store results
    
    Returns:
        list: Results for each configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(configurations)
    
    print(f"🏭 Data Factory: Starting batch of {total} simulations\n")
    print("=" * 60)
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{total}] Configuration: {config}")
        print("-" * 60)
        
        result = run_topopt(**config)
        result["config_index"] = i
        results.append(result)
        
        print()
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print("=" * 60)
    print(f"🏁 Batch Complete: {successful}/{total} successful")
    
    return results


# Example Usage
if __name__ == "__main__":
    # Single run example
    result = run_topopt(60, 20, 20, 0.2)
    
    if result["success"]:
        print(f"\n📊 Summary:")
        print(f"   Grid: {result['grid_size']}")
        print(f"   Threads used: {result['num_threads']}")
        print(f"   Execution time: {result['total_time']:.2f}s")
