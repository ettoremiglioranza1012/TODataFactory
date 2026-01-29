"""
Solver interface for C++ TopOpt binary.

Provides a clean Python interface to execute the C++ solver.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Optional

from topopt_ml.core.grid import GridCalculator
from topopt_ml.core.loads import LoadFactory
from topopt_ml.io.binary import read_density_field


class SolverInterface:
    """Interface to the C++ topology optimization solver."""
    
    def __init__(self, solver_path: Optional[Path] = None):
        """
        Initialize solver interface.
        
        Args:
            solver_path: Path to top3d executable. If None, looks in default location.
        """
        if solver_path is None:
            # Default: look in solver/ relative to package
            self.solver_path = Path(__file__).parent.parent.parent / "solver" / "top3d"
        else:
            self.solver_path = Path(solver_path)
        
        if not self.solver_path.exists():
            raise FileNotFoundError(f"Solver not found at: {self.solver_path}")
    
    def run(
        self,
        grid_calc: GridCalculator,
        load_file: str,
        vol_frac: float = 0.2,
        rmin: float = 1.5,
        penal: float = 3.0,
        max_iter: int = 20,
        nl: int = 4
    ) -> Dict:
        """
        Execute the C++ solver.
        
        Args:
            grid_calc: GridCalculator with grid dimensions
            load_file: Path to binary load file
            vol_frac: Volume fraction constraint
            rmin: Filter radius
            penal: SIMP penalization
            max_iter: Maximum iterations
            nl: Number of multigrid levels
        
        Returns:
            Dictionary with success status, output, timing
        """
        # Calculate coarse grid dimensions
        size_incr = 2 ** (nl - 1)
        nelx_coarse = grid_calc.nelx // size_incr
        nely_coarse = grid_calc.nely // size_incr
        nelz_coarse = grid_calc.nelz // size_incr
        
        # Build command
        cmd = [
            str(self.solver_path),
            "-x", str(nelx_coarse),
            "-y", str(nely_coarse),
            "-z", str(nelz_coarse),
            "-v", str(vol_frac),
            "-r", str(rmin),
            "-i", str(max_iter),
            "-l", str(nl),
            "-f", load_file
        ]
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.solver_path.parent)
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr,
                'returncode': result.returncode
            }
        
        # Parse timing
        timing_line = [l for l in result.stdout.split('\n') if "End time:" in l]
        solve_time = float(timing_line[0].split(":")[1].strip()) if timing_line else None
        
        return {
            'success': True,
            'solve_time': solve_time,
            'stdout': result.stdout
        }
    
    def read_density_output(self, grid_calc: GridCalculator) -> 'np.ndarray':
        """Read the density.bin output file."""
        import numpy as np
        density_file = self.solver_path.parent / "density.bin"
        if not density_file.exists():
            raise FileNotFoundError(f"Density file not found: {density_file}")
        
        return read_density_field(
            grid_calc.nelx,
            grid_calc.nely,
            grid_calc.nelz,
            str(density_file)
        )
    
    def cleanup_density_file(self):
        """Remove the density.bin output file."""
        density_file = self.solver_path.parent / "density.bin"
        if density_file.exists():
            density_file.unlink()
