"""Tests for SolverInterface."""

import pytest
from pathlib import Path

from topopt_ml.core.solver import SolverInterface


class TestSolverInterface:
    """Test suite for SolverInterface."""
    
    def test_solver_discovery(self):
        """Test that solver binary is found."""
        solver = SolverInterface()
        assert solver.solver_path.exists()
    
    def test_solver_path_custom(self, tmp_path):
        """Test custom solver path handling."""
        # Create fake solver
        fake_solver = tmp_path / "fake_solver"
        fake_solver.touch()
        
        solver = SolverInterface(fake_solver)
        assert solver.solver_path == fake_solver
    
    def test_solver_not_found(self):
        """Test error when solver not found."""
        with pytest.raises(FileNotFoundError):
            SolverInterface(Path("/nonexistent/path"))
