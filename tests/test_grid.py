"""Tests for GridCalculator."""

import pytest
from topopt_ml.core.grid import GridCalculator, compute_wrap_dimensions, compute_ndof


class TestGridCalculator:
    """Test suite for GridCalculator."""
    
    def test_basic_creation(self):
        """Test basic grid calculator creation."""
        calc = GridCalculator(64, 32, 32)
        
        assert calc.nelx == 64
        assert calc.nely == 32
        assert calc.nelz == 32
    
    def test_wrap_dimensions(self):
        """Test wrapped dimension calculation."""
        calc = GridCalculator(64, 32, 32)
        
        # Wrap = elements + padding + 3 halo cells
        # Formula: (stencil - ((nel + 1) % stencil)) % stencil + nel + 3
        # x: 64, stencil=1 → padding=0, wrap = 64 + 0 + 3 = 67
        # y: 32, stencil=8 → (8 - 33%8) % 8 = 7, wrap = 32 + 7 + 3 = 42
        # z: 32, stencil=1 → padding=0, wrap = 32 + 0 + 3 = 35
        
        assert calc.wrapx == 67
        assert calc.wrapy == 42
        assert calc.wrapz == 35
    
    def test_ndof(self):
        """Test NDOF calculation."""
        calc = GridCalculator(64, 32, 32)
        
        expected = 3 * calc.wrapx * calc.wrapy * calc.wrapz
        assert calc.ndof == expected
    
    def test_node_index_to_flat(self):
        """Test node index conversion."""
        calc = GridCalculator(64, 32, 32)
        
        # First node
        assert calc.node_index_to_flat(0, 0, 0) == 0
        
        # Increment j by 1
        assert calc.node_index_to_flat(0, 1, 0) == 1
        
        # Increment k by 1
        assert calc.node_index_to_flat(0, 0, 1) == calc.wrapy
    
    def test_backward_compat_functions(self):
        """Test backward compatibility functions."""
        wrapx, wrapy, wrapz = compute_wrap_dimensions(64, 32, 32)
        ndof = compute_ndof(64, 32, 32)
        
        calc = GridCalculator(64, 32, 32)
        
        assert wrapx == calc.wrapx
        assert wrapy == calc.wrapy
        assert wrapz == calc.wrapz
        assert ndof == calc.ndof


class TestGridDivisibility:
    """Test grid size divisibility requirements."""
    
    @pytest.mark.parametrize("nx,ny,nz", [
        (64, 32, 32),
        (128, 64, 64),
        (48, 24, 24),
    ])
    def test_valid_sizes(self, nx, ny, nz):
        """Test that valid sizes work."""
        calc = GridCalculator(nx, ny, nz)
        assert calc.ndof > 0
