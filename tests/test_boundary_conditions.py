"""Tests for BCFactory."""

import pytest
import tempfile
import os
import numpy as np

from topopt_ml.core.grid import GridCalculator
from topopt_ml.core.boundary_conditions import BCFactory, BC_TYPES, get_bc_types


class TestBCFactory:
    """Test suite for BCFactory."""
    
    @pytest.fixture
    def grid_calc(self):
        """Create grid calculator fixture."""
        return GridCalculator(64, 32, 32)
    
    @pytest.fixture
    def bc_factory(self, grid_calc):
        """Create BC factory fixture."""
        return BCFactory(grid_calc)
    
    def test_bc_types_list(self):
        """Test that BC_TYPES contains expected types."""
        assert 'cantilever_left' in BC_TYPES
        assert 'cantilever_right' in BC_TYPES
        assert 'simply_supported' in BC_TYPES
        assert 'corner_fixed' in BC_TYPES
        assert 'bridge' in BC_TYPES
    
    def test_get_bc_types(self):
        """Test get_bc_types returns a copy of BC_TYPES."""
        types = get_bc_types()
        assert types == BC_TYPES
        # Ensure it's a copy
        types.append('test')
        assert 'test' not in BC_TYPES
    
    def test_cantilever_left_generation(self, bc_factory, grid_calc):
        """Test cantilever left BC generation."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_bc('cantilever_left', bc_file)
            
            assert filename == bc_file
            assert meta['bc_type'] == 'cantilever_left'
            assert meta['num_fixed_dofs'] > 0
            assert 'num_fixed_nodes' in meta
            assert 'fixed_node_locations' in meta
            
            # Read and verify the binary file
            dofs = np.fromfile(bc_file, dtype=np.int32)
            assert len(dofs) == meta['num_fixed_dofs']
            assert np.all(dofs >= 0)
            assert np.all(dofs < grid_calc.ndof)
        finally:
            os.remove(bc_file)
    
    def test_cantilever_right_generation(self, bc_factory, grid_calc):
        """Test cantilever right BC generation."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_bc('cantilever_right', bc_file)
            
            assert meta['bc_type'] == 'cantilever_right'
            assert meta['num_fixed_dofs'] > 0
            
            dofs = np.fromfile(bc_file, dtype=np.int32)
            assert len(dofs) == meta['num_fixed_dofs']
        finally:
            os.remove(bc_file)
    
    def test_simply_supported_generation(self, bc_factory, grid_calc):
        """Test simply supported BC generation."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_bc('simply_supported', bc_file)
            
            assert meta['bc_type'] == 'simply_supported'
            assert meta['num_fixed_dofs'] > 0
            
            dofs = np.fromfile(bc_file, dtype=np.int32)
            assert len(dofs) == meta['num_fixed_dofs']
        finally:
            os.remove(bc_file)
    
    def test_corner_fixed_generation(self, bc_factory, grid_calc):
        """Test corner fixed BC generation."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_bc('corner_fixed', bc_file)
            
            assert meta['bc_type'] == 'corner_fixed'
            # 4 corners × 3 DOFs each = 12 DOFs
            assert meta['num_fixed_dofs'] == 12
            
            dofs = np.fromfile(bc_file, dtype=np.int32)
            assert len(dofs) == 12
        finally:
            os.remove(bc_file)
    
    def test_bridge_generation(self, bc_factory, grid_calc):
        """Test bridge BC generation."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_bc('bridge', bc_file)
            
            assert meta['bc_type'] == 'bridge'
            assert meta['num_fixed_dofs'] > 0
            
            dofs = np.fromfile(bc_file, dtype=np.int32)
            assert len(dofs) == meta['num_fixed_dofs']
        finally:
            os.remove(bc_file)
    
    def test_invalid_bc_type(self, bc_factory):
        """Test that invalid BC type raises ValueError."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            with pytest.raises(ValueError, match="Unknown BC type"):
                bc_factory.generate_bc('invalid_type', bc_file)
        finally:
            if os.path.exists(bc_file):
                os.remove(bc_file)
    
    def test_random_bc_generation(self, bc_factory):
        """Test random BC generation without weights."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_random_bc(
                filename=bc_file,
                random_seed=42
            )
            
            assert meta['bc_type'] in BC_TYPES
            assert meta['num_fixed_dofs'] > 0
        finally:
            os.remove(bc_file)
    
    def test_random_bc_with_weights(self, bc_factory):
        """Test random BC generation with weights."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            weights = {
                'cantilever_left': 0.0,  # Never select this
                'cantilever_right': 1.0,  # Always select this
                'simply_supported': 0.0,
                'corner_fixed': 0.0,
                'bridge': 0.0
            }
            
            filename, meta = bc_factory.generate_random_bc(
                filename=bc_file,
                weights=weights,
                random_seed=42
            )
            
            assert meta['bc_type'] == 'cantilever_right'
        finally:
            os.remove(bc_file)
    
    def test_random_bc_reproducibility(self, bc_factory):
        """Test that same seed produces same results."""
        fd1, f1 = tempfile.mkstemp(suffix=".bin")
        fd2, f2 = tempfile.mkstemp(suffix=".bin")
        os.close(fd1)
        os.close(fd2)
        
        try:
            _, m1 = bc_factory.generate_random_bc(filename=f1, random_seed=42)
            _, m2 = bc_factory.generate_random_bc(filename=f2, random_seed=42)
            
            assert m1['bc_type'] == m2['bc_type']
            assert m1['num_fixed_dofs'] == m2['num_fixed_dofs']
            
            dofs1 = np.fromfile(f1, dtype=np.int32)
            dofs2 = np.fromfile(f2, dtype=np.int32)
            np.testing.assert_array_equal(dofs1, dofs2)
        finally:
            os.remove(f1)
            os.remove(f2)
    
    def test_dof_indices_are_unique(self, bc_factory):
        """Test that generated DOF indices are unique."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            for bc_type in BC_TYPES:
                bc_factory.generate_bc(bc_type, bc_file)
                dofs = np.fromfile(bc_file, dtype=np.int32)
                assert len(dofs) == len(np.unique(dofs)), \
                    f"Duplicate DOFs found for BC type: {bc_type}"
        finally:
            os.remove(bc_file)
    
    def test_specific_bc_type_in_random(self, bc_factory):
        """Test that specifying bc_type in generate_random_bc works."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, meta = bc_factory.generate_random_bc(
                bc_type='corner_fixed',
                filename=bc_file,
                random_seed=42
            )
            
            assert meta['bc_type'] == 'corner_fixed'
        finally:
            os.remove(bc_file)


class TestBCCrossValidation:
    """Cross-validation tests for BC consistency."""
    
    @pytest.fixture
    def grid_calc(self):
        """Create grid calculator fixture."""
        return GridCalculator(32, 16, 16)
    
    @pytest.fixture
    def bc_factory(self, grid_calc):
        """Create BC factory fixture."""
        return BCFactory(grid_calc)
    
    def test_cantilever_left_matches_hardcoded(self, bc_factory, grid_calc):
        """
        Test that cantilever_left BC matches the hardcoded C implementation.
        
        The C code fixes nodes at i=1 for bottom quarter and top quarter.
        """
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, meta = bc_factory.generate_bc('cantilever_left', bc_file)
            dofs = np.fromfile(bc_file, dtype=np.int32)
            
            # Expected: 3 * nz_nodes * 2 * nodelimit DOFs
            nely = grid_calc.nely
            nelz = grid_calc.nelz
            nz_nodes = nelz + 1
            nodelimit = (nely // 4) + 1
            expected_num_dofs = 3 * nz_nodes * 2 * nodelimit
            
            assert len(dofs) == expected_num_dofs, \
                f"Expected {expected_num_dofs} DOFs, got {len(dofs)}"
            
            # Verify all DOFs are at i=1 (left face)
            for dof in dofs:
                node_idx = dof // 3
                i = node_idx // (grid_calc.wrapy * grid_calc.wrapz)
                assert i == 1, f"DOF {dof} not on left face (i={i})"
        finally:
            os.remove(bc_file)
    
    def test_all_bc_types_generate_valid_dofs(self, bc_factory, grid_calc):
        """Test all BC types generate valid DOF indices."""
        fd, bc_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            for bc_type in BC_TYPES:
                _, meta = bc_factory.generate_bc(bc_type, bc_file)
                dofs = np.fromfile(bc_file, dtype=np.int32)
                
                # All DOFs should be valid
                assert np.all(dofs >= 0), f"{bc_type}: Negative DOF indices found"
                assert np.all(dofs < grid_calc.ndof), f"{bc_type}: DOF indices exceed ndof"
                
                # Should have at least 3 DOFs (one node)
                assert len(dofs) >= 3, f"{bc_type}: Too few DOFs generated"
        finally:
            os.remove(bc_file)
