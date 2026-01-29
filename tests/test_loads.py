"""Tests for LoadFactory."""

import pytest
import tempfile
import os
import numpy as np

from topopt_ml.core.grid import GridCalculator
from topopt_ml.core.loads import LoadFactory


class TestLoadFactory:
    """Test suite for LoadFactory."""
    
    @pytest.fixture
    def grid_calc(self):
        """Create grid calculator fixture."""
        return GridCalculator(64, 32, 32)
    
    @pytest.fixture
    def load_factory(self, grid_calc):
        """Create load factory fixture."""
        return LoadFactory(grid_calc)
    
    def test_load_generation(self, load_factory):
        """Test basic load generation."""
        fd, load_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            filename, tensor, meta = load_factory.generate_random_load(
                load_file,
                total_force=-1000.0,
                random_seed=42
            )
            
            assert filename == load_file
            assert tensor.shape == (64, 32, 32, 4)
            assert meta['total_force'] == -1000.0
            assert 'load_center' in meta
            assert 'load_radius' in meta
        finally:
            os.remove(load_file)
    
    def test_force_balance(self, load_factory):
        """Test that total force matches expected value."""
        fd, load_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, tensor, meta = load_factory.generate_random_load(
                load_file,
                total_force=-1000.0,
                random_seed=42
            )
            
            total_fz = tensor[:, :, :, 3].sum()
            assert abs(total_fz - (-1000.0)) < 1.0
        finally:
            os.remove(load_file)
    
    def test_solid_mask(self, load_factory):
        """Test that solid mask channel is all ones."""
        fd, load_file = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        
        try:
            _, tensor, _ = load_factory.generate_random_load(
                load_file,
                random_seed=42
            )
            
            solid = tensor[:, :, :, 0]
            assert solid.sum() == 64 * 32 * 32
        finally:
            os.remove(load_file)
    
    def test_reproducibility(self, load_factory):
        """Test that same seed produces same results."""
        fd1, f1 = tempfile.mkstemp(suffix=".bin")
        fd2, f2 = tempfile.mkstemp(suffix=".bin")
        os.close(fd1)
        os.close(fd2)
        
        try:
            _, t1, m1 = load_factory.generate_random_load(f1, random_seed=42)
            _, t2, m2 = load_factory.generate_random_load(f2, random_seed=42)
            
            assert m1['load_center'] == m2['load_center']
            assert m1['load_radius'] == m2['load_radius']
            np.testing.assert_array_equal(t1, t2)
        finally:
            os.remove(f1)
            os.remove(f2)
