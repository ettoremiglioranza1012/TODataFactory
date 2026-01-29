"""Tests for tensor utilities."""

import pytest
import numpy as np

from topopt_ml.core.tensors import (
    normalize_density,
    threshold_density,
    compute_volume_fraction,
    get_tensor_stats,
)


class TestTensorUtilities:
    """Test suite for tensor utilities."""
    
    def test_normalize_density(self):
        """Test density normalization."""
        density = np.array([0.0, 0.5, 1.0, 1.5, -0.5])
        normalized = normalize_density(density)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        np.testing.assert_array_equal(normalized, [0.0, 0.5, 1.0, 1.0, 0.0])
    
    def test_threshold_density(self):
        """Test density thresholding."""
        density = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        thresholded = threshold_density(density, threshold=0.5)
        np.testing.assert_array_equal(thresholded, [0.0, 0.0, 1.0, 1.0, 1.0])
    
    def test_compute_volume_fraction(self):
        """Test volume fraction computation."""
        density = np.array([0.0, 0.2, 0.3, 0.5])
        vf = compute_volume_fraction(density)
        
        assert vf == pytest.approx(0.25)
    
    def test_get_tensor_stats(self):
        """Test tensor statistics."""
        tensor = np.random.randn(10, 10, 10).astype(np.float32)
        stats = get_tensor_stats(tensor)
        
        assert stats['shape'] == (10, 10, 10)
        assert stats['dtype'] == 'float32'
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
