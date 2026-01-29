"""
Dataset loading and saving utilities.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def _resolve_dataset_path(dataset_dir: str) -> Path:
    """
    Resolve dataset directory path relative to project root.
    
    Handles both absolute and relative paths, resolving relative paths
    from the project root (not the current working directory).
    """
    path = Path(dataset_dir)
    
    # If already absolute or exists from cwd, use as-is
    if path.is_absolute() or path.exists():
        return path
    
    # Try to find project root by looking for pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            resolved = parent / dataset_dir
            if resolved.exists():
                return resolved
    
    # Fallback to original path
    return path


def load_sample(dataset_dir: str, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a paired (input, target) sample.
    
    Args:
        dataset_dir: Dataset directory path
        sample_id: Sample ID (e.g., "0001")
    
    Returns:
        Tuple of (input_tensor, target_density)
    """
    path = _resolve_dataset_path(dataset_dir)
    X = np.load(path / f"sample_{sample_id}_inputs.npy")
    Y = np.load(path / f"sample_{sample_id}_target.npy")
    return X, Y


def load_dataset_index(dataset_dir: str) -> List[Dict]:
    """Load dataset metadata index."""
    path = _resolve_dataset_path(dataset_dir) / "dataset_index.json"
    with open(path) as f:
        return json.load(f)


def save_sample(
    output_dir: str,
    sample_id: str,
    input_tensor: np.ndarray,
    target_density: np.ndarray
) -> Tuple[Path, Path]:
    """
    Save a paired sample.
    
    Returns:
        Tuple of (input_path, target_path)
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    input_file = path / f"sample_{sample_id}_inputs.npy"
    target_file = path / f"sample_{sample_id}_target.npy"
    
    np.save(input_file, input_tensor)
    np.save(target_file, target_density)
    
    return input_file, target_file


def save_dataset_index(output_dir: str, index: List[Dict]):
    """Save dataset metadata index."""
    path = Path(output_dir) / "dataset_index.json"
    with open(path, 'w') as f:
        json.dump(index, f, indent=2)


def iterate_samples(dataset_dir: str, max_samples: Optional[int] = None):
    """
    Generator to iterate over dataset samples.
    
    Yields:
        Tuple of (sample_id, input_tensor, target_density)
    """
    index = load_dataset_index(dataset_dir)
    
    for i, sample in enumerate(index):
        if max_samples and i >= max_samples:
            break
        
        sample_id = sample['sample_id']
        X, Y = load_sample(dataset_dir, sample_id)
        yield sample_id, X, Y
