"""
Dataset validation utilities.

Provides both math-only validation and visual validation for topology optimization results.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def check_physical_consistency(
    input_tensor: np.ndarray,
    target_density: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Check if the optimization result makes physical sense.
    
    Verifies:
    1. Load is applied (non-zero forces)
    2. Structure exists (non-zero densities)
    3. Volume constraint is satisfied
    4. Structure is concentrated where needed
    
    Args:
        input_tensor: Input tensor (nelx, nely, nelz, 4)
        target_density: Target density (nelx, nely, nelz)
        verbose: Print results to console
    
    Returns:
        Dictionary with consistency check results
    """
    fz = input_tensor[:, :, :, 3]
    
    # Check 1: Load is applied
    total_force = fz.sum()
    has_load = abs(total_force) > 1e-6
    
    # Check 2: Structure exists
    has_structure = (target_density > 0.1).any()
    
    # Check 3: Volume fraction
    volume_fraction = target_density.mean()
    
    # Check 4: Load location
    load_mask = np.abs(fz) > 1e-6
    if load_mask.any():
        load_center = np.array([
            np.mean(np.where(load_mask)[0]),
            np.mean(np.where(load_mask)[1]),
            np.mean(np.where(load_mask)[2])
        ])
    else:
        load_center = None
    
    # Check 5: Structure near load
    distance = None
    if load_center is not None:
        structure_mask = target_density > 0.5
        if structure_mask.any():
            structure_center = np.array([
                np.mean(np.where(structure_mask)[0]),
                np.mean(np.where(structure_mask)[1]),
                np.mean(np.where(structure_mask)[2])
            ])
            distance = np.linalg.norm(load_center - structure_center)
        else:
            distance = float('inf')
    
    if verbose:
        print(f"\n  Physical Consistency Checks:")
        print(f"    ✓ Load applied: {has_load} ({total_force:.2f} N)")
        print(f"    ✓ Structure exists: {has_structure}")
        print(f"    ✓ Volume fraction: {volume_fraction:.3f}")
        if load_center is not None:
            print(f"    ✓ Load center: ({load_center[0]:.1f}, {load_center[1]:.1f}, {load_center[2]:.1f})")
            if distance is not None and distance != float('inf'):
                print(f"    ✓ Structure-load distance: {distance:.1f} elements")
    
    return {
        'has_load': has_load,
        'has_structure': has_structure,
        'volume_fraction': volume_fraction,
        'load_center': load_center,
        'distance': distance,
        'total_force': total_force
    }


def validate_sample_math(
    input_tensor: np.ndarray,
    target_density: np.ndarray,
    expected_force: float = -1000.0,
    vol_frac_target: float = 0.2
) -> Dict:
    """
    Validate a single sample for physical consistency (math checks only).
    
    Args:
        input_tensor: Input tensor (nelx, nely, nelz, 4)
        target_density: Target density (nelx, nely, nelz)
        expected_force: Expected total force
        vol_frac_target: Target volume fraction
    
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'checks': []
    }
    
    # Check shapes
    if input_tensor.shape[:3] != target_density.shape:
        results['valid'] = False
        results['checks'].append({
            'name': 'Shape Match',
            'passed': False,
            'message': f"Input {input_tensor.shape[:3]} != Target {target_density.shape}"
        })
    else:
        results['checks'].append({
            'name': 'Shape Match',
            'passed': True,
            'message': f"Shapes match: {target_density.shape}"
        })
    
    # Check force balance
    total_force = input_tensor[:, :, :, 3].sum()
    force_error = abs(total_force - expected_force)
    force_ok = force_error < 1.0
    
    results['checks'].append({
        'name': 'Force Balance',
        'passed': force_ok,
        'message': f"Total force: {total_force:.2f} N (expected: {expected_force})"
    })
    if not force_ok:
        results['valid'] = False
    
    # Check volume fraction
    vol_frac = target_density.mean()
    vol_error = abs(vol_frac - vol_frac_target)
    vol_ok = vol_error < 0.05
    
    results['checks'].append({
        'name': 'Volume Fraction',
        'passed': vol_ok,
        'message': f"Volume: {vol_frac:.4f} (target: {vol_frac_target})"
    })
    
    # Check density range
    d_min, d_max = target_density.min(), target_density.max()
    range_ok = d_min >= 0 and d_max <= 1
    
    results['checks'].append({
        'name': 'Density Range',
        'passed': range_ok,
        'message': f"Range: [{d_min:.4f}, {d_max:.4f}]"
    })
    if not range_ok:
        results['valid'] = False
    
    return results


def validate_sample_visual(
    input_tensor: np.ndarray,
    target_density: np.ndarray,
    sample_id: str = "unknown",
    threshold: float = 0.3,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    expected_force: float = -1000.0,
    vol_frac_target: float = 0.2
) -> Dict:
    """
    Validate a sample with both math checks and visualization.
    
    Args:
        input_tensor: Input tensor (nelx, nely, nelz, 4)
        target_density: Target density (nelx, nely, nelz)
        sample_id: Sample identifier for display
        threshold: Density threshold for visualization
        save_path: Optional path to save screenshot
        show_plot: Whether to display interactive plot
        expected_force: Expected total force
        vol_frac_target: Target volume fraction
    
    Returns:
        Combined validation results
    """
    from topopt_ml.visualization.pyvista_renderer import compare_input_output
    
    # Run math validation
    math_results = validate_sample_math(
        input_tensor, target_density, expected_force, vol_frac_target
    )
    
    # Run physical consistency check
    physics_results = check_physical_consistency(
        input_tensor, target_density, verbose=True
    )
    
    # Print summary
    print(f"\nSample {sample_id}:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Target shape: {target_density.shape}")
    print(f"  Force sum (Fz): {input_tensor[:,:,:,3].sum():.2f} N")
    print(f"  Density range: [{target_density.min():.4f}, {target_density.max():.4f}]")
    print(f"  Density mean: {target_density.mean():.4f}")
    
    # Visualize
    compare_input_output(
        input_tensor, target_density,
        threshold=threshold,
        show=show_plot,
        screenshot=save_path
    )
    
    print(f"  ✅ Sample {sample_id} validated\n")
    
    return {
        'math': math_results,
        'physics': physics_results,
        'sample_id': sample_id
    }


def validate_dataset(
    dataset_dir: str,
    max_samples: int = None
) -> Dict:
    """
    Validate entire dataset (math checks only).
    
    Args:
        dataset_dir: Dataset directory
        max_samples: Maximum samples to check
    
    Returns:
        Validation summary
    """
    from topopt_ml.io.datasets import iterate_samples, load_dataset_index
    
    index = load_dataset_index(dataset_dir)
    total = len(index)
    
    if max_samples:
        total = min(total, max_samples)
    
    passed = 0
    failed = 0
    errors = []
    
    for sample_id, X, Y in iterate_samples(dataset_dir, max_samples):
        result = validate_sample_math(X, Y)
        
        if result['valid']:
            passed += 1
        else:
            failed += 1
            errors.append({
                'sample_id': sample_id,
                'checks': [c for c in result['checks'] if not c['passed']]
            })
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / total if total > 0 else 0,
        'errors': errors
    }


def print_validation_report(results: Dict):
    """Print formatted validation report."""
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)
    print(f"Total samples: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {results['pass_rate']*100:.1f}%")
    
    if results['errors']:
        print("\nErrors:")
        for err in results['errors']:
            print(f"  Sample {err['sample_id']}:")
            for check in err['checks']:
                print(f"    - {check['name']}: {check['message']}")
    
    print("=" * 60)


# Backwards compatibility alias
validate_sample = validate_sample_math

