"""
Dataset validation utilities.
"""

import numpy as np
from typing import Dict, List, Tuple


def validate_sample(
    input_tensor: np.ndarray,
    target_density: np.ndarray,
    expected_force: float = -1000.0,
    vol_frac_target: float = 0.2
) -> Dict:
    """
    Validate a single sample for physical consistency.
    
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


def validate_dataset(
    dataset_dir: str,
    max_samples: int = None
) -> Dict:
    """
    Validate entire dataset.
    
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
        result = validate_sample(X, Y)
        
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
