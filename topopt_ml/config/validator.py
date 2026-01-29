"""
Configuration validation utilities.
"""

from typing import Dict, Any, List


class ValidationError(Exception):
    """Configuration validation error."""
    pass


def validate_grid_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate grid configuration.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required = ['nx', 'ny', 'nz']
    for key in required:
        if key not in config:
            errors.append(f"Missing required grid parameter: {key}")
    
    for key in ['nx', 'ny', 'nz']:
        if key in config:
            val = config[key]
            if not isinstance(val, int) or val < 8:
                errors.append(f"Grid {key} must be integer >= 8, got {val}")
    
    if 'vol_fraction' in config:
        vf = config['vol_fraction']
        if not 0 < vf < 1:
            errors.append(f"vol_fraction must be in (0, 1), got {vf}")
    
    return errors


def validate_solver_config(config: Dict[str, Any]) -> List[str]:
    """Validate solver configuration."""
    errors = []
    
    if 'nl' in config:
        nl = config['nl']
        if not isinstance(nl, int) or nl < 1 or nl > 5:
            errors.append(f"nl must be integer in [1, 5], got {nl}")
    
    if 'rmin' in config:
        rmin = config['rmin']
        if rmin <= 0:
            errors.append(f"rmin must be positive, got {rmin}")
    
    if 'max_iter' in config:
        iters = config['max_iter']
        if not isinstance(iters, int) or iters < 1:
            errors.append(f"max_iter must be positive integer, got {iters}")
    
    return errors


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate full configuration.
    
    Returns:
        List of all validation errors
    """
    errors = []
    
    if 'grid' in config:
        errors.extend(validate_grid_config(config['grid']))
    
    if 'solver' in config:
        errors.extend(validate_solver_config(config['solver']))
    
    return errors
