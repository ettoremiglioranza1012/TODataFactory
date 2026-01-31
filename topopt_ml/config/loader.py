"""
YAML configuration loader.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage YAML configuration files."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load()
        self._validate()
    
    def _load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _validate(self):
        """Validate configuration parameters."""
        grid = self.config.get('grid', {})
        solver = self.config.get('solver', {})
        
        # Check grid divisibility for multigrid
        nl = solver.get('nl', 4)
        size_incr = 2 ** (nl - 1)
        
        for dim in ['nx', 'ny', 'nz']:
            val = grid.get(dim, 0)
            if val > 0 and val % size_incr != 0:
                raise ValueError(
                    f"Grid {dim}={val} must be divisible by "
                    f"2^(nl-1) = {size_incr} for nl={nl}"
                )
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    @property
    def grid(self) -> Dict:
        return self.config.get('grid', {})
    
    @property
    def solver(self) -> Dict:
        return self.config.get('solver', {})
    
    @property
    def hpc(self) -> Dict:
        return self.config.get('hpc', {})
    
    @property
    def ml(self) -> Dict:
        return self.config.get('ml', {})
    
    @property
    def load(self) -> Dict:
        return self.config.get('load', {})
    
    @property
    def boundary_conditions(self) -> Dict:
        """Get boundary conditions configuration."""
        return self.config.get('boundary_conditions', {
            'bc_type': 'cantilever_left',  # Default fallback
            'random_weights': None
        })
    
    @property
    def output(self) -> Dict:
        return self.config.get('output', {})


def load_config(path: str = "config/default.yaml") -> ConfigLoader:
    """Convenience function to load configuration."""
    return ConfigLoader(path)

