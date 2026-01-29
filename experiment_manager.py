#!/usr/bin/env python3
"""
experiment_manager.py - HPC Experiment Manager for TopOpt

Centralized experiment management system with:
- YAML configuration
- Versioned experiment tracking
- Accurate memory layout matching C++ solver
- Randomized load generation
- Automated solver execution
- ML tensor export
"""

import yaml
import numpy as np
import subprocess
import tempfile
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import shutil


class ExperimentConfig:
    """Load and validate experiment configuration from YAML."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        grid = self.config['grid']
        solver = self.config['solver']
        
        # Check grid divisibility for multigrid
        size_incr = 2 ** (solver['nl'] - 1)
        for dim in ['nx', 'ny', 'nz']:
            if grid[dim] % size_incr != 0:
                raise ValueError(
                    f"Grid {dim}={grid[dim]} must be divisible by "
                    f"2^(nl-1) = {size_incr} for nl={solver['nl']}"
                )
        
        logging.info("✓ Configuration validated")
    
    def __getitem__(self, key):
        return self.config[key]
    
    def get(self, key, default=None):
        return self.config.get(key, default)


class GridCalculator:
    """Calculate grid dimensions and indexing matching C++ memory layout."""
    
    def __init__(self, nx: int, ny: int, nz: int, 
                 stencil_x: int = 1, stencil_y: int = 8, stencil_z: int = 1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.stencil_x = stencil_x
        self.stencil_y = stencil_y
        self.stencil_z = stencil_z
        
        # Calculate wrap dimensions (C++ grid_utilities.c logic)
        self.wrapx, self.wrapy, self.wrapz = self._compute_wrap_dimensions()
        self.ndof = self.wrapx * self.wrapy * self.wrapz * 3
        
        logging.debug(f"Grid: {nx}×{ny}×{nz}")
        logging.debug(f"Wrap: {self.wrapx}×{self.wrapy}×{self.wrapz}")
        logging.debug(f"NDOF: {self.ndof}")
    
    def _compute_wrap_dimensions(self) -> Tuple[int, int, int]:
        """
        Compute wrapped dimensions matching C++ initializeGridContext.
        
        Returns node dimensions with padding and halo cells.
        """
        # Padding for SIMD alignment
        padding_x = (self.stencil_x - ((self.nx + 1) % self.stencil_x)) % self.stencil_x
        padding_y = (self.stencil_y - ((self.ny + 1) % self.stencil_y)) % self.stencil_y
        padding_z = (self.stencil_z - ((self.nz + 1) % self.stencil_z)) % self.stencil_z
        
        # Add padding and 3 halo cells
        wrapx = self.nx + padding_x + 3
        wrapy = self.ny + padding_y + 3
        wrapz = self.nz + padding_z + 3
        
        return wrapx, wrapy, wrapz
    
    def node_index_to_flat(self, i: int, j: int, k: int) -> int:
        """
        Convert 3D node index to flat index using C++ scheme.
        
        Args:
            i, j, k: Node indices (0-based)
        
        Returns:
            Flat index in C++ memory layout
        """
        return i * self.wrapy * self.wrapz + self.wrapy * k + j


class LoadFactory:
    """Generate randomized load cases."""
    
    def __init__(self, grid_calc: GridCalculator, config: Dict):
        self.grid = grid_calc
        self.config = config
    
    def generate_circular_patch(self, seed: int) -> Tuple[np.ndarray, Dict]:
        """
        Generate random circular load patch on top surface.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            force_vector: Binary force vector (ndof doubles)
            metadata: Load metadata
        """
        np.random.seed(seed)
        
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        
        # Random center (20-80% of domain)
        cx = np.random.randint(
            int(nx * self.config['center_min_frac']),
            int(nx * self.config['center_max_frac'])
        )
        cy = np.random.randint(
            int(ny * self.config['center_min_frac']),
            int(ny * self.config['center_max_frac'])
        )
        
        # Random radius
        radius = np.random.uniform(
            self.config['radius_min'],
            self.config['radius_max']
        )
        
        # Find nodes in circular patch on top surface
        load_nodes = []
        for ix in range(nx + 1):
            for iy in range(ny + 1):
                dist = np.sqrt((ix - cx)**2 + (iy - cy)**2)
                if dist <= radius:
                    load_nodes.append((ix, iy, nz))  # Top surface
        
        # Create force vector
        force_vector = np.zeros(self.grid.ndof, dtype=np.float64)
        
        if len(load_nodes) > 0:
            nodal_force = self.config['total_force'] / len(load_nodes)
            
            for ix, iy, iz in load_nodes:
                # Z-direction DOF (index 2 of 3 DOFs per node)
                flat_idx = self.grid.node_index_to_flat(ix, iy, iz)
                z_dof_idx = flat_idx * 3 + 2
                force_vector[z_dof_idx] = nodal_force
        
        metadata = {
            'load_center': [int(cx), int(cy)],
            'load_radius': float(radius),
            'num_load_nodes': len(load_nodes),
            'nodal_force': float(nodal_force) if load_nodes else 0.0,
            'total_force': float(self.config['total_force']),
            'random_seed': seed
        }
        
        return force_vector, metadata


class ExperimentManager:
    """Manage HPC topology optimization experiments."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = ExperimentConfig(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize grid calculator
        grid_cfg = self.config['grid']
        hpc_cfg = self.config['hpc']
        self.grid_calc = GridCalculator(
            grid_cfg['nx'], grid_cfg['ny'], grid_cfg['nz'],
            hpc_cfg['stencil_size_x'],
            hpc_cfg['stencil_size_y'],
            hpc_cfg['stencil_size_z']
        )
        
        # Initialize load factory
        self.load_factory = LoadFactory(self.grid_calc, self.config['load'])
        
        # Create experiment directory
        self.exp_dir = self._create_experiment_dir()
        
        logging.info(f"Experiment directory: {self.exp_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config['output'].get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _create_experiment_dir(self) -> Path:
        """Create versioned experiment directory."""
        output_cfg = self.config['output']
        grid_cfg = self.config['grid']
        
        base_dir = Path(output_cfg['base_dir'])
        
        # Create experiment name
        if output_cfg['create_timestamp']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"EXP_{timestamp}_{grid_cfg['nx']}x{grid_cfg['ny']}x{grid_cfg['nz']}"
        else:
            exp_name = f"EXP_{grid_cfg['nx']}x{grid_cfg['ny']}x{grid_cfg['nz']}"
        
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration copy
        if output_cfg['save_config']:
            shutil.copy("config.yaml", exp_dir / "config.yaml")
        
        return exp_dir
    
    def run_single_sample(self, sample_idx: int) -> Dict:
        """
        Run single optimization sample.
        
        Args:
            sample_idx: Sample index
        
        Returns:
            Dictionary with results and metadata
        """
        ml_cfg = self.config['ml']
        seed = ml_cfg['base_seed'] + sample_idx
        
        logging.info(f"Sample {sample_idx + 1}/{ml_cfg['num_samples']}")
        
        # Generate load
        force_vector, load_meta = self.load_factory.generate_circular_patch(seed)
        
        logging.info(f"  Load: center={load_meta['load_center']}, "
                    f"radius={load_meta['load_radius']:.2f}, "
                    f"nodes={load_meta['num_load_nodes']}")
        
        # Save force vector to temporary file
        fd, load_file = tempfile.mkstemp(suffix=".bin", prefix="topopt_load_")
        os.close(fd)
        
        try:
            force_vector.tofile(load_file)
            
            # Run solver
            result = self._run_solver(load_file, sample_idx)
            
            # Save outputs
            if result['success']:
                self._save_sample(sample_idx, force_vector, result, load_meta)
            
            return result
            
        finally:
            if os.path.exists(load_file):
                os.remove(load_file)
    
    def _run_solver(self, load_file: str, sample_idx: int) -> Dict:
        """Execute C++ solver via subprocess."""
        grid_cfg = self.config['grid']
        solver_cfg = self.config['solver']
        
        # Calculate coarse grid dimensions
        nl = solver_cfg['nl']
        size_incr = 2 ** (nl - 1)
        nx_coarse = grid_cfg['nx'] // size_incr
        ny_coarse = grid_cfg['ny'] // size_incr
        nz_coarse = grid_cfg['nz'] // size_incr
        
        # Solver executable
        solver_path = Path(__file__).parent / "TopOpt-in-OpenMP" / "top3d"
        
        if not solver_path.exists():
            raise FileNotFoundError(f"Solver not found: {solver_path}")
        
        # Build command
        cmd = [
            str(solver_path),
            "-x", str(nx_coarse),
            "-y", str(ny_coarse),
            "-z", str(nz_coarse),
            "-v", str(grid_cfg['vol_fraction']),
            "-r", str(solver_cfg['rmin']),
            "-i", str(solver_cfg['max_iter']),
            "-l", str(nl),
            "-f", load_file
        ]
        
        logging.debug(f"  Command: {' '.join(cmd)}")
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(solver_path.parent)
        )
        
        if result.returncode != 0:
            logging.error(f"  Solver failed: {result.stderr}")
            return {'success': False, 'error': result.stderr}
        
        # Parse timing
        timing_line = [l for l in result.stdout.split('\n') if "End time:" in l]
        solve_time = float(timing_line[0].split(":")[1].strip()) if timing_line else None
        
        logging.info(f"  Solver completed in {solve_time:.2f}s")
        
        return {
            'success': True,
            'solve_time': solve_time,
            'stdout': result.stdout
        }
    
    def _save_sample(self, sample_idx: int, force_vector: np.ndarray, 
                     result: Dict, load_meta: Dict):
        """Save sample inputs and outputs."""
        ml_cfg = self.config['ml']
        grid_cfg = self.config['grid']
        
        sample_id = f"{sample_idx + 1:04d}"
        
        # Save input force vector as tensor
        if ml_cfg['save_inputs']:
            # Reshape to 4-channel tensor (solid mask + forces)
            nx, ny, nz = grid_cfg['nx'], grid_cfg['ny'], grid_cfg['nz']
            input_tensor = self._create_input_tensor(force_vector, nx, ny, nz)
            
            input_file = self.exp_dir / f"sample_{sample_id}_inputs.npy"
            np.save(input_file, input_tensor)
            logging.debug(f"  Saved input: {input_file.name}")
        
        # Save output density field
        if ml_cfg['save_targets']:
            density_file = Path("TopOpt-in-OpenMP") / "density.bin"
            
            if density_file.exists():
                from run_factory import read_density_field
                density = read_density_field(
                    grid_cfg['nx'], grid_cfg['ny'], grid_cfg['nz'],
                    str(density_file)
                )
                
                target_file = self.exp_dir / f"sample_{sample_id}_target.npy"
                np.save(target_file, density)
                logging.debug(f"  Saved target: {target_file.name}")
                
                # Clean up
                density_file.unlink()
    
    def _create_input_tensor(self, force_vector: np.ndarray, 
                            nx: int, ny: int, nz: int) -> np.ndarray:
        """
        Create 4-channel input tensor from force vector.
        
        Channels:
          0: Solid domain mask (all 1s)
          1: Force X (all 0s for vertical load)
          2: Force Y (all 0s for vertical load)
          3: Force Z (mapped from nodes to elements)
        """
        input_tensor = np.zeros((nx, ny, nz, 4), dtype=np.float32)
        
        # Channel 0: Solid mask
        input_tensor[:, :, :, 0] = 1.0
        
        # Channel 3: Force Z (map from nodes to elements)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Element receives force from its 4 corner nodes (weighted)
                    for di in [0, 1]:
                        for dj in [0, 1]:
                            for dk in [0, 1]:
                                node_flat = self.grid_calc.node_index_to_flat(
                                    ix + di, iy + dj, iz + dk
                                )
                                z_dof = node_flat * 3 + 2
                                if z_dof < len(force_vector):
                                    input_tensor[ix, iy, iz, 3] += force_vector[z_dof] / 8.0
        
        return input_tensor
    
    def run_batch(self):
        """Run batch of experiments."""
        ml_cfg = self.config['ml']
        num_samples = ml_cfg['num_samples']
        
        logging.info("=" * 70)
        logging.info(f"Starting batch generation: {num_samples} samples")
        logging.info("=" * 70)
        
        # Collect metadata
        dataset_index = []
        successful = 0
        
        for i in range(num_samples):
            try:
                result = self.run_single_sample(i)
                
                if result['success']:
                    successful += 1
                    # Add to index (you can extend this)
                    dataset_index.append({
                        'sample_id': f"{i+1:04d}",
                        'solve_time': result.get('solve_time')
                    })
            
            except Exception as e:
                logging.error(f"Sample {i+1} failed: {e}")
        
        # Save dataset index
        index_file = self.exp_dir / "dataset_index.json"
        with open(index_file, 'w') as f:
            json.dump(dataset_index, f, indent=2)
        
        logging.info("=" * 70)
        logging.info(f"Batch complete: {successful}/{num_samples} successful")
        logging.info(f"Results saved to: {self.exp_dir}")
        logging.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HPC Experiment Manager for TopOpt")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--samples", type=int, help="Override number of samples")
    
    args = parser.parse_args()
    
    # Create manager
    manager = ExperimentManager(args.config)
    
    # Override samples if specified
    if args.samples:
        manager.config.config['ml']['num_samples'] = args.samples
    
    # Run batch
    manager.run_batch()
