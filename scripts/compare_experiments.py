#!/usr/bin/env python3
"""
Compare and summarize results from different experiments

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --details
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_experiment_info(exp_dir: Path):
    """Load experiment configuration and metadata."""
    config_path = exp_dir / "config.yaml"
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Count samples
    sample_files = list(exp_dir.glob("sample_*_inputs.npy"))
    num_samples = len(sample_files)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in exp_dir.glob("*"))
    
    return {
        "name": exp_dir.name,
        "config": config,
        "num_samples": num_samples,
        "total_size_mb": total_size / (1024 * 1024),
        "path": exp_dir
    }


def print_summary(experiments, detailed=False):
    """Print a formatted summary of all experiments."""
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Header
    print(f"{'Experiment':<30} {'Grid':<15} {'Vol%':<6} {'Samples':<8} {'Size (MB)':<10}")
    print("-" * 80)
    
    for exp in experiments:
        grid = exp['config']['grid']
        grid_str = f"{grid['nx']}×{grid['ny']}×{grid['nz']}"
        vol_frac = grid['vol_fraction'] * 100
        
        print(f"{exp['name']:<30} {grid_str:<15} {vol_frac:<6.1f} {exp['num_samples']:<8} {exp['total_size_mb']:<10.2f}")
    
    print("=" * 80)
    
    if detailed:
        print("\nDETAILED CONFIGURATION:")
        print("-" * 80)
        
        for exp in experiments:
            print(f"\n{exp['name']}:")
            print(f"  Path: {exp['path']}")
            
            cfg = exp['config']
            print(f"  Grid: {cfg['grid']['nx']}×{cfg['grid']['ny']}×{cfg['grid']['nz']}")
            print(f"  Volume Fraction: {cfg['grid']['vol_fraction']}")
            print(f"  Max Iterations: {cfg['solver']['max_iter']}")
            print(f"  Filter Radius: {cfg['solver']['rmin']}")
            print(f"  Multigrid Levels: {cfg['solver']['nl']}")
            
            if 'load' in cfg:
                print(f"  Load Force: {cfg['load']['total_force']}")
                print(f"  Load Radius: {cfg['load']['radius_min']}-{cfg['load']['radius_max']}")
            
            print(f"  Samples Generated: {exp['num_samples']}")
            print(f"  Storage Size: {exp['total_size_mb']:.2f} MB")
    
    print()


def compare_load_diversity(experiments):
    """Compare load pattern diversity across experiments."""
    print("\n" + "=" * 80)
    print("LOAD PATTERN DIVERSITY")
    print("=" * 80)
    
    for exp in experiments:
        cfg = exp['config']
        if 'load' not in cfg:
            continue
        
        load = cfg['load']
        grid = cfg['grid']
        
        # Calculate actual range
        center_range_x = (load['center_max_frac'] - load['center_min_frac']) * grid['nx']
        radius_range = load['radius_max'] - load['radius_min']
        
        print(f"\n{exp['name']}:")
        print(f"  Center Range: {load['center_min_frac']:.1f}-{load['center_max_frac']:.1f} "
              f"({center_range_x:.1f} elements)")
        print(f"  Radius Range: {load['radius_min']:.1f}-{load['radius_max']:.1f} "
              f"({radius_range:.1f} element variation)")
        print(f"  Force: {load['total_force']} N")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiments from different configurations"
    )
    parser.add_argument(
        "--details", "-d",
        action="store_true",
        help="Show detailed configuration for each experiment"
    )
    parser.add_argument(
        "--loads",
        action="store_true",
        help="Show load pattern diversity comparison"
    )
    
    args = parser.parse_args()
    
    # Find all experiments
    exp_base_dir = Path(__file__).parent.parent / "data" / "experiments"
    
    if not exp_base_dir.exists():
        print(f"❌ No experiments directory found: {exp_base_dir}")
        return
    
    exp_dirs = [d for d in exp_base_dir.iterdir() if d.is_dir() and d.name.startswith("EXP_")]
    
    # Load experiment info
    experiments = []
    for exp_dir in sorted(exp_dirs, key=lambda x: x.name):
        info = load_experiment_info(exp_dir)
        if info:
            experiments.append(info)
    
    # Print summaries
    print_summary(experiments, detailed=args.details)
    
    if args.loads:
        compare_load_diversity(experiments)


if __name__ == "__main__":
    main()
