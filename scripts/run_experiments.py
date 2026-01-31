#!/usr/bin/env python3
"""
Experiment Runner - Run different configurations and compare results

Usage:
    python scripts/run_experiments.py --all                    # Run all configs
    python scripts/run_experiments.py --config quick_test      # Run specific config
    python scripts/run_experiments.py --list                   # List available configs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from topopt_ml.experiments import ExperimentManager


def list_configs():
    """List all available configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    configs = sorted(config_dir.glob("*.yaml"))
    
    print("=" * 60)
    print("Available Configurations:")
    print("=" * 60)
    for cfg in configs:
        if cfg.stem != "schemas":  # Skip schemas directory
            print(f"  • {cfg.stem:20s} ({cfg.name})")
    print()


def run_config(config_name: str):
    """Run a single configuration."""
    config_dir = Path(__file__).parent.parent / "config"
    
    # Handle both with and without .yaml extension
    if not config_name.endswith(".yaml"):
        config_path = config_dir / f"{config_name}.yaml"
    else:
        config_path = config_dir / config_name
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return None
    
    print("\n" + "=" * 60)
    print(f"Running: {config_path.stem}")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        manager = ExperimentManager(str(config_path))
        results = manager.run_batch()
        
        print("\n" + "=" * 60)
        print(f"✅ {config_path.stem} Complete!")
        print("=" * 60)
        print(f"Samples:   {len(results)}")
        print(f"Output:    {manager.exp_dir}")
        print("=" * 60 + "\n")
        
        return {
            "config": config_path.stem,
            "results": results,
            "output_dir": manager.exp_dir
        }
    
    except Exception as e:
        print(f"\n❌ Error running {config_path.stem}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_configs():
    """Run all configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    configs = [c for c in config_dir.glob("*.yaml") if c.is_file()]
    
    print("\n" + "=" * 60)
    print(f"Running All Configs ({len(configs)} total)")
    print("=" * 60 + "\n")
    
    summary = []
    for cfg in configs:
        result = run_config(cfg.stem)
        if result:
            summary.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for s in summary:
        print(f"  {s['config']:20s} → {len(s['results'])} samples → {s['output_dir']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run topology optimization experiments with different configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available configs
  python scripts/run_experiments.py --list
  
  # Run quick test
  python scripts/run_experiments.py --config quick_test
  
  # Run all configurations
  python scripts/run_experiments.py --all
  
  # Run specific config with custom sample count
  python scripts/run_experiments.py --config high_resolution --samples 10
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available configurations"
    )
    parser.add_argument(
        "--config", "-c",
        help="Configuration name to run (e.g., 'quick_test' or 'quick_test.yaml')"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all available configurations"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        help="Override number of samples (if using --config)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
    elif args.all:
        run_all_configs()
    elif args.config:
        result = run_config(args.config)
        # TODO: Handle custom sample count override if needed
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
