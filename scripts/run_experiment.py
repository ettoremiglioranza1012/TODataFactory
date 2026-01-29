#!/usr/bin/env python3
"""
run_experiment.py - Unified CLI for running experiments

Usage:
    python scripts/run_experiment.py --config config/default.yaml --samples 10
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from topopt_ml.experiments import ExperimentManager


def main():
    parser = argparse.ArgumentParser(
        description="TopOpt Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        help="Number of samples to generate (overrides config)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    manager = ExperimentManager(args.config)
    
    if args.output:
        manager.exp_dir = Path(args.output)
        manager.exp_dir.mkdir(parents=True, exist_ok=True)
    
    results = manager.run_batch(args.samples)
    
    print(f"\n✅ Generated {len(results)} samples")
    print(f"   Output: {manager.exp_dir}")


if __name__ == "__main__":
    main()
