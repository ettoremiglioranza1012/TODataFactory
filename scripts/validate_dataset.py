#!/usr/bin/env python3
"""
validate_dataset.py - CLI for validating topology optimization datasets.

This is a thin CLI wrapper that uses the topopt_ml library functions.
For programmatic use, import directly from topopt_ml.visualization.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML dataset for topology optimization"
    )
    parser.add_argument(
        "--dataset", type=str, default="paired_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of samples to validate"
    )
    parser.add_argument(
        "--output", type=str, default="validation",
        help="Output directory for validation images"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Show interactive plots (one at a time)"
    )
    parser.add_argument(
        "--math-only", action="store_true",
        help="Run math validation only (no visualization)"
    )
    
    args = parser.parse_args()
    
    # Import from library
    from topopt_ml.io import load_sample, load_dataset_index
    from topopt_ml.visualization import (
        validate_sample_visual,
        validate_sample_math,
        check_physical_consistency,
        compare_input_output,
        validate_dataset,
        print_validation_report,
    )
    
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset index
    try:
        index = load_dataset_index(args.dataset)
        print(f"Found {len(index)} samples in dataset index")
    except FileNotFoundError:
        print(f"Warning: dataset_index.json not found in {args.dataset}")
        index = [{"sample_id": f"{i+1:04d}"} for i in range(args.samples)]
    
    print("=" * 60)
    print(f"Dataset Validation: {args.dataset}")
    print("=" * 60)
    
    if args.math_only:
        # Math-only validation
        results = validate_dataset(args.dataset, max_samples=args.samples)
        print_validation_report(results)
    elif args.interactive:
        # Interactive visual validation
        sample_id = index[0]["sample_id"]
        print(f"Showing sample {sample_id} interactively...")
        X, Y = load_sample(args.dataset, sample_id)
        check_physical_consistency(X, Y, verbose=True)
        validate_sample_visual(
            X, Y,
            sample_id=sample_id,
            show_plot=True
        )
    else:
        # Batch validation with screenshots
        for i in range(min(args.samples, len(index))):
            sample_id = index[i]["sample_id"]
            save_path = output_path / f"validation_{sample_id}.png"
            
            try:
                X, Y = load_sample(args.dataset, sample_id)
                validate_sample_visual(
                    X, Y,
                    sample_id=sample_id,
                    save_path=str(save_path),
                    show_plot=False
                )
            except Exception as e:
                print(f"  ❌ Failed to validate sample {sample_id}: {e}\n")
        
        print("=" * 60)
        print(f"Validation complete! Screenshots saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
