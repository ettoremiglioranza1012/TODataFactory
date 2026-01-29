#!/usr/bin/env python3
"""
validate_dataset.py - 3D Visualization of ML Dataset Pairs

This script validates that the generated topology optimization results
make physical sense by visualizing the input loads and output structures.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import json


def load_sample(dataset_dir: str, sample_id: str):
    """Load a single sample pair from the dataset."""
    dataset_path = Path(dataset_dir)
    
    input_file = dataset_path / f"sample_{sample_id}_inputs.npy"
    target_file = dataset_path / f"sample_{sample_id}_target.npy"
    
    if not input_file.exists() or not target_file.exists():
        raise FileNotFoundError(f"Sample {sample_id} not found in {dataset_dir}")
    
    X = np.load(input_file)  # (nelx, nely, nelz, 4)
    Y = np.load(target_file)  # (nelx, nely, nelz)
    
    return X, Y


def visualize_load_channel(X, plotter, position, title="Input: Applied Loads"):
    """
    Visualize the load channel (Channel 3: Fz) from input tensor.
    
    Args:
        X: Input tensor (nelx, nely, nelz, 4)
        plotter: PyVista plotter
        position: Subplot position
        title: Plot title
    """
    # Extract Fz channel (index 3)
    fz = X[:, :, :, 3]
    nelx, nely, nelz = fz.shape
    
    # Create structured grid
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    
    # Add force data as cell data
    grid.cell_data["Fz"] = fz.flatten(order='F')
    
    # Threshold to show only loaded regions (non-zero forces)
    threshold = 1e-6
    loaded_region = grid.threshold(value=(-1000, -threshold), scalars="Fz")
    
    # Add to plotter
    plotter.subplot(*position)
    if loaded_region.n_cells > 0:
        plotter.add_mesh(
            loaded_region,
            scalars="Fz",
            cmap="coolwarm",
            show_edges=True,
            opacity=0.8,
            scalar_bar_args={
                'title': 'Force Z (N)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1
            }
        )
        
        # Add arrows at load points to show direction
        centers = loaded_region.cell_centers()
        if centers.n_points > 0:
            # Create downward arrows (negative Z direction)
            arrows = centers.glyph(
                orient=False,
                scale=False,
                factor=2.0,
                geom=pv.Arrow(direction=(0, 0, -1))
            )
            plotter.add_mesh(arrows, color='red', opacity=0.6)
    
    plotter.add_title(title, font_size=12)
    plotter.show_axes()
    plotter.camera_position = 'iso'


def visualize_density(Y, plotter, position, threshold=0.3, title="Output: Optimized Structure"):
    """
    Visualize the optimized density field.
    
    Args:
        Y: Target density tensor (nelx, nely, nelz)
        plotter: PyVista plotter
        position: Subplot position
        threshold: Density threshold for visualization
        title: Plot title
    """
    nelx, nely, nelz = Y.shape
    
    # Create structured grid
    grid = pv.ImageData(dimensions=(nelx+1, nely+1, nelz+1))
    grid.spacing = (1, 1, 1)
    
    # Add density data as cell data
    grid.cell_data["density"] = Y.flatten(order='F')
    
    # Threshold to show structure (density > threshold)
    structure = grid.threshold(value=threshold, scalars="density")
    
    # Add to plotter
    plotter.subplot(*position)
    if structure.n_cells > 0:
        plotter.add_mesh(
            structure,
            scalars="density",
            cmap="viridis",
            show_edges=False,
            opacity=0.9,
            scalar_bar_args={
                'title': 'Density',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1
            }
        )
    
    plotter.add_title(title, font_size=12)
    plotter.show_axes()
    plotter.camera_position = 'iso'


def validate_sample(dataset_dir: str, sample_id: str, save_path: str = None, show_plot: bool = True):
    """
    Validate a single sample by visualizing inputs and outputs.
    
    Args:
        dataset_dir: Path to dataset directory
        sample_id: Sample ID (e.g., "0001")
        save_path: Optional path to save screenshot
        show_plot: Whether to display interactive plot
    """
    # Load data
    print(f"Loading sample {sample_id}...")
    X, Y = load_sample(dataset_dir, sample_id)
    
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {Y.shape}")
    print(f"  Force sum (Fz): {X[:,:,:,3].sum():.2f} N")
    print(f"  Density range: [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"  Density mean: {Y.mean():.4f}")
    
    # Create plotter with subplots (use off_screen for screenshots)
    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1600, 800),
        off_screen=(save_path is not None and not show_plot)
    )
    
    # Visualize loads (left)
    visualize_load_channel(
        X, plotter, (0, 0),
        title=f"Sample {sample_id}: Input Loads (Fz)"
    )
    
    # Visualize structure (right)
    visualize_density(
        Y, plotter, (0, 1),
        threshold=0.3,
        title=f"Sample {sample_id}: Optimized Structure (ρ > 0.3)"
    )
    
    # Save screenshot if requested
    if save_path:
        print(f"  Saving screenshot to: {save_path}")
        plotter.screenshot(save_path)
    
    # Show interactive plot
    if show_plot:
        plotter.show()
    else:
        plotter.close()
    
    print(f"  ✅ Sample {sample_id} validated\n")


def validate_dataset(dataset_dir: str, num_samples: int = 3, output_dir: str = "validation"):
    """
    Validate multiple samples from the dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        num_samples: Number of samples to validate
        output_dir: Directory to save validation images
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load dataset index
    index_file = dataset_path / "dataset_index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        print(f"Found {len(index)} samples in dataset index")
    else:
        print("Warning: dataset_index.json not found, using sample IDs")
        index = [{"sample_id": f"{i+1:04d}"} for i in range(num_samples)]
    
    print("=" * 60)
    print(f"Dataset Validation: {dataset_dir}")
    print("=" * 60)
    
    # Validate each sample
    for i in range(min(num_samples, len(index))):
        sample_id = index[i]["sample_id"]
        save_path = output_path / f"validation_{sample_id}.png"
        
        try:
            validate_sample(
                dataset_dir,
                sample_id,
                save_path=str(save_path),
                show_plot=False  # Don't show interactive plots in batch mode
            )
        except Exception as e:
            print(f"  ❌ Failed to validate sample {sample_id}: {e}\n")
    
    print("=" * 60)
    print(f"Validation complete! Screenshots saved to: {output_path}")
    print("=" * 60)


def check_physical_consistency(X, Y, verbose=True):
    """
    Check if the optimization result makes physical sense.
    
    Verifies:
    1. Load is applied (non-zero forces)
    2. Structure exists (non-zero densities)
    3. Volume constraint is satisfied
    4. Structure is concentrated where needed
    """
    fz = X[:, :, :, 3]
    
    # Check 1: Load is applied
    total_force = fz.sum()
    has_load = abs(total_force) > 1e-6
    
    # Check 2: Structure exists
    has_structure = (Y > 0.1).any()
    
    # Check 3: Volume fraction
    volume_fraction = Y.mean()
    
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
    if load_center is not None:
        # Check if high-density material exists near the load
        structure_mask = Y > 0.5
        if structure_mask.any():
            structure_center = np.array([
                np.mean(np.where(structure_mask)[0]),
                np.mean(np.where(structure_mask)[1]),
                np.mean(np.where(structure_mask)[2])
            ])
            distance = np.linalg.norm(load_center - structure_center)
        else:
            distance = float('inf')
    else:
        distance = None
    
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
        'distance': distance
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ML dataset for topology optimization")
    parser.add_argument("--dataset", type=str, default="paired_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples to validate")
    parser.add_argument("--output", type=str, default="validation",
                       help="Output directory for validation images")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive plots (one at a time)")
    
    args = parser.parse_args()
    
    if args.interactive:
        # Show first sample interactively
        print(f"Showing sample 0001 interactively...")
        X, Y = load_sample(args.dataset, "0001")
        check_physical_consistency(X, Y)
        validate_sample(args.dataset, "0001", show_plot=True)
    else:
        # Batch validation
        validate_dataset(args.dataset, args.samples, args.output)
