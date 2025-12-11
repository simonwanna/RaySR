"""
Analyze how different antenna patterns affect radio map data.
Compares sparsity, value distributions, and tensor statistics across patterns.
"""

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_tensor(tensor: torch.Tensor, name: str, db_floor: float = -150.0) -> Dict:
    """Analyze a single radio map tensor.

    Args:
        tensor: Radio map tensor
        name: Name for this analysis
        db_floor: dB value used as floor for missing data (default: -150.0)
    """
    # Remove channel dimension if present
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Convert to numpy for analysis
    data = tensor.cpu().numpy()

    # Identify sparse pixels: anything at or below db_floor is considered "no signal"
    sparse_mask = data <= db_floor
    valid_mask = ~sparse_mask & np.isfinite(data)

    # Calculate statistics
    stats = {
        "name": name,
        "shape": data.shape,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        # Sparsity metrics
        "total_pixels": data.size,
        "sparse_pixels": int(np.sum(sparse_mask)),  # Pixels at db_floor
        "inf_pixels": int(np.sum(np.isinf(data))),
        "nan_pixels": int(np.sum(np.isnan(data))),
        "valid_pixels": int(np.sum(valid_mask)),
        "db_floor_pixels": int(np.sum(data == db_floor)),  # Exactly at floor
    }

    # Calculate sparsity percentage (sparse + inf + nan)
    stats["sparsity_percent"] = 100 * (1 - stats["valid_pixels"] / stats["total_pixels"])

    # Value distribution (for VALID pixels only - excluding db_floor values)
    valid_data = data[valid_mask]
    if len(valid_data) > 0:
        stats["valid_min"] = float(np.min(valid_data))
        stats["valid_max"] = float(np.max(valid_data))
        stats["valid_mean"] = float(np.mean(valid_data))
        stats["valid_std"] = float(np.std(valid_data))
        # Percentiles
        stats["p25"] = float(np.percentile(valid_data, 25))
        stats["p50"] = float(np.percentile(valid_data, 50))
        stats["p75"] = float(np.percentile(valid_data, 75))
        stats["p95"] = float(np.percentile(valid_data, 95))
    else:
        # No valid data
        stats["valid_min"] = float("nan")
        stats["valid_max"] = float("nan")
        stats["valid_mean"] = float("nan")
        stats["valid_std"] = float("nan")
        stats["p25"] = float("nan")
        stats["p50"] = float("nan")
        stats["p75"] = float("nan")
        stats["p95"] = float("nan")

    return stats


def compare_patterns(data_dir: str, patterns: List[str], db_floor: float = -150.0) -> None:
    """Compare radio maps generated with different antenna patterns.

    Args:
        data_dir: Directory containing pattern data
        patterns: List of antenna patterns to compare
        db_floor: dB value used as floor for missing data
    """

    results = {}

    for pattern in patterns:
        pattern_dir = os.path.join(data_dir, f"pattern_{pattern}")
        if not os.path.exists(pattern_dir):
            logger.warning(f"Directory not found: {pattern_dir}")
            continue

        # Load samples
        sample_files = sorted([f for f in os.listdir(pattern_dir) if f.endswith(".pt")])

        if not sample_files:
            logger.warning(f"No samples found in {pattern_dir}")
            continue

        logger.info(f"\nAnalyzing {len(sample_files)} samples for pattern: {pattern}")

        pattern_stats = {"hr": [], "lr": []}

        for sample_file in sample_files:
            sample_path = os.path.join(pattern_dir, sample_file)
            data = torch.load(sample_path, map_location="cpu")

            # Analyze HR and LR maps with db_floor
            hr_stats = analyze_tensor(data["map_hr"], f"{pattern}_hr", db_floor=db_floor)
            lr_stats = analyze_tensor(data["map_lr"], f"{pattern}_lr", db_floor=db_floor)

            pattern_stats["hr"].append(hr_stats)
            pattern_stats["lr"].append(lr_stats)

        results[pattern] = pattern_stats

    # Print comparison
    print_comparison(results, patterns)

    # Plot comparison
    plot_comparison(results, patterns, data_dir)

    return results


def print_comparison(results: Dict, patterns: List[str]) -> None:
    """Print comparison statistics."""

    print("\n" + "=" * 80)
    print("ANTENNA PATTERN COMPARISON - HIGH RESOLUTION MAPS")
    print("=" * 80)

    for resolution in ["hr", "lr"]:
        print(f"\n{'HIGH RESOLUTION' if resolution == 'hr' else 'LOW RESOLUTION'} MAPS:")
        print("-" * 80)

        # Header
        print(f"{'Metric':<25} " + " ".join([f"{p:>15}" for p in patterns]))
        print("-" * 80)

        # Metrics to compare
        metrics = [
            ("Sparsity %", "sparsity_percent"),
            ("Valid pixels", "valid_pixels"),
            ("Mean (dB)", "valid_mean"),
            ("Std (dB)", "valid_std"),
            ("Min (dB)", "valid_min"),
            ("Max (dB)", "valid_max"),
            ("25th percentile", "p25"),
            ("Median", "p50"),
            ("75th percentile", "p75"),
            ("95th percentile", "p95"),
        ]

        for metric_name, metric_key in metrics:
            values = []
            for pattern in patterns:
                if pattern in results and resolution in results[pattern]:
                    # Average across all samples
                    pattern_values = [s.get(metric_key, 0) for s in results[pattern][resolution]]
                    avg_value = np.mean(pattern_values)
                    values.append(avg_value)
                else:
                    values.append(0)

            print(f"{metric_name:<25} " + " ".join([f"{v:>15.2f}" for v in values]))

        print("-" * 80)


def plot_comparison(results: Dict, patterns: List[str], save_dir: str) -> None:
    """Create visualization comparing patterns."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Antenna Pattern Comparison", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for res_idx, resolution in enumerate(["hr", "lr"]):
        # 1. Sparsity comparison
        ax = axes[res_idx, 0]
        sparsities = []
        for pattern in patterns:
            if pattern in results:
                vals = [s["sparsity_percent"] for s in results[pattern][resolution]]
                sparsities.append(np.mean(vals))
            else:
                sparsities.append(0)

        ax.bar(patterns, sparsities, color=colors[: len(patterns)])
        ax.set_ylabel("Sparsity (%)")
        ax.set_title(f"{'HR' if resolution == 'hr' else 'LR'} Map Sparsity")
        ax.grid(axis="y", alpha=0.3)

        # 2. Value distribution (box plot)
        ax = axes[res_idx, 1]
        data_for_box = []
        for pattern in patterns:
            if pattern in results:
                # Collect valid means from all samples
                vals = [s.get("valid_mean", 0) for s in results[pattern][resolution]]
                data_for_box.append(vals)
            else:
                data_for_box.append([])

        bp = ax.boxplot(data_for_box, labels=patterns, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(patterns)]):
            patch.set_facecolor(color)
        ax.set_ylabel("Mean Signal (dB)")
        ax.set_title(f"{'HR' if resolution == 'hr' else 'LR'} Value Distribution")
        ax.grid(axis="y", alpha=0.3)

        # 3. Valid pixel count
        ax = axes[res_idx, 2]
        valid_pixels = []
        for pattern in patterns:
            if pattern in results:
                vals = [s["valid_pixels"] for s in results[pattern][resolution]]
                valid_pixels.append(np.mean(vals))
            else:
                valid_pixels.append(0)

        ax.bar(patterns, valid_pixels, color=colors[: len(patterns)])
        ax.set_ylabel("Valid Pixels")
        ax.set_title(f"{'HR' if resolution == 'hr' else 'LR'} Coverage")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(save_dir, "antenna_pattern_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison plot saved to: {output_path}")
    plt.show()


def visualize_sample_maps(data_dir: str, patterns: List[str], sample_idx: int = 0) -> None:
    """Visualize actual radio maps for different patterns side-by-side."""

    fig, axes = plt.subplots(2, len(patterns), figsize=(5 * len(patterns), 10))
    fig.suptitle(f"Radio Maps - Sample {sample_idx}", fontsize=16, fontweight="bold")

    for col, pattern in enumerate(patterns):
        pattern_dir = os.path.join(data_dir, f"pattern_{pattern}")
        sample_files = sorted([f for f in os.listdir(pattern_dir) if f.endswith(".pt")])

        if sample_idx >= len(sample_files):
            logger.warning(f"Sample {sample_idx} not found for pattern {pattern}")
            continue

        sample_path = os.path.join(pattern_dir, sample_files[sample_idx])
        data = torch.load(sample_path, map_location="cpu")

        # HR map
        hr_map = data["map_hr"].squeeze().cpu().numpy()
        im1 = axes[0, col].imshow(hr_map, cmap="viridis", origin="upper")
        axes[0, col].set_title(f"{pattern.upper()} - HR")
        axes[0, col].axis("off")
        plt.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)

        # LR map
        lr_map = data["map_lr"].squeeze().cpu().numpy()
        im2 = axes[1, col].imshow(lr_map, cmap="viridis", origin="upper")
        axes[1, col].set_title(f"{pattern.upper()} - LR")
        axes[1, col].axis("off")
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save
    output_path = os.path.join(data_dir, f"sample_{sample_idx}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Sample comparison saved to: {output_path}")
    plt.show()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze antenna pattern effects on radio maps")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing pattern_* subdirectories")
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["iso", "dipole", "hw_dipole", "tr38901"],
        help="Antenna patterns to compare",
    )
    parser.add_argument("--visualize_sample", type=int, default=0, help="Sample index to visualize")
    parser.add_argument(
        "--db_floor", type=float, default=-150.0, help="dB floor value for missing data (default: -150.0)"
    )

    args = parser.parse_args()

    # Run comparison
    compare_patterns(args.data_dir, args.patterns, db_floor=args.db_floor)

    # Visualize sample maps
    visualize_sample_maps(args.data_dir, args.patterns, args.visualize_sample)

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
