"""
Compare optimal configuration data vs baseline configuration data.
Shows the improvement in sparsity and data quality.
"""

import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_dataset(data_dir: str, name: str, db_floor: float = -150.0) -> Dict:
    """Analyze all samples in a dataset directory."""

    sample_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

    if not sample_files:
        logger.warning(f"No samples found in {data_dir}")
        return None

    logger.info(f"\nAnalyzing {len(sample_files)} samples in {name}")

    hr_stats = []
    lr_stats = []

    for sample_file in sample_files:
        sample_path = os.path.join(data_dir, sample_file)
        data = torch.load(sample_path, map_location="cpu")

        # Analyze HR and LR maps
        hr_map = data["map_hr"].cpu().numpy()
        lr_map = data["map_lr"].cpu().numpy()

        # HR stats
        hr_valid = np.sum(np.isfinite(hr_map) & (hr_map > db_floor))
        hr_total = hr_map.size
        hr_sparsity = 100 * (1 - hr_valid / hr_total)

        valid_hr_data = hr_map[np.isfinite(hr_map) & (hr_map > db_floor)]

        hr_stats.append(
            {
                "sparsity": hr_sparsity,
                "valid_pixels": hr_valid,
                "total_pixels": hr_total,
                "mean": np.mean(valid_hr_data) if len(valid_hr_data) > 0 else np.nan,
                "std": np.std(valid_hr_data) if len(valid_hr_data) > 0 else np.nan,
                "min": np.min(valid_hr_data) if len(valid_hr_data) > 0 else np.nan,
                "max": np.max(valid_hr_data) if len(valid_hr_data) > 0 else np.nan,
            }
        )

        # LR stats
        lr_valid = np.sum(np.isfinite(lr_map) & (lr_map > db_floor))
        lr_total = lr_map.size
        lr_sparsity = 100 * (1 - lr_valid / lr_total)

        valid_lr_data = lr_map[np.isfinite(lr_map) & (lr_map > db_floor)]

        lr_stats.append(
            {
                "sparsity": lr_sparsity,
                "valid_pixels": lr_valid,
                "total_pixels": lr_total,
                "mean": np.mean(valid_lr_data) if len(valid_lr_data) > 0 else np.nan,
                "std": np.std(valid_lr_data) if len(valid_lr_data) > 0 else np.nan,
            }
        )

    return {
        "name": name,
        "n_samples": len(sample_files),
        "hr": hr_stats,
        "lr": lr_stats,
    }


def print_comparison(baseline_stats: Dict, optimal_stats: Dict) -> None:
    """Print detailed comparison."""

    print("\n" + "=" * 80)
    print("BASELINE vs OPTIMAL CONFIGURATION COMPARISON")
    print("=" * 80)

    # HR Maps
    print("\nHIGH RESOLUTION MAPS:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<20} {'Optimal':<20} {'Improvement':<15}")
    print("-" * 80)

    baseline_hr = baseline_stats["hr"]
    optimal_hr = optimal_stats["hr"]

    # Average sparsity
    baseline_sparsity = np.mean([s["sparsity"] for s in baseline_hr])
    optimal_sparsity = np.mean([s["sparsity"] for s in optimal_hr])
    improvement = baseline_sparsity - optimal_sparsity

    print(f"{'Sparsity %':<30} {baseline_sparsity:>15.2f}%    {optimal_sparsity:>15.2f}%    {improvement:>10.2f}% ↓")

    # Valid pixels
    baseline_valid = np.mean([s["valid_pixels"] for s in baseline_hr])
    optimal_valid = np.mean([s["valid_pixels"] for s in optimal_hr])
    improvement_pct = ((optimal_valid - baseline_valid) / baseline_valid) * 100

    print(f"{'Valid pixels':<30} {baseline_valid:>15,.0f}    {optimal_valid:>15,.0f}    {improvement_pct:>10.1f}% ↑")

    # Signal mean
    baseline_mean = np.mean([s["mean"] for s in baseline_hr if not np.isnan(s["mean"])])
    optimal_mean = np.mean([s["mean"] for s in optimal_hr if not np.isnan(s["mean"])])

    print(f"{'Mean signal (dB)':<30} {baseline_mean:>15.2f}    {optimal_mean:>15.2f}")

    # Signal std
    baseline_std = np.mean([s["std"] for s in baseline_hr if not np.isnan(s["std"])])
    optimal_std = np.mean([s["std"] for s in optimal_hr if not np.isnan(s["std"])])

    print(f"{'Std (dB)':<30} {baseline_std:>15.2f}    {optimal_std:>15.2f}")

    print("-" * 80)

    # LR Maps
    print("\nLOW RESOLUTION MAPS:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Baseline':<20} {'Optimal':<20} {'Improvement':<15}")
    print("-" * 80)

    baseline_lr = baseline_stats["lr"]
    optimal_lr = optimal_stats["lr"]

    baseline_sparsity = np.mean([s["sparsity"] for s in baseline_lr])
    optimal_sparsity = np.mean([s["sparsity"] for s in optimal_lr])
    improvement = baseline_sparsity - optimal_sparsity

    print(f"{'Sparsity %':<30} {baseline_sparsity:>15.2f}%    {optimal_sparsity:>15.2f}%    {improvement:>10.2f}% ↓")

    baseline_valid = np.mean([s["valid_pixels"] for s in baseline_lr])
    optimal_valid = np.mean([s["valid_pixels"] for s in optimal_lr])
    improvement_pct = ((optimal_valid - baseline_valid) / baseline_valid) * 100

    print(f"{'Valid pixels':<30} {baseline_valid:>15,.0f}    {optimal_valid:>15,.0f}    {improvement_pct:>10.1f}% ↑")

    print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    hr_sparsity_reduction = np.mean([s["sparsity"] for s in baseline_hr]) - np.mean([s["sparsity"] for s in optimal_hr])
    hr_coverage_increase = (
        (np.mean([s["valid_pixels"] for s in optimal_hr]) - np.mean([s["valid_pixels"] for s in baseline_hr]))
        / np.mean([s["valid_pixels"] for s in baseline_hr])
    ) * 100

    print(f"✅ Sparsity reduced by {hr_sparsity_reduction:.1f} percentage points (HR)")
    print(f"✅ Coverage increased by {hr_coverage_increase:.1f}% (HR)")
    print("✅ Configuration: 9 transmitters, 5M rays, 7 bounces")
    print("=" * 80)


def plot_comparison(
    baseline_stats: Dict, optimal_stats: Dict, save_path: str = "src/poc/optimal_vs_baseline.png"
) -> None:
    """Create visualization comparing baseline vs optimal."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs Optimal Configuration Comparison", fontsize=16, fontweight="bold")

    # HR Sparsity - use averages instead of per-sample bars
    ax = axes[0, 0]
    baseline_hr_sparsity_avg = np.mean([s["sparsity"] for s in baseline_stats["hr"]])
    optimal_hr_sparsity_avg = np.mean([s["sparsity"] for s in optimal_stats["hr"]])

    categories = ["Baseline\n(1 TX, 1M rays)", "Optimal\n(9 TX, 5M rays)"]
    values = [baseline_hr_sparsity_avg, optimal_hr_sparsity_avg]
    colors = ["#ff7f0e", "#2ca02c"]

    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("HR Map Sparsity Comparison")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + 2, f"{val:.1f}%", ha="center", fontweight="bold")

    # HR Valid Pixels
    ax = axes[0, 1]
    baseline_hr_valid_avg = np.mean([s["valid_pixels"] for s in baseline_stats["hr"]])
    optimal_hr_valid_avg = np.mean([s["valid_pixels"] for s in optimal_stats["hr"]])

    values = [baseline_hr_valid_avg, optimal_hr_valid_avg]

    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylabel("Valid Pixels")
    ax.set_title("HR Map Coverage Comparison")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + 5000, f"{val:,.0f}", ha="center", fontweight="bold")

    # Average comparison
    ax = axes[1, 0]
    categories = ["HR Sparsity\n(%)", "LR Sparsity\n(%)", "HR Valid\nPixels (k)", "LR Valid\nPixels (k)"]

    baseline_vals = [
        np.mean([s["sparsity"] for s in baseline_stats["hr"]]),
        np.mean([s["sparsity"] for s in baseline_stats["lr"]]),
        np.mean([s["valid_pixels"] for s in baseline_stats["hr"]]) / 1000,
        np.mean([s["valid_pixels"] for s in baseline_stats["lr"]]) / 1000,
    ]

    optimal_vals = [
        np.mean([s["sparsity"] for s in optimal_stats["hr"]]),
        np.mean([s["sparsity"] for s in optimal_stats["lr"]]),
        np.mean([s["valid_pixels"] for s in optimal_stats["hr"]]) / 1000,
        np.mean([s["valid_pixels"] for s in optimal_stats["lr"]]) / 1000,
    ]

    x_pos = np.arange(len(categories))
    width = 0.35

    ax.bar(x_pos - width / 2, baseline_vals, width, label="Baseline", color="#ff7f0e", alpha=0.8)
    ax.bar(x_pos + width / 2, optimal_vals, width, label="Optimal", color="#2ca02c", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_title("Average Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Improvement percentages
    ax = axes[1, 1]

    improvements = [
        baseline_vals[0] - optimal_vals[0],  # Sparsity reduction (HR)
        baseline_vals[1] - optimal_vals[1],  # Sparsity reduction (LR)
        ((optimal_vals[2] - baseline_vals[2]) / baseline_vals[2]) * 100,  # Coverage increase (HR)
        ((optimal_vals[3] - baseline_vals[3]) / baseline_vals[3]) * 100,  # Coverage increase (LR)
    ]

    improvement_labels = [
        "HR Sparsity\nReduction (%)",
        "LR Sparsity\nReduction (%)",
        "HR Coverage\nIncrease (%)",
        "LR Coverage\nIncrease (%)",
    ]

    colors = ["#2ca02c" if imp > 0 else "#d62728" for imp in improvements]

    ax.bar(improvement_labels, improvements, color=colors, alpha=0.8)
    ax.set_ylabel("Improvement")
    ax.set_title("Configuration Improvements")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"\n📊 Comparison plot saved to: {save_path}")
    plt.show()


def visualize_radio_maps(
    baseline_dir: str, optimal_dir: str, save_path: str = "src/poc/radio_map_comparison.png"
) -> None:
    """Visualize actual radio maps side by side."""

    # Load first sample from each
    baseline_files = sorted([f for f in os.listdir(baseline_dir) if f.endswith(".pt")])
    optimal_files = sorted([f for f in os.listdir(optimal_dir) if f.endswith(".pt")])

    if not baseline_files or not optimal_files:
        logger.warning("Cannot visualize - missing samples")
        return

    baseline_sample = torch.load(os.path.join(baseline_dir, baseline_files[0]), map_location="cpu")
    optimal_sample = torch.load(os.path.join(optimal_dir, optimal_files[0]), map_location="cpu")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Radio Map Quality Comparison: Baseline vs Optimal", fontsize=18, fontweight="bold")

    # Baseline HR
    ax = axes[0, 0]
    baseline_hr = baseline_sample["map_hr"].squeeze().cpu().numpy()
    im = ax.imshow(baseline_hr, cmap="viridis", origin="upper", vmin=-150, vmax=-70)
    baseline_sparsity = (baseline_hr <= -150).sum() / baseline_hr.size * 100
    ax.set_title(
        f"Baseline HR Map\n(1 TX, 1M rays, 5 bounces)\nSparsity: {baseline_sparsity:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signal (dB)")

    # Optimal HR
    ax = axes[0, 1]
    optimal_hr = optimal_sample["map_hr"].squeeze().cpu().numpy()
    im = ax.imshow(optimal_hr, cmap="viridis", origin="upper", vmin=-150, vmax=-70)
    optimal_sparsity = (optimal_hr <= -150).sum() / optimal_hr.size * 100
    ax.set_title(
        f"Optimal HR Map\n(9 TX, 5M rays, 7 bounces)\nSparsity: {optimal_sparsity:.1f}%",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signal (dB)")

    # Baseline LR
    ax = axes[1, 0]
    baseline_lr = baseline_sample["map_lr"].squeeze().cpu().numpy()
    im = ax.imshow(baseline_lr, cmap="viridis", origin="upper", vmin=-150, vmax=-70)
    ax.set_title(
        f"Baseline LR Map\nSparsity: {((baseline_lr <= -150).sum() / baseline_lr.size * 100):.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signal (dB)")

    # Optimal LR
    ax = axes[1, 1]
    optimal_lr = optimal_sample["map_lr"].squeeze().cpu().numpy()
    im = ax.imshow(optimal_lr, cmap="viridis", origin="upper", vmin=-150, vmax=-70)
    ax.set_title(
        f"Optimal LR Map\nSparsity: {((optimal_lr <= -150).sum() / optimal_lr.size * 100):.1f}%",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Signal (dB)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"📷 Radio map visualization saved to: {save_path}")
    plt.show()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare baseline vs optimal configuration")
    parser.add_argument(
        "--baseline_dir", type=str, default="src/poc/pattern_analysis/pattern_iso", help="Directory with baseline data"
    )
    parser.add_argument("--optimal_dir", type=str, default="src/poc/optimal_data", help="Directory with optimal data")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COMPARING BASELINE vs OPTIMAL CONFIGURATION")
    logger.info("=" * 80)

    # Analyze both datasets
    baseline_stats = analyze_dataset(args.baseline_dir, "BASELINE (1 TX, 1M rays, 5 bounces)")
    optimal_stats = analyze_dataset(args.optimal_dir, "OPTIMAL (9 TX, 5M rays, 7 bounces)")

    if baseline_stats is None or optimal_stats is None:
        logger.error("Could not load data. Check directory paths.")
        return

    # Print comparison
    print_comparison(baseline_stats, optimal_stats)

    # Plot comparison
    plot_comparison(baseline_stats, optimal_stats)

    # Visualize actual radio maps
    visualize_radio_maps(args.baseline_dir, args.optimal_dir)

    logger.info("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
