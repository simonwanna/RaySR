"""
Investigate how to get denser HR data with fewer missing points.

This script tests different configurations of the Radio Map Solver to reduce sparsity:
1. Increase number of samples per transmitter
2. Increase max_depth (ray bounces)
3. Adjust cell size
4. Multiple transmitters with different positions
"""

import logging
import os
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import sionna
import torch
from omegaconf import DictConfig
from sionna.rt import load_scene

from poc.data_modules.generator import RadioMapDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_sparsity(radio_map: torch.Tensor, name: str, db_floor: float = -150.0) -> Dict:
    """Analyze sparsity of a radio map.

    Args:
        radio_map: Radio map tensor
        name: Name for this analysis
        db_floor: dB value used as floor for missing data (default: -150.0)
    """
    data = radio_map.squeeze().cpu().numpy()

    total = data.size

    # Pixels at or below db_floor are considered sparse (no signal)
    sparse_pixels = np.sum(data <= db_floor)
    invalid_pixels = np.sum(~np.isfinite(data))

    # Valid pixels are finite AND above db_floor
    valid_pixels = np.sum(np.isfinite(data) & (data > db_floor))

    # Sparsity includes both sparse pixels and invalid pixels
    sparsity_percent = 100 * (1 - valid_pixels / total)

    # Get statistics on valid data only
    valid_data = data[np.isfinite(data) & (data > db_floor)]

    result = {
        "name": name,
        "total_pixels": total,
        "sparse_pixels": int(sparse_pixels),  # At db_floor
        "invalid_pixels": int(invalid_pixels),  # inf/nan
        "valid_pixels": int(valid_pixels),
        "sparsity_percent": sparsity_percent,
        "db_floor": db_floor,
    }

    # Add statistics for valid data
    if len(valid_data) > 0:
        result.update(
            {
                "valid_mean": float(np.mean(valid_data)),
                "valid_std": float(np.std(valid_data)),
                "valid_min": float(np.min(valid_data)),
                "valid_max": float(np.max(valid_data)),
            }
        )
    else:
        result.update(
            {
                "valid_mean": float("nan"),
                "valid_std": float("nan"),
                "valid_min": float("nan"),
                "valid_max": float("nan"),
            }
        )

    return result


def test_samples_per_tx(scene: object, tx_config: object, save_dir: str) -> None:
    """Test different values of samples_per_tx."""

    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Varying samples_per_tx (number of rays)")
    logger.info("=" * 80)

    # Test different sample counts
    sample_counts = [10**5, 5 * 10**5, 10**6, 5 * 10**6, 10**7]

    results = []

    for samples_per_tx in sample_counts:
        logger.info(f"\nTesting samples_per_tx = {samples_per_tx:,.0f}")

        # Create generator with custom samples_per_tx
        generator = RadioMapDataGenerator(
            metric_type="path_gain",
            n_samples=1,
            dataset_path=os.path.join(save_dir, f"test_samples_{samples_per_tx}"),
            to_db=True,
            db_floor=-150.0,
            scene=scene,
        )

        # Temporarily modify the generator to use custom samples_per_tx
        def custom_generate(sample_id: int, config: object, scene_corners: object):
            # Build transmitters
            tx_positions, grid_info = generator.builder.build(config, scene_corners, generator.tx_grid_info)
            cx, cy = grid_info["center_x"], grid_info["center_y"]

            # Generate HR with custom samples_per_tx
            rm_hr = generator.rm_solver(
                generator.scene,
                max_depth=5,
                samples_per_tx=samples_per_tx,  # Use custom value
                cell_size=config.hr_cell_size,
                center=[cx, cy, 0.0],
                size=[config.coverage_size, config.coverage_size],
                orientation=[0, 0, 0],
            )

            metric_hr = generator._extract_metric(rm_hr)
            if metric_hr.dim() == 3:
                map_hr = torch.max(metric_hr, dim=0)[0]
            else:
                map_hr = metric_hr

            if generator.to_db:
                map_hr = generator._apply_db_conversion(map_hr, generator.db_floor)

            return map_hr

        # Get scene corners
        scene_corners = generator._get_scene_boundary(tx_config.margin)

        # Generate sample
        map_hr = custom_generate(1, tx_config, scene_corners)

        # Analyze
        stats = analyze_sparsity(map_hr, f"samples_{samples_per_tx}")
        results.append(stats)

        logger.info(f"  Sparsity: {stats['sparsity_percent']:.2f}%")
        logger.info(f"  Valid pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(
        [r["name"].split("_")[1] for r in results],
        [r["sparsity_percent"] for r in results],
        marker="o",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Samples per TX")
    ax1.set_ylabel("Sparsity (%)")
    ax1.set_title("Effect of samples_per_tx on Sparsity")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    ax2.plot(
        [r["name"].split("_")[1] for r in results],
        [r["valid_pixels"] for r in results],
        marker="o",
        linewidth=2,
        markersize=8,
        color="green",
    )
    ax2.set_xlabel("Samples per TX")
    ax2.set_ylabel("Valid Pixels")
    ax2.set_title("Effect of samples_per_tx on Coverage")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "samples_per_tx_comparison.png"), dpi=300)
    logger.info(f"\nPlot saved to: {os.path.join(save_dir, 'samples_per_tx_comparison.png')}")
    plt.show()


def test_max_depth(scene: object, tx_config: object, save_dir: str) -> None:
    """Test different values of max_depth (ray bounces)."""

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Varying max_depth (ray bounces)")
    logger.info("=" * 80)

    # Test different depths
    depths = [3, 5, 7, 10]

    results = []

    for max_depth in depths:
        logger.info(f"\nTesting max_depth = {max_depth}")

        generator = RadioMapDataGenerator(
            metric_type="path_gain",
            n_samples=1,
            dataset_path=os.path.join(save_dir, f"test_depth_{max_depth}"),
            to_db=True,
            db_floor=-150.0,
            scene=scene,
        )

        # Custom generate with varying max_depth
        def custom_generate(sample_id: int, config: object, scene_corners: object):
            tx_positions, grid_info = generator.builder.build(config, scene_corners, generator.tx_grid_info)
            cx, cy = grid_info["center_x"], grid_info["center_y"]

            rm_hr = generator.rm_solver(
                generator.scene,
                max_depth=max_depth,  # Use custom value
                samples_per_tx=10**6,
                cell_size=config.hr_cell_size,
                center=[cx, cy, 0.0],
                size=[config.coverage_size, config.coverage_size],
                orientation=[0, 0, 0],
            )

            metric_hr = generator._extract_metric(rm_hr)
            if metric_hr.dim() == 3:
                map_hr = torch.max(metric_hr, dim=0)[0]
            else:
                map_hr = metric_hr

            if generator.to_db:
                map_hr = generator._apply_db_conversion(map_hr, generator.db_floor)

            return map_hr

        scene_corners = generator._get_scene_boundary(tx_config.margin)
        map_hr = custom_generate(1, tx_config, scene_corners)

        stats = analyze_sparsity(map_hr, f"depth_{max_depth}")
        results.append(stats)

        logger.info(f"  Sparsity: {stats['sparsity_percent']:.2f}%")
        logger.info(f"  Valid pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(depths, [r["sparsity_percent"] for r in results], marker="o", linewidth=2, markersize=8, color="orange")
    ax1.set_xlabel("Max Depth (bounces)")
    ax1.set_ylabel("Sparsity (%)")
    ax1.set_title("Effect of max_depth on Sparsity")
    ax1.grid(True, alpha=0.3)

    ax2.plot(depths, [r["valid_pixels"] for r in results], marker="o", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Max Depth (bounces)")
    ax2.set_ylabel("Valid Pixels")
    ax2.set_title("Effect of max_depth on Coverage")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "max_depth_comparison.png"), dpi=300)
    logger.info(f"\nPlot saved to: {os.path.join(save_dir, 'max_depth_comparison.png')}")
    plt.show()


def test_multi_transmitters(scene: object, tx_config: object, save_dir: str) -> None:
    """Test effect of multiple transmitters."""

    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Varying number of transmitters")
    logger.info("=" * 80)

    # Test different numbers of transmitters
    n_tx_values = [1, 2, 4, 9]

    results = []

    for n_tx in n_tx_values:
        logger.info(f"\nTesting n_tx = {n_tx}")

        # Create new config with different n_tx
        from dataclasses import replace

        config = replace(tx_config, n_tx=n_tx)

        generator = RadioMapDataGenerator(
            metric_type="path_gain",
            n_samples=1,
            dataset_path=os.path.join(save_dir, f"test_ntx_{n_tx}"),
            to_db=True,
            db_floor=-150.0,
            scene=scene,
        )

        scene_corners = generator._get_scene_boundary(config.margin)
        sample = generator._generate_sample(1, config, scene_corners)

        stats = analyze_sparsity(sample.map_hr, f"n_tx_{n_tx}")
        results.append(stats)

        logger.info(f"  Sparsity: {stats['sparsity_percent']:.2f}%")
        logger.info(f"  Valid pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(n_tx_values, [r["sparsity_percent"] for r in results], marker="o", linewidth=2, markersize=8, color="red")
    ax1.set_xlabel("Number of Transmitters")
    ax1.set_ylabel("Sparsity (%)")
    ax1.set_title("Effect of n_tx on Sparsity")
    ax1.grid(True, alpha=0.3)

    ax2.plot(n_tx_values, [r["valid_pixels"] for r in results], marker="o", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Number of Transmitters")
    ax2.set_ylabel("Valid Pixels")
    ax2.set_title("Effect of n_tx on Coverage")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "n_tx_comparison.png"), dpi=300)
    logger.info(f"\nPlot saved to: {os.path.join(save_dir, 'n_tx_comparison.png')}")
    plt.show()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run density experiments."""

    logger.info("=" * 80)
    logger.info("INVESTIGATING RADIO MAP DENSITY")
    logger.info("=" * 80)

    # Load scene
    if cfg.scene_name == "etoile":
        scene = load_scene(sionna.rt.scene.etoile, merge_shapes=False)
    elif cfg.scene_name == "san_francisco":
        scene = load_scene(sionna.rt.scene.san_francisco, merge_shapes=False)
    elif cfg.scene_name == "munich":
        scene = load_scene(sionna.rt.scene.munich, merge_shapes=False)
    elif cfg.scene_name == "florence":
        scene = load_scene(sionna.rt.scene.florence, merge_shapes=False)
    else:
        raise ValueError(f"Unknown scene: {cfg.scene_name}")

    scene.frequency = cfg.frequency

    # Create transmitter config
    tx_config = hydra.utils.instantiate(cfg.transmitter)

    # Create output directory
    save_dir = "src/poc/density_analysis"
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Results will be saved to: {save_dir}")
    logger.info(f"Scene: {cfg.scene_name}")
    logger.info(f"Base config: {tx_config}")

    # Run tests
    test_samples_per_tx(scene, tx_config, save_dir)
    test_max_depth(scene, tx_config, save_dir)
    test_multi_transmitters(scene, tx_config, save_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETE!")
    logger.info(f"Results saved in: {save_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
