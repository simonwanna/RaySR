"""
Test the optimal configuration based on density analysis results.
This validates that combining optimal parameters gives the expected sparsity reduction.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sionna.rt import load_scene

from poc.data_modules.builder import SceneTransmitterBuilder, TransmitterConfig
from poc.data_modules.generator import RadioMapDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_sparsity(radio_map: torch.Tensor, name: str, db_floor: float = -150.0) -> object:
    """Quick sparsity analysis."""
    data = radio_map.squeeze().cpu().numpy()

    valid_pixels = np.sum(np.isfinite(data) & (data > db_floor))
    total = data.size
    sparsity = 100 * (1 - valid_pixels / total)

    valid_data = data[np.isfinite(data) & (data > db_floor)]

    return {
        "name": name,
        "sparsity_percent": sparsity,
        "valid_pixels": valid_pixels,
        "total_pixels": total,
        "valid_mean_db": float(np.mean(valid_data)) if len(valid_data) > 0 else float("nan"),
        "valid_std_db": float(np.std(valid_data)) if len(valid_data) > 0 else float("nan"),
    }


def main() -> None:
    logger.info("=" * 80)
    logger.info("TESTING OPTIMAL CONFIGURATION")
    logger.info("=" * 80)

    # Load transmitter config
    tx_cfg_path = Path(__file__).parent / "configs" / "data" / "transmitter.yaml"
    if not tx_cfg_path.exists():
        logger.error(f"Config not found: {tx_cfg_path}")
        logger.info("Please run this from the src/poc directory or check your config paths")
        sys.exit(1)

    tx_cfg = OmegaConf.load(tx_cfg_path)

    # Load scene (use default from analyze_density.py)
    scene_name = "etoile"  # Default scene
    logger.info(f"\nLoading scene: {scene_name}")
    scene = load_scene(f"sionna://{scene_name}")
    scene.frequency = 3.5e9  # Default frequency

    # Configuration comparison
    configs_to_test = [
        {
            "name": "BASELINE (Current)",
            "n_tx": 1,
            "samples_per_tx": 10**6,
            "max_depth": 5,
        },
        {
            "name": "OPTIMAL (Recommended)",
            "n_tx": 9,
            "samples_per_tx": 5 * 10**6,
            "max_depth": 7,
        },
        {
            "name": "BALANCED (Fast)",
            "n_tx": 4,
            "samples_per_tx": 2 * 10**6,
            "max_depth": 7,
        },
    ]

    results = []

    for config in configs_to_test:
        logger.info("\n" + "-" * 80)
        logger.info(f"Testing: {config['name']}")
        logger.info(f"  n_tx: {config['n_tx']}")
        logger.info(f"  samples_per_tx: {config['samples_per_tx']:,}")
        logger.info(f"  max_depth: {config['max_depth']}")
        logger.info("-" * 80)

        # Create transmitter config
        tx_config = TransmitterConfig(
            n_tx=config["n_tx"],
            scale=tx_cfg.scale,
            coverage_size=tx_cfg.coverage_size,
            hr_grid_size=tx_cfg.hr_grid_size,
            grid_randomization=tx_cfg.grid_randomization,
            margin=tx_cfg.margin,
            tx_power_dbm=tx_cfg.tx_power_dbm,
            tx_array_pattern=tx_cfg.tx_array_pattern,
            polarization=tx_cfg.polarization,
            tx_default_height=tx_cfg.tx_default_height,
            tx_height_margin=tx_cfg.tx_height_margin,
        )

        # Build transmitters
        builder = SceneTransmitterBuilder(scene, tx_config)
        builder.build()

        # Create generator
        generator = RadioMapDataGenerator(
            metric_type="path_gain",
            n_samples=1,
            dataset_path="temp_optimal_test",
            to_db=True,
            db_floor=-150.0,
            scene=scene,
        )

        # Generate one sample with custom parameters
        logger.info("Generating sample...")
        import time

        start_time = time.time()

        # Get scene boundaries
        scene_corners = generator._get_scene_boundary(tx_config.margin)

        # Build transmitters
        tx_positions, grid_info = builder.build(tx_config, scene_corners, generator.tx_grid_info)
        cx, cy = grid_info["center_x"], grid_info["center_y"]

        # Generate HR map with custom parameters
        rm_hr = generator.rm_solver(
            scene,
            max_depth=config["max_depth"],
            samples_per_tx=config["samples_per_tx"],
            cell_size=tx_config.hr_cell_size,
            center=[cx, cy, 0.0],
            size=[tx_config.coverage_size, tx_config.coverage_size],
            orientation=[0, 0, 0],
        )

        # Extract metric
        metric_hr = generator._extract_metric(rm_hr)
        if metric_hr.dim() == 3:
            map_hr = torch.max(metric_hr, dim=0)[0]
        else:
            map_hr = metric_hr

        # Apply dB conversion
        map_hr = generator._apply_db_conversion(map_hr, -150.0)

        # Generate LR map
        rm_lr = generator.rm_solver(
            scene,
            max_depth=config["max_depth"],
            samples_per_tx=config["samples_per_tx"],
            cell_size=tx_config.lr_cell_size,
            center=[cx, cy, 0.0],
            size=[tx_config.coverage_size, tx_config.coverage_size],
            orientation=[0, 0, 0],
        )

        metric_lr = generator._extract_metric(rm_lr)
        if metric_lr.dim() == 3:
            map_lr = torch.max(metric_lr, dim=0)[0]
        else:
            map_lr = metric_lr

        map_lr = generator._apply_db_conversion(map_lr, -150.0)

        generation_time = time.time() - start_time

        # Analyze
        hr_stats = analyze_sparsity(map_hr, f"{config['name']} HR")
        lr_stats = analyze_sparsity(map_lr, f"{config['name']} LR")

        logger.info(f"\n✓ Generation completed in {generation_time:.1f}s")
        logger.info("\nHR Map Results:")
        logger.info(f"  Sparsity: {hr_stats['sparsity_percent']:.2f}%")
        logger.info(f"  Valid pixels: {hr_stats['valid_pixels']:,} / {hr_stats['total_pixels']:,}")
        logger.info(f"  Mean signal: {hr_stats['valid_mean_db']:.2f} dB")
        logger.info(f"  Std: {hr_stats['valid_std_db']:.2f} dB")

        logger.info("\nLR Map Results:")
        logger.info(f"  Sparsity: {lr_stats['sparsity_percent']:.2f}%")
        logger.info(f"  Valid pixels: {lr_stats['valid_pixels']:,} / {lr_stats['total_pixels']:,}")

        results.append(
            {
                "config": config,
                "hr": hr_stats,
                "lr": lr_stats,
                "generation_time": generation_time,
            }
        )

    # Print comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\n{'Configuration':<25} {'HR Sparsity':<15} {'Valid Pixels':<15} {'Time (s)':<10}")
    logger.info("-" * 80)

    for result in results:
        logger.info(
            f"{result['config']['name']:<25} "
            f"{result['hr']['sparsity_percent']:>6.2f}%        "
            f"{result['hr']['valid_pixels']:>8,}        "
            f"{result['generation_time']:>6.1f}"
        )

    logger.info("-" * 80)

    # Calculate improvements
    baseline_sparsity = results[0]["hr"]["sparsity_percent"]
    optimal_sparsity = results[1]["hr"]["sparsity_percent"]
    balanced_sparsity = results[2]["hr"]["sparsity_percent"]

    logger.info("\n🎯 IMPROVEMENTS vs BASELINE:")
    logger.info(
        f"  OPTIMAL:  {baseline_sparsity:.1f}% → {optimal_sparsity:.1f}% "
        f"({baseline_sparsity - optimal_sparsity:.1f}% reduction)"
    )
    logger.info(
        f"  BALANCED: {baseline_sparsity:.1f}% → {balanced_sparsity:.1f}% "
        f"({baseline_sparsity - balanced_sparsity:.1f}% reduction)"
    )

    logger.info("\n⏱️  SPEED TRADE-OFF:")
    baseline_time = results[0]["generation_time"]
    optimal_time = results[1]["generation_time"]
    balanced_time = results[2]["generation_time"]

    logger.info(f"  OPTIMAL:  {optimal_time / baseline_time:.1f}x slower than baseline")
    logger.info(f"  BALANCED: {balanced_time / baseline_time:.1f}x slower than baseline")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
