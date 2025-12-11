"""
Generate sample data using optimal configuration.
This creates a few samples with the best settings from density analysis.
"""

import logging

import hydra
import sionna
from omegaconf import DictConfig
from sionna.rt import load_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def generate_optimal_data(cfg: DictConfig) -> None:
    """Generate data with optimal configuration."""

    logger.info("=" * 80)
    logger.info("GENERATING DATA WITH OPTIMAL CONFIGURATION")
    logger.info("=" * 80)

    # Override config with optimal parameters
    cfg.transmitter.n_tx = 9  # Multiple transmitters - BIGGEST IMPACT
    cfg.generator.n_samples = 5  # Generate 5 samples for testing
    cfg.generator.dataset_path = "src/poc/optimal_data"

    logger.info("\nOptimal Configuration:")
    logger.info(f"  n_tx: {cfg.transmitter.n_tx}")
    logger.info("  samples_per_tx: 5,000,000 (will be set in generator)")
    logger.info("  max_depth: 7 (will be set in generator)")
    logger.info(f"  n_samples: {cfg.generator.n_samples}")
    logger.info(f"  scene: {cfg.scene_name}")

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

    # Create transmitter config with optimal n_tx
    tx_config = hydra.utils.instantiate(cfg.transmitter)

    # Create generator

    # We need to temporarily modify the generator to use optimal parameters
    # The cleanest way is to monkey-patch the _generate_sample method
    generator = hydra.utils.instantiate(cfg.generator, scene=scene)

    def optimal_generate_sample(sample_id: int, config: object, scene_corners: object):
        """Modified _generate_sample with optimal parameters."""
        import torch

        # Build transmitters
        tx_positions, grid_info = generator.builder.build(config, scene_corners, generator.tx_grid_info)
        tx_positions = torch.tensor(tx_positions)
        cx, cy = grid_info["center_x"], grid_info["center_y"]

        # Generate LOW RESOLUTION radio map with OPTIMAL PARAMETERS
        rm_lr = generator.rm_solver(
            generator.scene,
            max_depth=7,  # OPTIMAL: increased from 5
            samples_per_tx=5 * 10**6,  # OPTIMAL: increased from 1M
            cell_size=config.lr_cell_size,
            center=[cx, cy, 0.0],
            size=[config.coverage_size, config.coverage_size],
            orientation=[0, 0, 0],
        )

        # Generate HIGH RESOLUTION radio map with OPTIMAL PARAMETERS
        rm_hr = generator.rm_solver(
            generator.scene,
            max_depth=7,  # OPTIMAL: increased from 5
            samples_per_tx=5 * 10**6,  # OPTIMAL: increased from 1M
            cell_size=config.hr_cell_size,
            center=[cx, cy, 0.0],
            size=[config.coverage_size, config.coverage_size],
            orientation=[0, 0, 0],
        )

        # Extract the specified metric
        metric_lr = generator._extract_metric(rm_lr)
        metric_hr = generator._extract_metric(rm_hr)

        # Handle multiple transmitters by taking max value
        if metric_lr.dim() == 3:
            map_lr = torch.max(metric_lr, dim=0)[0]
        else:
            map_lr = metric_lr

        if metric_hr.dim() == 3:
            map_hr = torch.max(metric_hr, dim=0)[0]
        else:
            map_hr = metric_hr

        if generator.to_db:
            if generator.metric_type == "path_gain" or generator.metric_type == "sinr":
                map_lr = generator._apply_db_conversion(map_lr, generator.db_floor)
                map_hr = generator._apply_db_conversion(map_hr, generator.db_floor)
            elif generator.metric_type == "rss":
                map_lr = generator._apply_db_conversion(map_lr, generator.db_floor, dbm=True)
                map_hr = generator._apply_db_conversion(map_hr, generator.db_floor, dbm=True)

        # Create sample
        from poc.data_modules.generator import SuperResolutionDataSample

        sample = SuperResolutionDataSample(
            sample_id=sample_id,
            tx_positions=tx_positions,
            map_lr=map_lr,
            map_hr=map_hr,
            scale=config.scale,
            metric_type=generator.metric_type,
            grid_info=grid_info,
            config=config,
        )

        return sample

    # Replace the method
    generator._generate_sample = optimal_generate_sample

    # Generate dataset
    logger.info("\nStarting generation with optimal parameters...")
    logger.info("This will take longer than baseline but produce much denser data!")

    generator.generate_dataset(base_config=tx_config)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE!")
    logger.info(f"Data saved to: {cfg.generator.dataset_path}")
    logger.info("=" * 80)

    # Quick analysis
    import os

    import numpy as np
    import torch

    sample_files = sorted([f for f in os.listdir(cfg.generator.dataset_path) if f.endswith(".pt")])

    if sample_files:
        logger.info("\nQuick Analysis of Generated Data:")
        logger.info("-" * 80)

        # Analyze first sample
        sample_path = os.path.join(cfg.generator.dataset_path, sample_files[0])
        data = torch.load(sample_path, map_location="cpu")

        hr_map = data["map_hr"].cpu().numpy()
        lr_map = data["map_lr"].cpu().numpy()

        # Calculate sparsity (pixels <= -150 dB)
        hr_valid = np.sum(np.isfinite(hr_map) & (hr_map > -150.0))
        hr_total = hr_map.size
        hr_sparsity = 100 * (1 - hr_valid / hr_total)

        lr_valid = np.sum(np.isfinite(lr_map) & (lr_map > -150.0))
        lr_total = lr_map.size
        lr_sparsity = 100 * (1 - lr_valid / lr_total)

        logger.info(f"Sample: {sample_files[0]}")
        logger.info(f"  HR Map: {hr_map.shape}, Sparsity: {hr_sparsity:.2f}%, Valid: {hr_valid:,}/{hr_total:,}")
        logger.info(f"  LR Map: {lr_map.shape}, Sparsity: {lr_sparsity:.2f}%, Valid: {lr_valid:,}/{lr_total:,}")
        logger.info(f"  Scale: {data['scale']}")
        logger.info(f"  Transmitters: {data['tx_positions'].shape[0]}")
        logger.info("-" * 80)

        logger.info("\n✅ Expected sparsity with optimal config: ~25-40% (vs ~91% baseline)")


if __name__ == "__main__":
    generate_optimal_data()
