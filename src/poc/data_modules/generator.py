import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from sionna.rt import RadioMapSolver
from tqdm import tqdm

from poc.data_modules.builder import SceneTransmitterBuilder, TransmitterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sionna.rt.radio_map_solvers.planar_radio_map import PlanarRadioMap
    from sionna.rt.scene import Scene


@dataclass
class SuperResolutionDataSample:
    """Training data sample for super-resolution"""

    sample_id: int
    tx_positions: torch.Tensor  # Shape: (n_tx, 3)
    map_lr: torch.Tensor  # Low resolution radio map
    map_hr: torch.Tensor  # High resolution radio map
    scale: int  # Super-resolution scale factor
    metric_type: str  # Type of metric stored
    grid_info: dict
    config: TransmitterConfig


class RadioMapDataGenerator:
    """
    Generates radio map super-resolution data samples (low res & high res pairs).
    Uses SceneTransmitterBuilder to construct the provided scene with transmitters.
    Uses Sionna's RadioMapSolver to compute radio maps.
    """

    def __init__(self, scene: "Scene", metric_type: Literal["path_gain", "rss", "sinr"] = "path_gain"):
        self.scene = scene
        self.builder = SceneTransmitterBuilder(scene)
        self.rm_solver = RadioMapSolver()
        self.metric_type = metric_type

    def _extract_metric(self, radio_map: "PlanarRadioMap") -> torch.Tensor:
        """Extract the specified metric from the radio map"""
        if self.metric_type == "path_gain":
            return radio_map.path_gain.torch().cpu()
        elif self.metric_type == "rss":
            return radio_map.rss.torch().cpu()
        elif self.metric_type == "sinr":
            return radio_map.sinr.torch().cpu()
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def _generate_sample(self, sample_id: int, config: TransmitterConfig) -> SuperResolutionDataSample:
        """Generate a single super-resolution data sample"""

        # Build transmitters in scene
        tx_positions, grid_info = self.builder.build(config)
        tx_positions = torch.tensor(tx_positions)
        cx, cy = grid_info["center_x"], grid_info["center_y"]

        # Generate LOW RESOLUTION radio map
        rm_lr = self.rm_solver(
            self.scene,
            max_depth=5,
            samples_per_tx=10**6,
            cell_size=config.lr_cell_size,
            center=[cx, cy, 0.0],
            size=[config.coverage_size, config.coverage_size],
            orientation=[0, 0, 0],
        )

        rm_hr = self.rm_solver(
            self.scene,
            max_depth=5,
            samples_per_tx=10**6,
            cell_size=config.hr_cell_size,
            center=[cx, cy, 0.0],
            size=[config.coverage_size, config.coverage_size],
            orientation=[0, 0, 0],
        )

        # Extract the specified metric
        metric_lr = self._extract_metric(rm_lr)
        metric_hr = self._extract_metric(rm_hr)

        # Handle multiple transmitters by taking max value; FIXME: is this the way to do it in Sionna?
        if metric_lr.dim() == 3:
            map_lr = torch.max(metric_lr, dim=0)[0]
        else:
            map_lr = metric_lr

        if metric_hr.dim() == 3:
            map_hr = torch.max(metric_hr, dim=0)[0]
        else:
            map_hr = metric_hr

        # Create sample
        sample = SuperResolutionDataSample(
            sample_id=sample_id,
            tx_positions=tx_positions,
            map_lr=map_lr,
            map_hr=map_hr,
            scale=config.scale,
            metric_type=self.metric_type,
            grid_info=grid_info,
            config=config,
        )

        return sample

    def generate_dataset(
        self,
        n_samples: int,
        save_dir: str,
        base_config: TransmitterConfig,
        naming_convention: str = "sample_{:04d}.pt",
        show_progress: bool = True,
    ) -> None:
        """Generate multiple super-resolution data samples"""

        logging.info(f"Generating {n_samples} super-resolution samples...")
        logging.info(f"Metric: {self.metric_type}, Scale: {base_config.scale}x")
        logging.info(f"LR cell: {base_config.lr_cell_size}, HR cell: {base_config.hr_cell_size}")
        logging.info(f"Coverage: {base_config.coverage_size}x{base_config.coverage_size}m")

        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Samples will be saved to: {save_dir}")

        iterator = tqdm(range(n_samples), desc="Generating samples") if show_progress else range(n_samples)

        for i in iterator:
            # Create config with unique seed for each sample
            from dataclasses import replace

            config = replace(base_config, seed=i + 200)

            sample = self._generate_sample(i + 1, config)
            self._save_data(sample, save_dir, naming=naming_convention)

            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "LR": str(tuple(sample.map_lr.shape)),
                        "HR": str(tuple(sample.map_hr.shape)),
                    }
                )

        logger.info(f"Dataset generation complete: {n_samples} samples saved to {save_dir}")

    @staticmethod
    def _save_data(sample: SuperResolutionDataSample, save_dir: str, naming: str) -> None:
        """Save a single data sample to disk"""

        sample_data = {
            "sample_id": sample.sample_id,
            "tx_positions": sample.tx_positions,
            "map_lr": sample.map_lr.cpu(),
            "map_hr": sample.map_hr.cpu(),
            "scale": sample.scale,
            "metric_type": sample.metric_type,
            "grid_info": sample.grid_info,
            "coverage_size": sample.config.coverage_size,
        }

        sample_path = os.path.join(save_dir, f"{naming}_{sample.sample_id:04d}.pt")
        torch.save(sample_data, sample_path)
