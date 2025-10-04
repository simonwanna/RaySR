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

    def __init__(
        self,
        metric_type: Literal["path_gain", "rss", "sinr"] = "path_gain",
        n_samples: int = 100,
        dataset_path: str = "",
        naming_convention: str = "sample_{:04d}.pt",
        to_db: bool = True,
        db_floor: float = -150.0,
        scene: "Scene" = None,
    ) -> None:
        self.metric_type = metric_type
        self.n_samples = n_samples
        self.dataset_path = dataset_path
        self.naming_convention = naming_convention
        self.to_db = to_db
        self.db_floor = db_floor
        self._setup(scene)

    def _setup(self, scene: "Scene") -> None:
        """Setup the data generator with the given scene"""
        self.scene = scene
        self.builder = SceneTransmitterBuilder(scene)
        self.rm_solver = RadioMapSolver()

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

    @staticmethod
    def _apply_db_conversion(radio_map: torch.Tensor, floor_db: float = -150.0) -> torch.Tensor:
        """Apply dB conversion to the radio map"""
        radio_map_db = 10.0 * torch.log10(torch.clamp(radio_map, min=1e-15))
        if floor_db is not None:
            radio_map_db = torch.clamp(radio_map_db, min=floor_db)
        return radio_map_db

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

        if self.to_db:
            if self.metric_type == "path_gain" or self.metric_type == "sinr":
                map_lr = self._apply_db_conversion(map_lr, self.db_floor)
                map_hr = self._apply_db_conversion(map_hr, self.db_floor)
            elif self.metric_type == "rss":  # TODO: implement dB conversion for RSS (dBm)
                raise NotImplementedError("dB conversion for RSS not implemented yet")

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
        base_config: TransmitterConfig,
        show_progress: bool = True,
    ) -> None:
        """Generate multiple super-resolution data samples"""

        logging.info(f"Generating {self.n_samples} super-resolution samples...")
        logging.info(f"Metric: {self.metric_type}, Scale: {base_config.scale}x")
        logging.info(f"LR cell: {base_config.lr_cell_size}, HR cell: {base_config.hr_cell_size}")
        logging.info(f"Coverage: {base_config.coverage_size}x{base_config.coverage_size}m")

        os.makedirs(self.dataset_path, exist_ok=True)
        logging.info(f"Samples will be saved to: {self.dataset_path}")

        iterator = tqdm(range(self.n_samples), desc="Generating samples") if show_progress else range(self.n_samples)

        for i in iterator:
            # Create config with unique seed for each sample
            from dataclasses import replace

            config = replace(base_config, seed=i + 200)

            sample = self._generate_sample(i + 1, config)
            self._save_data(sample, self.dataset_path, naming=self.naming_convention)

            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "LR": str(tuple(sample.map_lr.shape)),
                        "HR": str(tuple(sample.map_hr.shape)),
                    }
                )

        logger.info(f"Dataset generation complete: {self.n_samples} samples saved to {self.dataset_path}")

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
