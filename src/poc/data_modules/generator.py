import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import sionna
from sionna.rt import RadioMapSolver
from scipy import ndimage as ndi
import numpy as np
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
        min_object_height: float = 10.0,
        step_size_power: float = 10.0,
        step_size_exponent: float = -1.0
    ) -> None:
        self.metric_type = metric_type
        self.n_samples = n_samples
        self.dataset_path = dataset_path
        self.naming_convention = naming_convention
        self.to_db = to_db
        self.db_floor = db_floor
        self.min_object_height = min_object_height
        self.step_size_power = step_size_power
        self.step_size_exponent = step_size_exponent
        self.tx_grid_info = None
        self._generate_tx_grid_info()
        self._setup(scene)

    def _setup(self, scene: "Scene") -> None:
        """Setup the data generator with the given scene"""
        self.scene = scene
        self.builder = SceneTransmitterBuilder(scene)
        self.rm_solver = RadioMapSolver()

    def _generate_tx_grid_info(self) -> None:
        """Generate transmitter grid info based on scene geometry"""

        # Obtain scene bounding box
        scene_bbox = self.scene.mi_scene.bbox()
        xmin, xmax = float(scene_bbox.min.x), float(scene_bbox.max.x)
        ymin, ymax = float(scene_bbox.min.y), float(scene_bbox.max.y)

        # Determine grid resolution
        step_size = self.step_size_power ** self.step_size_exponent
        num_x_points = int(np.ceil((xmax - xmin) / step_size)) + 1
        num_y_points = int(np.ceil((ymax - ymin) / step_size)) + 1

        height_matrix = np.full((num_y_points, num_x_points), np.nan, dtype=float)

        # Determine valid objects based on min height
        valid_object_bboxes = []
        for obj in tqdm(self.scene.objects.values(), desc="Extracting valid scene objects"):
            if getattr(obj, "mi_mesh", None) and obj.name not in ['ground', 'Terrain', 'Plane', 'floor']:
                if obj.mi_mesh.bbox().extents()[2] >= self.min_object_height:
                    valid_object_bboxes.append(obj.mi_mesh.bbox())

        if len(valid_object_bboxes) == 0:
            logger.warning("Scene contains no valid objects.")
            return

        # Populate height matrix
        for bb in tqdm(valid_object_bboxes, desc="Generating height map"):
            col_idx_min = int(np.ceil((float(bb.min.x) - xmin) / step_size))
            col_idx_max = int(np.floor((float(bb.max.x) - xmin) / step_size))
            row_idx_min = int(np.ceil((float(bb.min.y) - ymin) / step_size))
            row_idx_max = int(np.floor((float(bb.max.y) - ymin) / step_size))

            col_idx_min = max(col_idx_min, 0)
            row_idx_min = max(row_idx_min, 0)
            col_idx_max = min(col_idx_max, num_x_points-1)
            row_idx_max = min(row_idx_max, num_y_points-1)

            if col_idx_min > col_idx_max or row_idx_min > row_idx_max: 
                continue

            block = height_matrix[row_idx_min:(row_idx_max + 1), col_idx_min:(col_idx_max + 1)]
            np.fmax(block, float(bb.max.z), out=block)

        # Generate "nearest valid neighbor" indexes for NaN values
        valid = ~np.isnan(height_matrix)
        nearest_idx = ndi.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )

        tx_grid_info = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "num_x_points": num_x_points,
            "num_y_points": num_y_points,
            "step_size": step_size,
            "height_matrix": height_matrix,
            "nearest_idx": nearest_idx,
        }

        self.tx_grid_info = tx_grid_info

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
    def _apply_db_conversion(radio_map: torch.Tensor, floor_db: float = -150.0, dbm: bool = False) -> torch.Tensor:
        """Apply dB conversion to the radio map"""
        radio_map_db = 10.0 * torch.log10(torch.clamp(radio_map, min=1e-15))
        if dbm:
            radio_map_db += 30.0  # Convert to dBm
        if floor_db is not None:
            radio_map_db = torch.clamp(radio_map_db, min=floor_db)
        return radio_map_db

    def _generate_sample(
        self, sample_id: int, config: TransmitterConfig, scene_corners: tuple
    ) -> SuperResolutionDataSample:
        """Generate a single super-resolution data sample"""

        #TODO: use min_tx_height and more...

        # Build transmitters in scene
        tx_positions, grid_info = self.builder.build(config, scene_corners, self.tx_grid_info)
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
            elif self.metric_type == "rss":
                map_lr = self._apply_db_conversion(map_lr, self.db_floor, dbm=True)
                map_hr = self._apply_db_conversion(map_hr, self.db_floor, dbm=True)

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

        # Get boundaries of the scene
        scene_corners = self._get_scene_boundary(base_config.margin)

        iterator = tqdm(range(self.n_samples), desc="Generating samples") if show_progress else range(self.n_samples)

        for i in iterator:
            # Create config with unique seed for each sample
            from dataclasses import replace

            config = replace(base_config, seed=i + 200)

            sample = self._generate_sample(i + 1, config, scene_corners)
            self._save_data(sample, self.dataset_path, naming=self.naming_convention)

            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "LR": str(tuple(sample.map_lr.shape)),
                        "HR": str(tuple(sample.map_hr.shape)),
                    }
                )

        logger.info(f"Dataset generation complete: {self.n_samples} samples saved to {self.dataset_path}")

    def _get_scene_boundary(self, margin: float) -> tuple:
        """Get the boundary of the scene for transmitter placement"""
        bbox = self.scene.mi_scene.bbox()
        x_min = bbox.min.x
        x_max = bbox.max.x
        y_min = bbox.min.y
        y_max = bbox.max.y

        # add margin
        x_min += margin
        x_max -= margin
        y_min += margin
        y_max -= margin

        return ((x_min, x_max), (y_min, y_max))

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
