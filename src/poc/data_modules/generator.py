import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mitsuba as mi
import numpy as np
import torch
from scipy import ndimage as ndi
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
    height_map: torch.Tensor  # Height map of the scene
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
        step_length: float = 0.1,
    ) -> None:
        self.metric_type = metric_type
        self.n_samples = n_samples
        self.dataset_path = dataset_path
        self.naming_convention = naming_convention
        self.to_db = to_db
        self.db_floor = db_floor
        self.min_object_height = min_object_height
        self.step_length = step_length
        self.tx_grid_info = None
        self._setup(scene)
        self._generate_tx_grid_info()

    def _setup(self, scene: "Scene") -> None:
        """Setup the data generator with the given scene"""
        self.scene = scene
        self.builder = SceneTransmitterBuilder(scene)
        self.rm_solver = RadioMapSolver()

    def _get_scene_disc_info(self) -> dict:
        """Get scene discretization info for height map generation for valid transmitter placement"""
        # Get scene bounding box
        mi_scene = self.scene.mi_scene
        bbox = mi_scene.bbox()

        # Calculate grid step size
        h = self.step_length
        xmin, xmax_raw = float(bbox.min.x), float(bbox.max.x)
        ymin, ymax_raw = float(bbox.min.y), float(bbox.max.y)

        # Calculate number of grid points (including both endpoints)
        nx = int(np.floor((xmax_raw - xmin) / h)) + 1
        ny = int(np.floor((ymax_raw - ymin) / h)) + 1

        # Snap xmax and ymax to grid
        xmax = xmin + (nx - 1) * h
        ymax = ymin + (ny - 1) * h

        scene_disc_info = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "nx": nx,
            "ny": ny,
            "h": h,
        }

        return scene_disc_info
    
    @staticmethod
    def _get_sample_disc_info(grid_info: dict, config: TransmitterConfig) -> dict:
        """Get sample discretization info for height map extraction"""
        # Retrieve grid parameters
        map_bounds = grid_info["map_bounds"]
        xmin, xmax = map_bounds[0]
        ymin, ymax = map_bounds[1]

        # Calculate number of grid points (including both endpoints)
        nx = int(config.hr_grid_size / config.scale)
        ny = int(config.hr_grid_size / config.scale)

        sample_disc_info = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "nx": nx,
            "ny": ny,
            }

        return sample_disc_info
    
    def _height_map_ray_casting(self, disc_info: dict, direction: tuple[float, float, float]) -> np.ndarray:
        """Generate height map using ray casting method"""
        # Get grid information
        xmin = disc_info["xmin"]
        xmax = disc_info["xmax"]
        ymin = disc_info["ymin"]
        ymax = disc_info["ymax"]
        nx = disc_info["nx"]
        ny = disc_info["ny"]

        # Generate ray origins on a grid above the scene
        x_vals = np.linspace(xmin, xmax, nx)
        y_vals = np.linspace(ymin, ymax, ny)
        nx, ny = x_vals.shape[0], y_vals.shape[0]

        # Access mitsuba scene
        mi_scene = self.scene.mi_scene

        # Get ray origin height
        ray_origin_height = float(mi_scene.bbox().max.z) + 1.0  # some arbitrary margin above scene

        # Create meshgrid for ray origins
        X, Y = np.meshgrid(x_vals, y_vals)
        Xf = X.ravel()
        Yf = Y.ravel()
        Zf = np.full(X.size, -1.0 * direction[2] * ray_origin_height)   # set z coordinate based on ray direction:
                                                                        # if direction is downward (0,0,-1), z = +ray_origin_height
                                                                        # if direction is upward (0,0,1), z = -ray_origin_height

        # Cast rays and get intersection points
        ray = mi.Ray3f(o=mi.Point3f(Xf, Yf, Zf), d=mi.Vector3f(direction))
        intersect = mi_scene.ray_intersect(ray)

        # Convert to numpy array and invalidate non-hit points
        hits = np.array(intersect.p.z, dtype=float)
        valid_mask = np.array(intersect.is_valid(), dtype=bool)
        hits[~valid_mask] = np.nan

        # Reshape results into height map
        height_map = hits.reshape(ny, nx)

        return height_map
    
    @staticmethod
    def _generate_nearest_neighbor_indexes(height_map: np.ndarray) -> np.ndarray:
        """Generate nearest valid neighbor indexes for NaN values in height map"""
        valid = ~np.isnan(height_map)
        nearest_idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        nearest_idx = np.array(nearest_idx)
        return nearest_idx
    
    def _generate_scene_height_map(self, scene_disc_info: dict) -> np.ndarray:
        """Generate height map of the scene using ray casting"""
        # Ray casting in downward direction for object z coordinates
        scene_height_map = self._height_map_ray_casting(scene_disc_info, direction=(0.0, 0.0, -1.0))
        
        # Ray cast in upward direction to determine ground z coordinates
        scene_ground_height_map = self._height_map_ray_casting(scene_disc_info, direction=(0.0, 0.0, 1.0))

        # Relative object heights
        height_above_ground = scene_height_map - scene_ground_height_map

        # Invalidate heights below minimum object height
        scene_height_map[height_above_ground < self.min_object_height] = np.nan

        return scene_height_map
    
    def _generate_sample_height_map(self, grid_info: dict, config: TransmitterConfig) -> np.ndarray:
        """Extract height map subset for the current sample based on coverage area"""
        # Get sample discretization info
        sample_disc_info = self._get_sample_disc_info(grid_info, config)

        # Generate sample height map
        sample_height_map = self._height_map_ray_casting(sample_disc_info, direction=(0.0, 0.0, -1.0))

        return sample_height_map

    def _generate_tx_grid_info(self) -> None:
        """Generate transmitter grid info based on scene geometry"""
        # Get scene discretization info
        scene_disc_info = self._get_scene_disc_info()

        # Generate height map and discretization info
        scene_height_map = self._generate_scene_height_map(scene_disc_info)

        # Generate "nearest valid neighbor" indexes for NaN values
        nearest_idx = self._generate_nearest_neighbor_indexes(scene_height_map)

        # Store transmitter grid info
        tx_grid_info = {
            "xmin": scene_disc_info["xmin"],    # minimum x coordinate
            "xmax": scene_disc_info["xmax"],    # maximum x coordinate
            "ymin": scene_disc_info["ymin"],    # minimum y coordinate
            "ymax": scene_disc_info["ymax"],    # maximum y coordinate
            "nx": scene_disc_info["nx"],        # number of points in x direction
            "ny": scene_disc_info["ny"],        # number of points in y direction
            "h": scene_disc_info["h"],          # grid step size
            "height_map": scene_height_map,     # height map matrix
            "nearest_idx": nearest_idx,         # nearest valid neighbor indexes
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

        # Build transmitters in scene
        tx_positions, grid_info = self.builder.build(config, scene_corners, self.tx_grid_info)
        tx_positions = torch.tensor(tx_positions)
        cx, cy = grid_info["center_x"], grid_info["center_y"]

        height_map = self._generate_sample_height_map(grid_info, config)
        height_map = torch.tensor(height_map)

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
            height_map=height_map,
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

        # TODO: add extra channel for height map
        sample_data = {
            "sample_id": sample.sample_id,
            "tx_positions": sample.tx_positions,
            "height_map": sample.height_map.cpu(),
            "map_lr": sample.map_lr.cpu(),
            "map_hr": sample.map_hr.cpu(),
            "scale": sample.scale,
            "metric_type": sample.metric_type,
            "grid_info": sample.grid_info,
            "coverage_size": sample.config.coverage_size,
        }

        sample_path = os.path.join(save_dir, f"{naming}_{sample.sample_id:04d}.pt")
        torch.save(sample_data, sample_path)
