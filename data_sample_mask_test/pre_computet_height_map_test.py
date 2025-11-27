from matplotlib.pylab import sample
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'

import scipy.ndimage as ndi
from sionna.rt import load_scene, Transmitter, RadioMapSolver, PlanarArray, RadioMap, Scene, Camera
import sionna
import mitsuba as mi

from typing import Tuple, List, cast
from pathlib import Path
import os
from tqdm import tqdm

from dataclasses import dataclass
import time
import argparse

RM_SOLVER = RadioMapSolver()


@dataclass(frozen=True)
class TransmitterConfig:
    hr_grid_size: int = 512
    scale: int = 2
    tx_array_pattern: str = "iso"
    polarization: str = "V"
    coverage_size: float = 160.0

    hr_size = coverage_size / hr_grid_size
    lr_size = hr_size * scale

    hr_cell_size: tuple[float, float] = (hr_size, hr_size)
    lr_cell_size: tuple[float, float] = (lr_size, lr_size)
    
    grid_randomization: float = 1.0
    scene_grid_margin: float = 30.0
    n_tx: int = 1
    tx_default_height: float = 45.0
    tx_height_margin: float = 5.0
    tx_power_dbm: int = 44

    seed: int = 42
    metric_type: str = "path_gain"
    to_db: bool = True
    db_floor: float = -150.0
    db_ceiling: float = -50.0

    n_samples: int = 5
    min_object_height: float = 10.0
    step_length: float = 0.1

    building_mask_threshold: float = 0.0
    los_mask_threshold: float = 0.0

    def validate(self) -> "TransmitterConfig":
        if self.n_tx <= 0:
            raise ValueError("n_tx must be > 0.")
        if self.scale <= 1:
            raise ValueError("scale must be > 1.")
        if self.coverage_size <= 0:
            raise ValueError("coverage_size must be > 0.")
        if not (0.0 <= self.grid_randomization <= 1.0):
            raise ValueError("grid_randomization must be between 0.0 and 1.0.")
        return self


@dataclass
class SuperResolutionDataSample:
    """Training data sample for super-resolution"""

    sample_id: int
    tx_positions: np.ndarray    # Shape: (n_tx, 3)
    height_map_from_ray: np.ndarray | None  # Height map generated via ray casting
    height_map_from_map: np.ndarray  # Height map generated via grid info
    map_lr: np.ndarray          # Low resolution radio map
    map_hr: np.ndarray          # High resolution radio map
    scale: int                  # Super-resolution scale factor
    metric_type: str            # Type of metric stored
    grid_info: dict             # Grid info for the sample
    config: TransmitterConfig   # Transmitter configuration
    time_hmr: float       # Time taken for height map generation via ray casting
    time_hmm: float       # Time taken for height map generation via map info
    time_lr: float       # Time taken for low-resolution map generation
    time_hr: float       # Time taken for high-resolution map generation
    rendered_image: np.ndarray  # Rendered image of the radio map


def plot_height_map(
    Z: np.ndarray | None, 
    meta: dict[str, float],
    txs: dict[str, dict[str, float | str]] | None = None,
    cmap: str = 'viridis',
    title: str = 'Ray-casted height map',
    xlabel: str = 'x (m)',
    ylabel: str = 'y (m)',
    colorbar_label: str = 'Height (m)',
    show_title: bool = True,
    show_axis: bool = True,
    show_colorbar: bool = True,
    ) -> None:
    """Plot top-down view of the height map."""
    if Z is None:
        raise ValueError("Height map Z is None.")
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
            Z,
            origin="lower",
            cmap=cmap,
            extent=(float(meta["xmin"]), float(meta["xmax"]), float(meta["ymin"]), float(meta["ymax"]))
    )

    if txs is not None:
        for label, data in txs.items():
            x = data["x"]
            y = data["y"]
            color = data.get("color", None)
            plt.scatter(x, y, label=label, s=35, edgecolors="black", linewidths=0.5, color=color)
        plt.legend()

    if show_colorbar:
        plt.colorbar(im, label=colorbar_label)

    if show_axis:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.axis("off")

    if show_title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def get_scene_disc_info(scene: "Scene", config: TransmitterConfig) -> dict:
    """Get scene discretization info for height map generation for valid transmitter placement"""
    # Get scene bounding box
    mi_scene = scene.mi_scene
    bbox = mi_scene.bbox()  # type: ignore

    # Calculate grid step size
    h = config.step_length
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

def get_sample_disc_info(grid_info: dict, config: TransmitterConfig) -> dict:
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

def height_map_ray_casting(scene: "Scene", disc_info: dict, direction: tuple[float, float, float]) -> np.ndarray:
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

    # Access mitsuba scene
    mi_scene = scene.mi_scene

    # Get ray origin height
    ray_origin_height = float(mi_scene.bbox().max.z)    # type: ignore

    # Create meshgrid for ray origins
    X, Y = np.meshgrid(x_vals, y_vals)
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = np.full(X.size, -1.0 * direction[2] * ray_origin_height)   # set z coordinate based on ray direction:
                                                                    # if direction is downward (0,0,-1), z = +ray_origin_height
                                                                    # if direction is upward (0,0,1), z = -ray_origin_height

    # Cast rays and get intersection points
    ray = mi.Ray3f(o=mi.Point3f(Xf, Yf, Zf), d=mi.Vector3f(direction))
    intersect = mi_scene.ray_intersect(ray) # type: ignore

    # Convert to numpy array and invalidate non-hit points
    hits = np.array(intersect.p.z, dtype=float)
    valid_mask = np.array(intersect.is_valid(), dtype=bool)
    hits[~valid_mask] = np.nan

    # Reshape results into height map
    height_map = hits.reshape(ny, nx)

    return height_map

def tx_to_ground_ray_casting(scene: "Scene", ground_height_map: np.ndarray, disc_info: dict) -> np.ndarray:
    """Generate LoS mask using ray casting method"""
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

    # Access mitsuba scene
    mi_scene = scene.mi_scene

    # Retrieve scene transmitter
    tx = next(iter(scene.transmitters.values()))

    # Create meshgrid for receiver positions
    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = ground_height_map.ravel()

    # Create ray origins at transmitter position
    ray_origins = mi.Point3f(
        np.full_like(X.size, tx.position.x[0]),
        np.full_like(X.size, tx.position.y[0]),
        np.full_like(X.size, tx.position.z[0]),
    )

    # Create ray directions towards each grid point
    dx = Xf - tx.position.x[0]
    dy = Yf - tx.position.y[0]
    dz = Zf - tx.position.z[0]
    L  = np.sqrt(dx * dx + dy * dy + dz * dz)

    ray_directions = mi.Vector3f(
        dx / L,
        dy / L,
        dz / L
    )

    # Cast rays and get intersection points
    ray = mi.Ray3f(o=ray_origins, d=ray_directions)
    intersect = mi_scene.ray_intersect(ray) # type: ignore

    # Convert to numpy array and invalidate non-hit points
    height_hits = np.array(intersect.p.z, dtype=float)
    valid_mask = np.array(intersect.is_valid(), dtype=bool)
    height_hits[~valid_mask] = np.nan

    # Reshape results into height map
    height_hits_map = height_hits.reshape(ny, nx)

    return height_hits_map

@staticmethod
def generate_nearest_neighbor_indexes(height_map: np.ndarray) -> np.ndarray:
    """Generate nearest valid neighbor indexes for NaN values in height map"""
    valid = ~np.isnan(height_map)
    nearest_idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest_idx = np.array(nearest_idx)
    return nearest_idx

def generate_scene_height_map(scene: "Scene", config: TransmitterConfig, scene_disc_info: dict) -> np.ndarray:
    """Generate height map of the scene using ray casting"""
    # Ray casting in downward direction for object z coordinates
    scene_height_map = height_map_ray_casting(scene, scene_disc_info, direction=(0.0, 0.0, -1.0))
    
    # Ray cast in upward direction to determine ground z coordinates
    scene_ground_height_map = height_map_ray_casting(scene, scene_disc_info, direction=(0.0, 0.0, 1.0))

    # Relative object heights
    height_above_ground = scene_height_map - scene_ground_height_map

    # Invalidate heights below minimum object height
    scene_height_map[height_above_ground < config.min_object_height] = np.nan

    return scene_height_map

def generate_sample_height_map(scene: "Scene", grid_info: dict, config: TransmitterConfig) -> np.ndarray:
    """Extract height map subset for the current sample based on coverage area"""
    # Get sample discretization info
    sample_disc_info = get_sample_disc_info(grid_info, config)

    # Generate sample height map
    sample_height_map = height_map_ray_casting(scene, sample_disc_info, direction=(0.0, 0.0, -1.0))

    return sample_height_map

def generate_sample_building_mask(
        scene: "Scene",
        height_map: np.ndarray,
        grid_info: dict,
        config: TransmitterConfig
    ) -> tuple[np.ndarray, np.ndarray]:
    """Generate building mask and ground height map for the sample"""
    # Get sample discretization info
    sample_disc_info = get_sample_disc_info(grid_info, config)

    # Generate ground height map
    ground_height_map = height_map_ray_casting(scene, sample_disc_info, direction=(0.0, 0.0, 1.0))

    # Calculate relative heights
    height_above_ground = height_map - ground_height_map

    # Generate building mask
    building_mask = np.where(height_above_ground > config.building_mask_threshold, 1, 0)

    return building_mask, ground_height_map

def generate_sample_los_mask(
        scene: "Scene",
        ground_height_map: np.ndarray,
        grid_info: dict,
        config: TransmitterConfig
    ) -> np.ndarray:
    """Generate line-of-sight (LoS) mask for the sample"""
    # Get sample discretization info
    sample_disc_info = get_sample_disc_info(grid_info, config)

    # Perform ray casting from transmitter to ground grid points
    tx_hits_height_map = tx_to_ground_ray_casting(scene, ground_height_map, sample_disc_info)
    
    # Determine LoS by comparing hit heights with ground heights inside threshold
    los_mask = np.where(tx_hits_height_map <= ground_height_map + config.los_mask_threshold, 1, 0)

    return los_mask

def generate_tx_grid_info(scene: "Scene", config: TransmitterConfig) -> dict:
    """Generate transmitter grid info based on scene geometry"""
    # Get scene discretization info
    scene_disc_info = get_scene_disc_info(scene, config)

    # Generate height map and discretization info
    scene_height_map = generate_scene_height_map(scene, config, scene_disc_info)

    # Generate "nearest valid neighbor" indexes for NaN values
    nearest_idx = generate_nearest_neighbor_indexes(scene_height_map)

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

    return tx_grid_info

def generate_global_grid_info(scene: "Scene", config: TransmitterConfig) -> dict:
    scene_disc_info = get_scene_disc_info(scene, config)
    xmin, xmax_raw = scene_disc_info["xmin"], scene_disc_info["xmax"]
    ymin, ymax_raw = scene_disc_info["ymin"], scene_disc_info["ymax"]
    
    n_grid = int(config.hr_grid_size / config.scale)
    h = config.coverage_size / (n_grid - 1)

    nx = int(np.floor((xmax_raw - xmin) / h)) + 1
    ny = int(np.floor((ymax_raw - ymin) / h)) + 1

    xmax = xmin + (nx - 1) * h
    ymax = ymin + (ny - 1) * h

    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)

    half_grid = n_grid // 2

    # Allows centers so patch fits inside [0, nx-1] x [0, ny-1]
    valid_center_col_indices = np.arange(half_grid, nx - half_grid)
    valid_center_row_indices = np.arange(half_grid, ny - half_grid)

    global_grid_info = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "nx": nx,
        "ny": ny,
        "n_grid": n_grid,
        "half_grid": half_grid,
        "h": h,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "valid_center_col_indices": valid_center_col_indices,
        "valid_center_row_indices": valid_center_row_indices
    }

    # height map over entire global grid
    global_grid_info["global_height_map"] = height_map_ray_casting(
        scene, global_grid_info, direction=(0.0, 0.0, -1.0)
    )

    return global_grid_info


def world_to_index(x: float, y: float, tx_grid_info: dict) -> tuple[int, int]:
    """Convert world position to height map index"""
    # Retrive grid information
    xmin = tx_grid_info["xmin"] # minimum x coordinate
    ymin = tx_grid_info["ymin"] # minimum y coordinate
    nx = tx_grid_info["nx"]     # number of points in x direction
    ny = tx_grid_info["ny"]     # number of points in y direction
    h = tx_grid_info["h"]       # grid step size

    # Convert world position to height map index
    row_index = int(np.clip(np.round((y - ymin) / h), 0, ny - 1))  # row (y)
    col_index = int(np.clip(np.round((x - xmin) / h), 0, nx - 1))  # col (x)
    return row_index, col_index

def _index_to_world(row_index: int, col_index: int, tx_grid_info: dict) -> tuple[float, float]:
    """Convert height map indices to world coordinates"""
    # Retrive grid information
    xmin = tx_grid_info["xmin"] # minimum x coordinate
    ymin = tx_grid_info["ymin"] # minimum y coordinate
    h = tx_grid_info["h"]       # grid step size

    # Convert to height map index to world postion
    x = xmin + col_index * h
    y = ymin + row_index * h

    return x, y

def snap_to_nearest_valid_position(position: List[float], tx_grid_info: dict) -> List[float]:
    """Snap world coordinates to nearest valid height map point"""
    # Retrieve current transmitter position
    x_current, y_current, _ = position

    # Convert world position to height map index
    row_index, col_index = world_to_index(x_current, y_current, tx_grid_info)

    # Get nearest index with valid height
    # If (col_index, row_index) is already valid then (col_index, row_index) = (valid_col_index, valid_row_index)
    valid_row_index, valid_col_index = tx_grid_info["nearest_idx"][:, row_index, col_index]

    z = tx_grid_info["height_map"][valid_row_index, valid_col_index]
    x, y = _index_to_world(valid_row_index, valid_col_index, tx_grid_info)

    return [float(x), float(y), float(z)]

def generate_grid_positions(
    config: TransmitterConfig, global_grid_info: dict | None = None
) -> Tuple[List[List[float]], dict]:
    if global_grid_info is None:
        raise ValueError("Global grid info must be provided.")

    grid_dim = int(np.ceil(np.sqrt(config.n_tx)))

    valid_center_col_indices = global_grid_info["valid_center_col_indices"]
    valid_center_row_indices = global_grid_info["valid_center_row_indices"]
    global_height_map = global_grid_info["global_height_map"]
    half_grid = global_grid_info["half_grid"]

    # Random grid center index
    col_center_index = np.random.choice(valid_center_col_indices)
    row_center_index = np.random.choice(valid_center_row_indices)

    # Determine grid corners indexes (start inclusive, end exclusive)
    col_left_index   = col_center_index - half_grid
    col_right_index  = col_center_index + half_grid        # end index (exclusive)
    row_bottom_index = row_center_index - half_grid
    row_top_index    = row_center_index + half_grid        # end index (exclusive)

    # Center world coordinates
    center_x, center_y = _index_to_world(row_center_index, col_center_index, global_grid_info)

    # **Bounds must use last included index: right-1, top-1**
    col_left_world,  row_bottom_world = _index_to_world(row_bottom_index, col_left_index, global_grid_info)
    col_right_world, row_top_world = _index_to_world(row_top_index - 1, col_right_index - 1, global_grid_info)

    # grid spacing within the patch
    grid_spacing = config.coverage_size / grid_dim

    # Extract height submap (n_grid x n_grid)
    grid_height_map = global_height_map[
        row_bottom_index:row_top_index,
        col_left_index:col_right_index
    ]

    grid_info = {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "map_bounds": [
            [float(col_left_world), float(col_right_world)],
            [float(row_bottom_world), float(row_top_world)],
        ],
        "grid_spacing": grid_spacing,
        "grid_dim": grid_dim,
        "grid_height_map": grid_height_map,
    }

    # Generate positions
    positions = []
    tx_count = 0

    for i in range(grid_dim):
        for j in range(grid_dim):
            if tx_count >= config.n_tx:
                break

            base_x = center_x + (i - grid_dim / 2 + 0.5) * grid_spacing
            base_y = center_y + (j - grid_dim / 2 + 0.5) * grid_spacing

            if config.grid_randomization > 0:
                rand_x = np.random.uniform(-grid_spacing / 2, grid_spacing / 2) * config.grid_randomization
                rand_y = np.random.uniform(-grid_spacing / 2, grid_spacing / 2) * config.grid_randomization
                base_x += rand_x
                base_y += rand_y

            positions.append([float(base_x), float(base_y), float(config.tx_default_height)])
            tx_count += 1

    return positions, grid_info

def safe_remove(scene: "Scene", name: str):
    try:
        scene.remove(name)
    except Exception:
        pass

def clear_previous(scene: "Scene", max_scan: int = 100):
    """Clear any previous transmitters"""
    for i in range(1, max_scan + 1):
        safe_remove(scene, f"tx_{i}")

def build(
    scene: "Scene",
    config: TransmitterConfig,
    global_grid_info: dict | None = None,
    tx_grid_info: dict | None = None
) -> tuple[List[List[float]], dict]:
    """Build transmitters on the scene"""
    config = config.validate()

    if config.seed is not None:
        np.random.seed(config.seed)

    # Clear previous transmitters
    clear_previous(scene)

    # Set up transmitter array
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=config.tx_array_pattern,
        polarization=config.polarization,
    )

    # Generate transmitter positions
    tx_positions, grid_info = generate_grid_positions(config, global_grid_info)

    # Create transmitters
    for i, pos in enumerate(tx_positions, start=1):
        # Validate current transmitter position
        if tx_grid_info is not None:
            pos = snap_to_nearest_valid_position(pos, tx_grid_info)
            pos[2] += config.tx_height_margin
            tx_positions[i - 1] = pos

        name = f"tx_{i}"
        scene.add(Transmitter(name=name, position=mi.Point3f(pos), power_dbm=config.tx_power_dbm, color=(0, 0, 1)))

    return tx_positions, grid_info

def extract_metric(radio_map: "RadioMap", metric_type: str) -> np.ndarray:
    """Extract the specified metric from the radio map"""
    if metric_type == "path_gain":
        return radio_map.path_gain
    elif metric_type == "rss":
        return radio_map.rss
    elif metric_type == "sinr":
        return radio_map.sinr
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    
def apply_db_conversion(radio_map: np.ndarray, config: TransmitterConfig, dbm: bool = False) -> np.ndarray:
        """Apply dB conversion to the radio map"""
        radio_map_db = 10.0 * np.log10(np.clip(radio_map, a_min=1e-15, a_max=None))
        if dbm:
            radio_map_db += 30.0  # Convert to dBm
        if config.db_floor is not None:
            radio_map_db = np.clip(radio_map_db, a_min=config.db_floor, a_max=config.db_ceiling)
        return radio_map_db

def generate_sample(
    scene: "Scene",
    sample_id: int,
    config: TransmitterConfig,
    tx_grid_info: dict | None = None,
    global_grid_info: dict | None = None,
    show_mask_generation_time: bool = False
) -> SuperResolutionDataSample:
    """Generate a single super-resolution data sample"""

    # Build transmitters in scene
    tx_positions, grid_info = build(scene, config, global_grid_info, tx_grid_info)
    tx_positions = np.asarray(tx_positions)
    cx, cy = grid_info["center_x"], grid_info["center_y"]

    # Generate sample height map
    start_time = time.time()
    height_map_from_ray = generate_sample_height_map(scene, grid_info, config)
    time_hmr = time.time() - start_time

    start_time = time.time()
    height_map_from_map = grid_info["grid_height_map"]
    time_hmm = time.time() - start_time
    
    # Generate (LR, HR) radio map pairs
    start_time = time.time()
    rm_lr = RM_SOLVER(
        scene,
        max_depth=5,
        samples_per_tx=10**6,
        cell_size=mi.Point2f(config.lr_cell_size),
        center=mi.Point3f([cx, cy, 0.0]),
        size=mi.Point2f([config.coverage_size, config.coverage_size]),
        orientation=mi.Point3f([0, 0, 0]),
    )
    time_lr = time.time() - start_time

    start_time = time.time()
    rm_hr = RM_SOLVER(
        scene,
        max_depth=5,
        samples_per_tx=10**6,
        cell_size=mi.Point2f(config.hr_cell_size),
        center=mi.Point3f([cx, cy, 0.0]),
        size=mi.Point2f([config.coverage_size, config.coverage_size]),
        orientation=mi.Point3f([0, 0, 0]),
    )
    time_hr = time.time() - start_time

    # Extract the specified metric
    metric_lr = extract_metric(rm_lr, metric_type=config.metric_type)
    metric_hr = extract_metric(rm_hr, metric_type=config.metric_type)

    # Assume one transmitter for now
    map_lr = np.max(metric_lr, axis=0)
    map_hr = np.max(metric_hr, axis=0)

    if config.to_db:
        if config.metric_type == "path_gain" or config.metric_type == "sinr":
            map_lr = apply_db_conversion(map_lr, config=config, dbm=False)
            map_hr = apply_db_conversion(map_hr, config=config, dbm=False)
        elif config.metric_type == "rss":
            map_lr = apply_db_conversion(map_lr, config=config, dbm=True)
            map_hr = apply_db_conversion(map_hr, config=config, dbm=True)

    rendered_image = scene.render(
        camera=Camera(
            position=mi.Point3f([getattr(rm_lr, "center")[0][0], getattr(rm_lr, "center")[1][0], 300]),
            look_at=mi.Point3f([getattr(rm_lr, "center")[0][0], getattr(rm_lr, "center")[1][0], 0])
        ),
        radio_map=rm_lr,
        fov=30.0,
        resolution=(256, 256),
        show_devices=True,
        return_bitmap=True,
        rm_metric="path_gain",
        rm_vmax=np.max(map_lr),
        rm_vmin=np.min(map_lr)
        )
    rendered_image = np.array(rendered_image)
    rendered_image = np.rot90(rendered_image, k=-1)
    rendered_image = np.flipud(rendered_image)
            
    # Create sample
    sample = SuperResolutionDataSample(
        sample_id=sample_id,
        tx_positions=tx_positions,
        height_map_from_ray=height_map_from_ray,
        height_map_from_map=height_map_from_map,
        map_lr=map_lr,
        map_hr=map_hr,
        scale=config.scale,
        metric_type=config.metric_type,
        grid_info=grid_info,
        config=config,
        time_hmr=time_hmr,
        time_hmm=time_hmm,
        time_lr=time_lr,
        time_hr=time_hr,
        rendered_image=rendered_image
    )

    return sample

def save_data(sample: SuperResolutionDataSample, save_dir: str | Path, naming: str) -> None:
    """Save a single data sample to disk as PNG image"""
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    save_dir = save_dir / naming
    os.makedirs(save_dir, exist_ok=True)
    sample_path = os.path.join(save_dir, f"sample{sample.sample_id:04d}.png")
    
    # Subplot LR and HR maps, height map, building mask, and LoS map
    n_rows = 1
    n_cols = 5

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Low resolution
    im0 = axes[0].imshow(sample.map_lr, origin="lower", cmap="viridis")
    axes[0].set_title(f"Low Resolution Map ({sample.metric_type})")
    fig.colorbar(im0, ax=axes[0], label="dB")
    
    # High resolution
    im1 = axes[1].imshow(sample.map_hr, origin="lower", cmap="viridis")
    axes[1].set_title(f"High Resolution Map ({sample.metric_type})")
    fig.colorbar(im1, ax=axes[1], label="dB")

    # Height map from ray casting
    if sample.height_map_from_ray is not None:
        im3 = axes[2].imshow(sample.height_map_from_ray, origin="lower", cmap="terrain")
        axes[2].set_title("Height Map (Ray Casting)")
        fig.colorbar(im3, ax=axes[2], label="Height (m)")
    else:
        axes[2].axis('off')

    # Height map from map info
    im4 = axes[3].imshow(sample.height_map_from_map, origin="lower", cmap="terrain")
    axes[3].set_title("Height Map (Map Info)")
    fig.colorbar(im4, ax=axes[3], label="Height (m)")

    # Rendered image
    im5 = axes[4].imshow(sample.rendered_image, origin="lower")
    axes[4].set_title("Rendered Image")
    fig.colorbar(im5, ax=axes[4])

    plt.tight_layout()
    plt.savefig(sample_path)
    plt.close(fig)

def get_scene(scene_name: str) -> "Scene":
    if scene_name == "etoile":
        scene = load_scene(sionna.rt.scene.etoile, merge_shapes=False)
    elif scene_name == "san_francisco":
        scene = load_scene(sionna.rt.scene.san_francisco, merge_shapes=False)
    elif scene_name == "munich":
        scene = load_scene(sionna.rt.scene.munich, merge_shapes=False)
    elif scene_name == "florence":
        scene = load_scene(sionna.rt.scene.florence, merge_shapes=False)
    else:
        raise ValueError(f"Unknown scene: {scene_name}")
    return scene

def generate_dataset(
    scene_name: str,
    base_config: TransmitterConfig,
    dataset_path: str | Path,
    show_progress: bool = True,
    show_mask_generation_time: bool = False
) -> None:
    """Generate multiple super-resolution data samples"""
    # Load scene
    scene = get_scene(scene_name)

    print(f"Generating {base_config.n_samples} super-resolution samples for {scene_name}...")
    print(f"Metric: {base_config.metric_type}, Scale: {base_config.scale}x")
    print(f"LR cell: {base_config.lr_cell_size}, HR cell: {base_config.hr_cell_size}")
    print(f"Coverage: {base_config.coverage_size}x{base_config.coverage_size}m")
    print("Low-res (LR) | High-res (HR) | Height Map Ray (HMR) | Height Map Mapping (HMM) | Coverage in X (dx) | Coverage in Y (dy)")

    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    print(f"Samples will be saved to: {dataset_path / scene_name}")

    # Get boundaries of the scene

    iterator = tqdm(range(base_config.n_samples), desc="Generating samples", leave=False) \
                if show_progress else range(base_config.n_samples)

    tx_grid_info = generate_tx_grid_info(scene, base_config)
    global_grid_info = generate_global_grid_info(scene, base_config)

    total_time_w_r = []
    total_time_wo_r = []

    for i in iterator:
        # Create config with unique seed for each sample
        from dataclasses import replace

        config = replace(base_config, seed=i + 200)
        sample = generate_sample(scene, i + 1, config, tx_grid_info, global_grid_info, show_mask_generation_time)
        save_data(sample, dataset_path, naming=scene_name)

        # Display sample info
        LR_str = str(tuple(sample.map_lr.shape))
        HR_str = str(tuple(sample.map_hr.shape))
        HMR_str = "N/A" if sample.height_map_from_ray is None else str(tuple(sample.height_map_from_ray.shape))
        HMM_str = str(tuple(sample.height_map_from_map.shape))
        dx = sample.grid_info["map_bounds"][0][1] - sample.grid_info["map_bounds"][0][0]
        dy = sample.grid_info["map_bounds"][1][1] - sample.grid_info["map_bounds"][1][0]
        if sample.height_map_from_ray is None:
            HMM_HRR_EQ = "N/A"
            HMM_HMR_SIM = "N/A"
        else:
            HMM_HRR_EQ = "Yes" if sample.height_map_from_ray is not None and \
                np.array_equal(sample.height_map_from_ray, sample.height_map_from_map) else "No"
            HMM_HMR_SIM = "Yes" if \
                np.allclose(sample.height_map_from_ray, sample.height_map_from_map, atol=1e-2, equal_nan=True) else "No"
        
        time_sample = np.nan
        time_sample_w_r = np.nan
        time_sample_wo_r = np.nan
        if show_mask_generation_time:
            LR_str += f" {sample.time_lr:.2f}s"
            HR_str += f" {sample.time_hr:.2f}s"
            HMR_str += f" {sample.time_hmr:.2f}s"
            HMM_str += f" {sample.time_hmm:.2f}s"

            time_sample_w_r = sample.time_lr + sample.time_hr + sample.time_hmr + sample.time_hmm
            total_time_w_r.append(time_sample_w_r)
            time_sample_wo_r = sample.time_lr + sample.time_hr + sample.time_hmm
            total_time_wo_r.append(time_sample_wo_r)

        if show_progress and isinstance(iterator, tqdm):
            iterator.set_postfix(
                {
                    "LR": LR_str,
                    "HR": HR_str,
                    "HMR": HMR_str,
                    "HMM": HMM_str,
                    "dx": f"{dx:.2f}m",
                    "dy": f"{dy:.2f}m",
                    "HMM=HRR": HMM_HRR_EQ,
                    "HMM~HMR": HMM_HMR_SIM
                }
            )
            tqdm.write(
                f"[sample {i+1}] LR={LR_str}, HR={HR_str}, "
                f"HMR={HMR_str}, HMM={HMM_str}, "
                f"dx={dx:.2f}m, dy={dy:.2f}m, "
                f"HMM=HRR={HMM_HRR_EQ}, HMM~HMR={HMM_HMR_SIM}"
                f", sample time (w/ ray)={time_sample_w_r:.2f}s"
                f", sample time (w/o ray)={time_sample_wo_r:.2f}s"
            )

    print(f"Sum of sample generation times (w/ ray): {np.sum(total_time_w_r):.2f}s")
    print(f"Average sample generation time (w/ ray): {np.mean(total_time_w_r):.2f}s")
    print(f"Sum of sample generation times (w/o ray): {np.sum(total_time_wo_r):.2f}s")
    print(f"Average sample generation time (w/o ray): {np.mean(total_time_wo_r):.2f}s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate super-resolution datasets for specified scenes."
    )

    parser.add_argument(
        "--scenes",
        nargs="+",  # one or more
        default=["etoile", "san_francisco", "munich", "florence"],
        help="List of scene names to generate datasets for.",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path.cwd() / "test_sample_data_generation",
        help="Directory where generated datasets (PNG images) will be saved.",
    )

    parser.add_argument(
        "--no-timer",
        action="store_true",
        default=False,
        help="Show mask generation time during dataset generation.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bar during dataset generation.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    for name in args.scenes:
        generate_dataset(
            scene_name = name,
            base_config = TransmitterConfig(),
            dataset_path = args.output_dir,
            show_progress = not args.no_progress,
            show_mask_generation_time = not args.no_timer,
        )

if __name__ == "__main__":
    main()
