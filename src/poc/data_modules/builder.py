from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# import drjit as dr
import numpy as np
import sionna
from sionna.rt import PlanarArray, Transmitter

# dr.set_flag(dr.JitFlag.Debug, True)  # NOTE: use if you get error in san_francisco processing...


@dataclass(frozen=True)
class TransmitterConfig:
    """Configurations for transmitter setup. Used by SceneTransmitterBuilder."""

    n_tx: int
    scale: int  # Super-resolution scale factor (e.g., 5 for 10m->2m)
    coverage_size: float  # Square coverage area size in meters (e.g., 500.0)
    hr_grid_size: int = 512

    # Grid placement (evenly spaced grid with optional randomization)
    grid_randomization: float = 0.0  # 0.0 = perfect grid, 1.0 = fully random within cells
    margin: float = 30.0  # margin to leave from scene borders when placing transmitters

    # TX parameters
    tx_power_dbm: float = 44.0
    tx_array_pattern: str = "iso"
    polarization: str = "V"
    tx_default_height: float = 45.0
    tx_height_margin: float = 5.0

    # Reproducibility
    seed: Optional[int] = None

    def validate(self) -> TransmitterConfig:
        if self.n_tx <= 0:
            raise ValueError("n_tx must be > 0.")
        if self.scale <= 1:
            raise ValueError("scale must be > 1.")
        if self.coverage_size <= 0:
            raise ValueError("coverage_size must be > 0.")
        if not (0.0 <= self.grid_randomization <= 1.0):
            raise ValueError("grid_randomization must be between 0.0 and 1.0.")
        return self

    @property
    def hr_cell_size(self) -> Tuple[float, float]:  # TODO: make HR cell size configurable
        """High resolution cell size (target resolution)"""
        # For 1:1 aspect ratio, use same size for both dimensions
        base_cell = self.coverage_size / self.hr_grid_size
        return (base_cell, base_cell)

    @property
    def lr_cell_size(self) -> Tuple[float, float]:
        """Low resolution cell size (input resolution)"""
        hr_size = self.hr_cell_size[0]
        lr_size = hr_size * self.scale
        return (lr_size, lr_size)


class SceneTransmitterBuilder:
    """
    Builder class for transmitter setup.
    Creates a grid for transmitters with configurable randomization.
    Build method will (based on the grid) create and add transmitters to the provided scene.
    """

    def __init__(self, scene: "sionna.rt.Scene"):
        self.scene = scene

    def _safe_remove(self, name: str):
        try:
            self.scene.remove(name)
        except Exception:
            pass

    def _clear_previous(self, max_scan: int = 100):
        """Clear any previous transmitters"""
        for i in range(1, max_scan + 1):
            self._safe_remove(f"tx_{i}")

    @staticmethod
    def _generate_grid_positions(config: TransmitterConfig, scene_corners: tuple) -> Tuple[List[List[float]], dict]:
        """Generate transmitter positions and return grid info"""
        grid_dim = int(np.ceil(np.sqrt(config.n_tx)))

        # Determine grid center
        if scene_corners is not None:
            (x_min, x_max), (y_min, y_max) = scene_corners
            x_min += config.coverage_size / 2
            x_max -= config.coverage_size / 2
            y_min += config.coverage_size / 2
            y_max -= config.coverage_size / 2
            center_x = np.random.uniform(x_min, x_max)
            center_y = np.random.uniform(y_min, y_max)
        else:
            center_x, center_y = 0.0, 0.0

        grid_spacing = config.coverage_size / grid_dim

        # Calculate map bounds (full coverage area)
        map_xmin = center_x - config.coverage_size / 2
        map_xmax = center_x + config.coverage_size / 2
        map_ymin = center_y - config.coverage_size / 2
        map_ymax = center_y + config.coverage_size / 2

        # Grid info
        grid_info = {
            "center_x": center_x,
            "center_y": center_y,
            "map_bounds": [[map_xmin, map_xmax], [map_ymin, map_ymax]],
            "grid_spacing": grid_spacing,
            "grid_dim": grid_dim,
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

    def build(
        self, config: TransmitterConfig, scene_corners: tuple, tx_grid_info: dict = None
    ) -> tuple[List[List[float]], dict]:
        """Build transmitters on the scene"""
        config = config.validate()

        if config.seed is not None:
            np.random.seed(config.seed)

        # Clear previous transmitters
        self._clear_previous()

        # Set up transmitter array
        self.scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            pattern=config.tx_array_pattern,
            polarization=config.polarization,
        )

        # Generate transmitter positions
        tx_positions, grid_info = self._generate_grid_positions(config, scene_corners)

        # Create transmitters
        for i, pos in enumerate(tx_positions, start=1):
            # Validate current transmitter position
            if tx_grid_info is not None:
                pos = _snap_to_nearest_valid_position(pos, tx_grid_info)
                pos[2] += config.tx_height_margin
                tx_positions[i - 1] = pos

            name = f"tx_{i}"
            self.scene.add(Transmitter(name=name, position=pos, power_dbm=config.tx_power_dbm, color=(0, 0, 1)))

        return tx_positions, grid_info


def _world_to_index(x: float, y: float, tx_grid_info: dict) -> tuple[int, int]:
    """Convert world position to height map index"""
    # Retrive grid information
    xmin = tx_grid_info["xmin"]
    ymin = tx_grid_info["ymin"]
    nx = tx_grid_info["nx"]  # number of points in x direction
    ny = tx_grid_info["ny"]  # number of points in y direction
    h = tx_grid_info["h"]  # grid step size

    # Convert world position to height map index
    row_index = int(np.clip(np.round((y - ymin) / h), 0, ny - 1))  # row (y)
    col_index = int(np.clip(np.round((x - xmin) / h), 0, nx - 1))  # col (x)

    return row_index, col_index


def _index_to_world(row_index: int, col_index: int, tx_grid_info: dict) -> tuple[float, float]:
    """Convert height map indices to world coordinates"""
    # Retrive grid information
    xmin = tx_grid_info["xmin"]
    ymin = tx_grid_info["ymin"]
    h = tx_grid_info["h"]

    # Convert to height map index to world postion
    x = xmin + col_index * h
    y = ymin + row_index * h

    return x, y


def _snap_to_nearest_valid_position(position: List[float], tx_grid_info: dict) -> List[float]:
    """Snap world coordinates to nearest valid height map point"""
    # Retrieve current transmitter position
    x_current, y_current, _ = position

    # Convert world position to height map index
    row_index, col_index = _world_to_index(x_current, y_current, tx_grid_info)

    # Get nearest index with valid height
    # If (col_index, row_index) is already valid then (col_index, row_index) = (valid_col_index, valid_row_index)
    valid_row_index, valid_col_index = tx_grid_info["nearest_idx"][:, row_index, col_index]

    z = tx_grid_info["height_map"][valid_row_index, valid_col_index]
    x, y = _index_to_world(valid_row_index, valid_col_index, tx_grid_info)

    return [float(x), float(y), float(z)]
