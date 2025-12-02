import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import mitsuba as mi
import numpy as np
import sionna
from scipy import ndimage as ndi
from sionna.rt import load_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sionna.rt.scene import Scene

DIRECTION_DOWN = (0.0, 0.0, -1.0)
DIRECTION_UP = (0.0, 0.0, 1.0)


def _ray_cast(
    scene: "Scene",
    disc_info: dict,
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> np.ndarray:
    """
    Generate height map by ray casting the scene.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param disc_info: Discretization information for the grid to be ray casted
    :type disc_info: dict
    :param direction: Direction of the ray casting
    :type direction: tuple[float, float, float]
    :return: Height map as a 2D numpy array
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    xmin = disc_info["xmin"]
    xmax = disc_info["xmax"]
    ymin = disc_info["ymin"]
    ymax = disc_info["ymax"]
    nx = disc_info["nx"]
    ny = disc_info["ny"]

    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)

    mi_scene = scene.mi_scene

    ray_origin_height = float(mi_scene.bbox().max.z) + 1.0
    ray_origin_height *= -1.0 * direction[2]

    X, Y = np.meshgrid(x_vals, y_vals)
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = np.full(X.size, ray_origin_height)

    directions = np.array(direction)

    origins = mi.Vector3f(Xf, Yf, Zf)
    directions = mi.Vector3f(directions)

    ray = mi.Ray3f(o=origins, d=directions)
    intersect = mi_scene.ray_intersect(ray=ray, coherent=mi.Bool(True), ray_flags=mi.RayFlags.Minimal)
    hits = intersect.p.z
    valid = intersect.is_valid()

    hits = np.array(hits, dtype=float)
    valid_mask = np.array(valid, dtype=bool)
    hits[~valid_mask] = np.nan

    height_map = hits.reshape(ny, nx)

    return height_map


def _generate_tx_height_map(scene: "Scene", tx_disc_info: dict, min_object_height: float) -> np.ndarray:
    """
    Generate transmitter height map by ray casting the scene.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param tx_disc_info: Transmitter grid discretization information
    :type tx_disc_info: dict
    :param min_object_height: Minimum height of objects to be considered
    :type min_object_height: float
    :return: Height map as a 2D numpy array
    :rtype: ndarray[Any, Any]
    """
    tx_height_map = _ray_cast(scene, tx_disc_info, direction=DIRECTION_DOWN)
    ground_height_map = _ray_cast(scene, tx_disc_info, direction=DIRECTION_UP)
    height_above_ground = tx_height_map - ground_height_map
    tx_height_map[height_above_ground < min_object_height] = np.nan

    return tx_height_map


def _generate_scene_height_map(scene: "Scene", scene_disc_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate scene height map, building mask, and ground height map by ray casting the scene.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param scene_disc_info: Scene grid discretization information
    :type scene_disc_info: dict
    :return: Tuple containing scene height map, building mask, and ground height map
    :rtype: tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    """
    scene_height_map = _ray_cast(scene, scene_disc_info, direction=DIRECTION_DOWN)
    ground_height_map = _ray_cast(scene, scene_disc_info, direction=DIRECTION_UP)
    height_above_ground = scene_height_map - ground_height_map
    building_mask = height_above_ground > 0.0

    return scene_height_map, building_mask, ground_height_map


def _generate_nearest_neighbor_indexes(height_map: np.ndarray) -> np.ndarray:
    """
    Generate nearest valid neighbor indexes for each point in the height map.

    :param height_map: Height map with NaN values for invalid points
    :type height_map: np.ndarray
    :return: Nearest valid neighbor indexes for each point in the height map
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    valid = ~np.isnan(height_map)
    nearest_idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    nearest_idx = np.array(nearest_idx)
    return nearest_idx


def _get_disc_info(scene: "Scene", step_length: float) -> dict:
    """
    Generate discretization information for the scene based on step length.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param step_length: Step length for discretization
    :type step_length: float
    :return: Discretization information as a dictionary
    :rtype: dict[Any, Any]
    """
    mi_scene = scene.mi_scene
    bbox = mi_scene.bbox()
    h = step_length

    xmin, xmax_raw = float(bbox.min.x), float(bbox.max.x)
    ymin, ymax_raw = float(bbox.min.y), float(bbox.max.y)

    nx = int(np.floor((xmax_raw - xmin) / h)) + 1
    ny = int(np.floor((ymax_raw - ymin) / h)) + 1

    xmax = xmin + (nx - 1) * h
    ymax = ymin + (ny - 1) * h

    disc_info = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "nx": nx,
        "ny": ny,
        "h": h,
    }

    return disc_info


def _generate_tx_grid_info(scene: "Scene", config: dict) -> dict:
    """
    Generate transmitter grid info based on scene geometry.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param config: Configuration dictionary
    :type config: dict
    :return: Transmitter grid information as a dictionary
    :rtype: dict[Any, Any]
    """
    # Get scene discretization info
    step_length = config["step_length"]
    min_object_height = config["min_object_height"]

    tx_disc_info = _get_disc_info(scene, step_length)
    tx_height_map = _generate_tx_height_map(scene, tx_disc_info, min_object_height)
    nearest_idx = _generate_nearest_neighbor_indexes(tx_height_map)

    logger.info(
        f"  Grid Coverage=({(tx_disc_info['xmin'], tx_disc_info['xmax'])}, "
        f"{(tx_disc_info['ymin'], tx_disc_info['ymax'])})"
    )
    logger.info(f"  Grid Size={(tx_disc_info['nx'], tx_disc_info['ny'])}")
    logger.info(f"  Grid Step Size={tx_disc_info['h']}")
    logger.info(f"  Height Map Shape={tx_height_map.shape}")

    # Store transmitter grid info
    tx_grid_info = {
        "xmin": tx_disc_info["xmin"],  # minimum x coordinate
        "xmax": tx_disc_info["xmax"],  # maximum x coordinate
        "ymin": tx_disc_info["ymin"],  # minimum y coordinate
        "ymax": tx_disc_info["ymax"],  # maximum y coordinate
        "nx": tx_disc_info["nx"],  # number of points in x direction
        "ny": tx_disc_info["ny"],  # number of points in y direction
        "h": tx_disc_info["h"],  # grid step size
        "height_map": tx_height_map,  # height map matrix
        "nearest_idx": nearest_idx,  # nearest valid neighbor indexes
    }

    return tx_grid_info


def _generate_scene_grid_info(scene: "Scene", config: dict) -> dict:
    """
    Generate scene grid info based on scene geometry.

    :param scene: Sionna Scene object
    :type scene: "Scene"
    :param config: Configuration dictionary
    :type config: dict
    :return: Scene grid information as a dictionary
    :rtype: dict[Any, Any]
    """
    # Get scene discretization info
    grid_length = int(config["hr_grid_size"] / config["scale"])
    half_grid_length = grid_length // 2
    step_length = config["coverage_size"] / (grid_length - 1)

    scene_disc_info = _get_disc_info(scene, step_length)

    # Generate height map
    scene_height_map, scene_building_mask, scene_ground_height_map = _generate_scene_height_map(scene, scene_disc_info)

    center_col_indices = np.arange(half_grid_length, scene_disc_info["nx"] - half_grid_length)
    center_row_indices = np.arange(half_grid_length, scene_disc_info["ny"] - half_grid_length)

    x_vals = np.linspace(scene_disc_info["xmin"], scene_disc_info["xmax"], scene_disc_info["nx"])
    y_vals = np.linspace(scene_disc_info["ymin"], scene_disc_info["ymax"], scene_disc_info["ny"])

    x_diffs = x_vals[center_col_indices + half_grid_length - 1] - x_vals[center_col_indices - half_grid_length]
    y_diffs = y_vals[center_row_indices + half_grid_length - 1] - y_vals[center_row_indices - half_grid_length]

    logger.info(
        f"  Grid Coverage=({(scene_disc_info['xmin'], scene_disc_info['xmax'])}, "
        f"{(scene_disc_info['ymin'], scene_disc_info['ymax'])})"
    )
    logger.info(f"  Grid Size={(scene_disc_info['nx'], scene_disc_info['ny'])}")
    logger.info(f"  Grid Step Size={scene_disc_info['h']}")
    logger.info(f"  Grid Length={grid_length}")
    logger.info(f"  Height Map Shape={scene_height_map.shape}")

    expected = config["coverage_size"]
    if not np.allclose(x_diffs, expected, rtol=1e-6, atol=1e-9):
        logger.error(
            "Error in scene grid generation: Current grid in x direction does not match expected coverage size."
        )
        for i in range(center_col_indices.shape[0]):
            if not np.isclose(x_diffs[i], expected, rtol=1e-6, atol=1e-9):
                logger.error(
                    f"    Index {center_col_indices[i]} to {center_col_indices[i] + grid_length - 1}: "
                    f"{x_diffs[i]} (expected {expected})"
                )
        logger.error("Exiting due to grid size mismatch.")
        exit(1)
    else:
        logger.info("    Scene grid in x direction matches expected coverage size.")

    if not np.allclose(y_diffs, expected, rtol=1e-6, atol=1e-9):
        logger.error(
            "Error in scene grid generation: Current grid in y direction does not match expected coverage size."
        )
        for i in range(center_row_indices.shape[0]):
            if not np.isclose(y_diffs[i], expected, rtol=1e-6, atol=1e-9):
                logger.error(
                    f"    Index {center_row_indices[i]} to {center_row_indices[i] + grid_length - 1}: "
                    f"{y_diffs[i]} (expected {expected})"
                )
        logger.error("Exiting due to grid size mismatch.")
        exit(1)
    else:
        logger.info("    Scene grid in y direction matches expected coverage size.")

    # Store scene grid info
    scene_grid_info = {
        "xmin": scene_disc_info["xmin"],  # minimum x coordinate
        "xmax": scene_disc_info["xmax"],  # maximum x coordinate
        "ymin": scene_disc_info["ymin"],  # minimum y coordinate
        "ymax": scene_disc_info["ymax"],  # maximum y coordinate
        "nx": scene_disc_info["nx"],  # number of points in x direction
        "ny": scene_disc_info["ny"],  # number of points in y direction
        "h": scene_disc_info["h"],  # grid step size
        "ngrid": grid_length,  # number of points along one side of the square LR grid
        "height_map": scene_height_map,  # height map matrix
        "building_mask": scene_building_mask,  # building mask
        "ground_height_map": scene_ground_height_map,  # ground height map
        "center_col_indices": center_col_indices,  # valid center column indices for LR grids
        "center_row_indices": center_row_indices,  # valid center row indices for LR grids
    }

    return scene_grid_info


def generate(config: dict) -> None:
    """
    Docstring for generate

    :param config: Configuration dictionary
    :type config: dict
    """
    base_path = Path(config["data_dir"]) / "grid_data" / config["scene_name"]
    suffix = (
        f"{str(config['scale']).replace('.', '-')}_"
        f"{str(config['coverage_size']).replace('.', '-')}_"
        f"{str(config['hr_grid_size']).replace('.', '-')}_"
        f"{str(config['step_length']).replace('.', '-')}_"
        f"{str(config['min_object_height']).replace('.', '-')}.pkl"
    )
    tx_grid_info_path = base_path / f"tx_grid_info_{suffix}"
    scene_grid_info_path = base_path / f"scene_grid_info_{suffix}"
    os.makedirs(base_path, exist_ok=True)

    if os.path.exists(tx_grid_info_path) and os.path.exists(scene_grid_info_path):
        logger.info("tx_grid_info already exists at " + str(tx_grid_info_path))
        logger.info("scene_grid_info already exists at " + str(scene_grid_info_path))
        logger.info("Skipping height map generation.")
        return

    if config["scene_name"] == "etoile":
        scene = load_scene(sionna.rt.scene.etoile)
    elif config["scene_name"] == "san_francisco":
        scene = load_scene(sionna.rt.scene.san_francisco)
    elif config["scene_name"] == "munich":
        scene = load_scene(sionna.rt.scene.munich)
    elif config["scene_name"] == "florence":
        scene = load_scene(sionna.rt.scene.florence)
    else:
        raise ValueError(f"Unknown scene: {config['scene_name']}")

    logger.info(f"Loaded scene: {config['scene_name']}")
    logger.info(
        "Generating height maps with "
        f"scale={config['scale']}, "
        f"coverage_size={config['coverage_size']}, "
        f"hr_grid_size={config['hr_grid_size']}, "
        f"step_length={config['step_length']}, "
        f"min_object_height={config['min_object_height']}"
    )

    logger.info("Transmitter grid info:")
    tx_grid_info = _generate_tx_grid_info(scene, config)

    logger.info("Scene grid info:")
    scene_grid_info = _generate_scene_grid_info(scene, config)

    with open(tx_grid_info_path, "wb") as f:
        pickle.dump(tx_grid_info, f)
    logger.info(f"Saved transmitter grid info to {tx_grid_info_path}")

    with open(scene_grid_info_path, "wb") as f:
        pickle.dump(scene_grid_info, f)
    logger.info(f"Saved scene grid info to {scene_grid_info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate height maps for a given scene.")
    parser.add_argument("--scene_name", type=str, required=True, help="Name of the scene to generate height maps for.")
    parser.add_argument("--scale", type=int, required=True, help="Scale factor for the height maps.")
    parser.add_argument("--coverage_size", type=float, required=True, help="Coverage size for the height maps.")
    parser.add_argument(
        "--hr_grid_size", type=int, required=True, help="High-resolution grid size for the height maps."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to save generated height maps.")
    parser.add_argument("--step_length", type=float, required=True, help="Step length for the height maps.")
    parser.add_argument(
        "--min_object_height", type=float, required=True, help="Minimum object height for the height maps."
    )

    config = vars(parser.parse_args())

    generate(config)
