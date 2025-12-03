import logging
import os
import pickle
import subprocess
from pathlib import Path

import hydra
import sionna
from omegaconf import DictConfig, OmegaConf
from sionna.rt import load_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def generate_data(cfg: DictConfig) -> None:
    logging.info("Data Generation Configuration:")
    logging.info(OmegaConf.to_yaml(cfg))

    tx_grid_info = None
    scene_grid_info = None

    # Set paths
    base_path = Path(cfg.data_dir) / "grid_data" / cfg.scene_name
    suffix = (
        f"{str(cfg.transmitter.scale).replace('.', '-')}_"
        f"{str(cfg.transmitter.coverage_size).replace('.', '-')}_"
        f"{str(cfg.transmitter.hr_grid_size).replace('.', '-')}_"
        f"{str(cfg.generator.step_length).replace('.', '-')}_"
        f"{str(cfg.generator.min_object_height).replace('.', '-')}.pkl"
    )
    tx_grid_info_path = base_path / f"tx_grid_info_{suffix}"
    scene_grid_info_path = base_path / f"scene_grid_info_{suffix}"

    # Check if height map and tx grid info needs to be created
    need_hm = not os.path.exists(scene_grid_info_path) and cfg.transmitter.include_height_map
    need_tx = not os.path.exists(tx_grid_info_path) and cfg.snap_to_grid

    if need_hm or need_tx:
        subprocess.run(
            [
                "python",
                "src/poc/data_modules/height_map_generator.py",
                "--scene_name",
                cfg.scene_name,
                "--scale",
                str(cfg.transmitter.scale),
                "--coverage_size",
                str(cfg.transmitter.coverage_size),
                "--hr_grid_size",
                str(cfg.transmitter.hr_grid_size),
                "--data_dir",
                cfg.data_dir,
                "--step_length",
                str(cfg.generator.step_length),
                "--min_object_height",
                str(cfg.generator.min_object_height),
                "--include_height_map",
                str(cfg.transmitter.include_height_map),
            ]
        )

    if cfg.snap_to_grid:
        try:
            with open(tx_grid_info_path, "rb") as f:
                tx_grid_info = pickle.load(f)
                logging.info("Loaded tx_grid_info from: %s", tx_grid_info_path)
        except Exception as e:
            logger.error("tx_grid_info file not needed or not found at: %s", tx_grid_info_path)
            raise e
    else:
        logger.info("tx_grid_info not used")

    if cfg.transmitter.include_height_map:
        try:
            with open(scene_grid_info_path, "rb") as f:
                scene_grid_info = pickle.load(f)
                logging.info("Loaded scene_grid_info from: %s", scene_grid_info_path)
        except Exception as e:
            logger.error("scene_grid_info file not needed or not found at: %s", scene_grid_info_path)
            raise e
    else:
        logging.info("scene_grid_info not used")

    # if cfg.scene_name == "empty":
    #     scene = load_scene()
    if cfg.scene_name == "etoile":
        scene = load_scene(sionna.rt.scene.etoile)
    elif cfg.scene_name == "san_francisco":
        scene = load_scene(sionna.rt.scene.san_francisco)
    elif cfg.scene_name == "munich":
        scene = load_scene(sionna.rt.scene.munich)
    elif cfg.scene_name == "florence":
        scene = load_scene(sionna.rt.scene.florence)
    else:
        raise ValueError(f"Unknown scene: {cfg.scene_name}")

    scene.frequency = cfg.frequency

    # Create transmitter config
    tx_config = hydra.utils.instantiate(cfg.transmitter)

    # Generate dataset
    generator = hydra.utils.instantiate(cfg.generator, scene=scene)
    generator.generate_dataset(base_config=tx_config, tx_grid_info=tx_grid_info, scene_grid_info=scene_grid_info)

    logging.info(f"Dataset saved at {cfg.data_dir}")


if __name__ == "__main__":
    generate_data()
