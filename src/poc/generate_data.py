import logging

import hydra
import sionna
from omegaconf import DictConfig, OmegaConf
from sionna.rt import load_scene

from poc.data_modules.builder import TransmitterConfig
from poc.data_modules.generator import RadioMapDataGenerator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def generate_data(cfg: DictConfig) -> None:
    logging.info("Data Generation Configuration:")
    logging.info(OmegaConf.to_yaml(cfg.data))

    # Load scene; TODO: set max grid size based on scene
    if cfg.data.scene_name == "empty":
        scene = load_scene()
    elif cfg.data.scene_name == "etoile":
        scene = load_scene(sionna.rt.scene.etoile)
    elif cfg.data.scene_name == "san_francisco":
        scene = load_scene(sionna.rt.scene.san_francisco)
    elif cfg.data.scene_name == "munich":
        scene = load_scene(sionna.rt.scene.munich)
    elif cfg.data.scene_name == "florence":
        scene = load_scene(sionna.rt.scene.florence)
    else:
        raise ValueError(f"Unknown scene: {cfg.data.scene_name}")

    scene.frequency = cfg.data.frequency

    # Create transmitter config
    tx_config = TransmitterConfig(**cfg.data.transmitter_config)

    # Generate dataset
    generator = RadioMapDataGenerator(scene, cfg.data.metric_type)
    generator.generate_dataset(
        cfg.data.n_samples,
        save_dir=cfg.data.dataset_path,
        base_config=tx_config,
        naming_convention=cfg.data.naming_convention,
    )

    logging.info(f"Dataset saved at {cfg.data.dataset_path}")


if __name__ == "__main__":
    generate_data()
