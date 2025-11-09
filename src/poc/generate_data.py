import logging

import hydra
import sionna
from omegaconf import DictConfig, OmegaConf
from sionna.rt import load_scene


@hydra.main(version_base=None, config_path="configs", config_name="config")
def generate_data(cfg: DictConfig) -> None:
    logging.info("Data Generation Configuration:")
    logging.info(OmegaConf.to_yaml(cfg))

    # Load scene; TODO: set max grid size based on scene
    # merge_shapes=False to determine seperate object heights
    # if cfg.scene_name == "empty":
    #     scene = load_scene()
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

    # Create transmitter config
    tx_config = hydra.utils.instantiate(cfg.transmitter)

    # Generate dataset
    generator = hydra.utils.instantiate(cfg.generator, scene=scene)
    generator.generate_dataset(base_config=tx_config)

    logging.info(f"Dataset saved at {cfg.data_dir}")


if __name__ == "__main__":
    generate_data()
