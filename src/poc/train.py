import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    L.seed_everything(cfg.seed, workers=True)

    # Instantiate data module and model
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate trainer
    trainer = hydra.utils.instantiate(cfg.trainer)

    # Train model
    trainer.fit(model, datamodule)

    # Test model (optional)
    if hasattr(cfg, "test") and cfg.test:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    train()
