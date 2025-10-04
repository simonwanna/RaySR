import logging
import os
from typing import List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

# from poc.data_modules.helpers import standarize_img
# from poc.data_modules.helpers import normalize_img


class SuperResolutionDataset(Dataset):
    """Lazy-loading Dataset for super-resolution radio map data"""

    def __init__(self, sample_paths: List[str], db_floor: float, db_ceiling: float):
        self.sample_paths = sample_paths
        self.db_floor = db_floor
        self.db_ceiling = db_ceiling

    def __len__(self):
        return len(self.sample_paths)

    def _norm_db01(self, x_db: torch.Tensor) -> torch.Tensor:
        x01 = (x_db - self.db_floor) / (self.db_ceiling - self.db_floor)
        return torch.clamp(x01, 0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        # Load only one sample
        sample_data = torch.load(self.sample_paths[idx], map_location="cpu", weights_only=False)

        # If only one channel, add channel dimension; FIXME: handle multi-channel case
        lr = sample_data["map_lr"].unsqueeze(0)  # [1, H, W]
        hr = sample_data["map_hr"].unsqueeze(0)  # [1, H, W]

        # Normalize to [0, 1] range
        lr = self._norm_db01(lr)
        hr = self._norm_db01(hr)

        return {
            "lr": lr,
            "hr": hr,
            "tx_positions": sample_data["tx_positions"],
            "sample_id": sample_data["sample_id"],
            "scale": sample_data["scale"],
            "metric_type": sample_data["metric_type"],
        }


class RadioMapDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        dataset_path: str = None,
        train_split: float = 0.8,
        val_split: float = 0.2,
        db_floor: float = -150.0,
        db_ceiling: float = -50.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None) -> None:
        """Setup datasets for training and validation"""
        if self.dataset is None:
            self.dataset = self._load_dataset(self.hparams.dataset_path, self.hparams.db_floor, self.hparams.db_ceiling)

        if stage == "fit" or stage is None:
            # Split dataset
            n_train = int(len(self.dataset) * self.hparams.train_split)
            n_val = len(self.dataset) - n_train

            self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def _load_dataset(save_dir: str, db_floor: float, db_ceiling: float) -> SuperResolutionDataset:
        """Lazy load dataset from directory with per-sample files"""

        # Collect all sample files
        sample_files = sorted(os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".pt"))
        if not sample_files:
            raise FileNotFoundError(f"No sample files found in {save_dir}")

        logging.info(f"Found {len(sample_files)} samples in {save_dir}")

        # Return lazy dataset
        return SuperResolutionDataset(sample_files, db_floor=db_floor, db_ceiling=db_ceiling)
