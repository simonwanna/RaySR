import os
from typing import Dict

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from poc.data_modules.rm_module import SuperResolutionDataset
from poc.models.pan import PANLightningModule


def load_model(checkpoint_path: str, device: torch.device) -> PANLightningModule:
    model = PANLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    psnr = PeakSignalNoiseRatio().to(pred.device)
    ssim = StructuralSimilarityIndexMeasure().to(pred.device)
    return {
        "psnr": psnr(pred, target).item(),
        "ssim": ssim(pred, target).item(),
    }


def save_result(
    save_dir: str,
    sample_id: int,
    lr: torch.Tensor,
    hr: torch.Tensor,
    pred: torch.Tensor,
    interp: torch.Tensor,
    metrics: Dict,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "sample_id": sample_id,
            "lr": lr.cpu(),
            "hr": hr.cpu(),
            "pred": pred.cpu(),
            "interp": interp.cpu(),
            "metrics": metrics,
        },
        os.path.join(save_dir, f"result_{sample_id:04d}.pt"),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def test(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SuperResolutionDataset(
        [os.path.join(cfg.test.data_dir, f) for f in sorted(os.listdir(cfg.test.data_dir)) if f.endswith(".pt")],
        db_floor=cfg.test.db_floor,
        db_ceiling=cfg.test.db_ceiling,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(cfg.test.checkpoint, device)
    scale = cfg.test.scale

    all_metrics = []

    for batch in tqdm(dataloader, desc="Testing"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sample_id = batch["sample_id"].item()

        with torch.no_grad():
            pred = model(lr)

        interp = bicubic_upsample(lr, scale=scale)

        metrics = {
            "model": compute_metrics(pred, hr),
            "bicubic": compute_metrics(interp, hr),
        }

        save_result(cfg.test.save_dir, sample_id, lr, hr, pred, interp, metrics)
        all_metrics.append({"sample_id": sample_id, **metrics})

    avg_model_psnr = sum(m["model"]["psnr"] for m in all_metrics) / len(all_metrics)
    avg_bicubic_psnr = sum(m["bicubic"]["psnr"] for m in all_metrics) / len(all_metrics)
    print(f"Average Model PSNR: {avg_model_psnr:.2f}, Bicubic PSNR: {avg_bicubic_psnr:.2f}")


if __name__ == "__main__":
    test()
