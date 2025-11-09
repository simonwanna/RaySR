import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule

try:
    from lightning.pytorch.loggers.wandb import WandbLogger  # type: ignore
except Exception:
    WandbLogger = None  # type: ignore

from poc.models.modeling_utils import (
    SCPA,
    BamBlock,
    CharbonnierLoss,
    GradientLoss,
    PixelAttentionBlock,
    make_layer,
)


class PANLightningModule(LightningModule):
    def __init__(
        self,
        scale: int = 5,
        in_nc: int = 1,
        out_nc: int = 1,
        nf: int = 40,
        unf: int = 24,
        nb: int = 16,
        bam: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        w_charb: float = 1.0,
        w_grad: float = 0.2,
        w_ssim: float = 0.1,
        num_val_log_images: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.model = PanModel(scale, in_nc, out_nc, nf, unf, nb, bam)

        # Loss function
        # self.criterion = nn.L1Loss()  # TODO: Add weighted MSE?
        self.loss_charb = CharbonnierLoss()
        self.loss_grad = GradientLoss()

        # Metrics
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.val_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.val_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        # Image logging buffer
        self._val_examples: List[dict] = []
        self.num_val_log_images: int = num_val_log_images

    def criterion(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Weighted sum of losses"""
        loss = 0.0
        if self.hparams.w_charb:
            loss = loss + self.hparams.w_charb * self.loss_charb(sr, hr)
        if self.hparams.w_grad:
            loss = loss + self.hparams.w_grad * self.loss_grad(sr, hr)
        if self.hparams.w_ssim:
            # SSIM as loss
            ssim = self.train_ssim(sr, hr)
            loss = loss + self.hparams.w_ssim * (1.0 - ssim)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32 and torch.isfinite(x).all()
        return self.model(x)

    def training_step(self, batch: dict) -> torch.Tensor:
        lr_maps = batch["lr"]
        hr_maps = batch["hr"]

        # Forward pass
        sr_maps = self(lr_maps)

        # Compute loss
        loss = self.criterion(sr_maps, hr_maps)

        # Compute metrics
        psnr = self.train_psnr(sr_maps, hr_maps)
        ssim = self.train_ssim(sr_maps, hr_maps)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("train_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("train_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        lr_maps = batch["lr"]
        hr_maps = batch["hr"]

        # Forward pass
        sr_maps = self(lr_maps)

        # Compute loss
        loss = self.criterion(sr_maps, hr_maps)

        # Compute metrics
        psnr = self.val_psnr(sr_maps, hr_maps)
        ssim = self.val_ssim(sr_maps, hr_maps)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_sr_std", sr_maps.std(), on_epoch=True, prog_bar=False, batch_size=lr_maps.size(0))

        # Collect sample triplets for image logging (first few from first batches)
        try:
            if len(self._val_examples) < self.num_val_log_images:
                remaining = self.num_val_log_images - len(self._val_examples)
                take = min(remaining, lr_maps.size(0))
                sample_ids = batch.get("sample_id")
                for i in range(take):
                    sid = None
                    if sample_ids is not None:
                        # sample_ids could be a tensor with shape [B]
                        try:
                            sid = int(sample_ids[i].item())
                        except Exception:
                            sid = None
                    self._val_examples.append(
                        {
                            "lr": lr_maps[i].detach().cpu(),
                            "hr": hr_maps[i].detach().cpu(),
                            "sr": sr_maps[i].detach().cpu(),
                            "sample_id": sid,
                        }
                    )
        except Exception:
            # Never break validation due to logging collection
            pass

        return loss

    def on_validation_epoch_start(self) -> None:
        # Reset buffer at the start of each validation epoch
        self._val_examples = []

    def on_validation_epoch_end(self) -> None:
        # Log collected validation image triplets (LR | SR | HR) to the logger (if WandB)
        if not self._val_examples:
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            self._val_examples = []
            return

        # Only log images when using WandB logger
        is_wandb = WandbLogger is not None and isinstance(logger, WandbLogger)
        if not is_wandb:
            self._val_examples = []
            return

        import numpy as np
        from matplotlib import cm

        cmap = cm.get_cmap("viridis")
        eps = 1e-12

        def to_rgb(panel: torch.Tensor) -> "np.ndarray":
            arr = panel.squeeze(0).numpy()
            a_min, a_max = float(arr.min()), float(arr.max())
            norm = (arr - a_min) / (a_max - a_min + eps) if a_max - a_min >= eps else np.zeros_like(arr)
            rgba = cmap(norm)
            return (rgba[..., :3] * 255).astype(np.uint8)

        colored_imgs: List["np.ndarray"] = []
        captions: List[str] = []
        for ex in self._val_examples:
            lr = ex["lr"]
            hr = ex["hr"]
            sr = ex["sr"]
            lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode="nearest").squeeze(0)
            triplet_rgb = np.concatenate([to_rgb(lr_up), to_rgb(sr), to_rgb(hr)], axis=1)
            colored_imgs.append(triplet_rgb)
            sid = ex.get("sample_id")
            captions.append(f"sample_id={sid}" if sid is not None else "val_sample")

        logger.log_image(key="val/image_triplets", images=colored_imgs, caption=captions, step=int(self.global_step))
        self._val_examples = []

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.hparams.scheduler != "cosine" else None,
                "frequency": 1,
            },
        }


class PanModel(nn.Module):
    """Borrowed with modifications from https://github.com/eugenesiow/super-image"""

    def __init__(self, scale: int, in_nc: int, out_nc: int, nf: int, unf: int, nb: int, bam: bool):
        super(PanModel, self).__init__()

        # SCPA
        self.scale = scale  # scale for upsampling
        in_nc = in_nc  # input channels
        out_nc = out_nc  # output channels
        nf = nf  # number of features
        unf = unf  # number of features for upsampling
        nb = nb  # number of blocks
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)

        # first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        if bam:
            self.att1 = BamBlock(unf, reduction=8)
        else:
            self.att1 = PixelAttentionBlock(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            if bam:
                self.att2 = BamBlock(unf, reduction=8)
            else:
                self.att2 = PixelAttentionBlock(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ilr = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = out + ilr
        return out
