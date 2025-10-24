from typing import List, Optional

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
    BasicConv,
    CharbonnierLoss,
    GradientLoss,
    Un,
    Upsampler,
    default_conv,
)


class ESRTLightningModule(LightningModule):
    def __init__(
        self,
        scale: int = 5,
        n_feats: int = 32,
        n_blocks: int = 1,
        kernel_size: int = 3,
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
        self.model = ESRT(upscale=scale, n_feats=n_feats, n_blocks=n_blocks, kernel_size=kernel_size)

        # Loss functions
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
            ssim = self.train_ssim(sr, hr)
            loss = loss + self.hparams.w_ssim * (1.0 - ssim)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32 and torch.isfinite(x).all()
        return self.model(x)

    def training_step(self, batch: dict) -> torch.Tensor:
        lr_maps = batch["lr"]
        hr_maps = batch["hr"]

        sr_maps = self(lr_maps)
        loss = self.criterion(sr_maps, hr_maps)

        psnr = self.train_psnr(sr_maps, hr_maps)
        ssim = self.train_ssim(sr_maps, hr_maps)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("train_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("train_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        lr_maps = batch["lr"]
        hr_maps = batch["hr"]

        sr_maps = self(lr_maps)
        loss = self.criterion(sr_maps, hr_maps)

        psnr = self.val_psnr(sr_maps, hr_maps)
        ssim = self.val_ssim(sr_maps, hr_maps)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=lr_maps.size(0))
        self.log("val_sr_std", sr_maps.std(), on_epoch=True, prog_bar=False, batch_size=lr_maps.size(0))

        try:
            if len(self._val_examples) < self.num_val_log_images:
                remaining = self.num_val_log_images - len(self._val_examples)
                take = min(remaining, lr_maps.size(0))
                sample_ids = batch.get("sample_id")
                for i in range(take):
                    sid = None
                    if sample_ids is not None:
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
            pass

        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_examples = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_examples:
            return

        logger = getattr(self, "logger", None)
        if logger is None:
            self._val_examples = []
            return

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


class ESRT(nn.Module):
    def __init__(
        self,
        upscale: int = 4,
        n_feats: int = 32,
        n_blocks: int = 1,
        kernel_size: int = 3,
        conv: callable = default_conv,
    ) -> None:
        super(ESRT, self).__init__()

        def wn(x: nn.Module) -> nn.Module:
            return torch.nn.utils.weight_norm(x)

        n_feats = 32
        n_blocks = 1
        kernel_size = 3
        scale = upscale  # args.scale[0] #gaile
        # act = nn.ReLU(True)
        # self.up_sample = F.interpolate(scale_factor=2, mode='nearest')
        self.n_blocks = n_blocks

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(1, n_feats, kernel_size)]  # changed from 3

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(Un(n_feats=n_feats, wn=wn))

        # define tail module
        modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, 1, kernel_size)]  # changed from 3

        # changed first from 3
        self.up = nn.Sequential(Upsampler(conv, scale, n_feats, act=False), BasicConv(n_feats, 1, 3, 1, 1))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.reduce = conv(n_blocks * n_feats, n_feats, kernel_size)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None, test: bool = False) -> torch.Tensor:
        # x1 = self.sub_mean(x1)
        x1 = self.head(x1)
        res2 = x1
        # res2 = x2
        body_out = []
        for i in range(self.n_blocks):
            x1 = self.body[i](x1)
            body_out.append(x1)
        res1 = torch.cat(body_out, 1)
        res1 = self.reduce(res1)

        x1 = self.tail(res1)
        x1 = self.up(res2) + x1
        # x1 = self.add_mean(x1)
        # x2 = self.tail(res2)
        return x1

    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find("tail") >= 0:
    #                     print("Replace pre-trained upsampler to new one...")
    #                 else:
    #                     raise RuntimeError(
    #                         "While copying the parameter named {}, "
    #                         "whose dimensions in the model are {} and "
    #                         "whose dimensions in the checkpoint are {}.".format(
    #                             name, own_state[name].size(), param.size()
    #                         )
    #                     )
    #         elif strict:
    #             if name.find("tail") == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'.format(name))

    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    #     # MSRB_out = []from model import common
