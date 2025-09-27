import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from poc.data_modules.helpers import standarize_img


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().cpu().numpy()
    return arr


def show_sample(result_path: str) -> None:
    data = torch.load(result_path, map_location="cpu")
    lr = tensor_to_image(standarize_img(data["lr"]))
    hr = tensor_to_image(standarize_img(data["hr"]))
    pred = tensor_to_image(standarize_img(data["pred"]))
    interp = tensor_to_image(standarize_img(data["interp"]))
    metrics = data["metrics"]

    extent = None
    origin = "upper"

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    im0 = axs[0, 0].imshow(lr, cmap="viridis", extent=extent, origin=origin)
    axs[0, 0].set_title(f"LR\nPSNR: {metrics['model']['psnr']:.2f}\nSSIM: {metrics['model']['ssim']:.3f}")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(hr, cmap="viridis", extent=extent, origin=origin)
    axs[0, 1].set_title("HR (Ground Truth)")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(interp, cmap="viridis", extent=extent, origin=origin)
    axs[1, 0].set_title(f"Bicubic\nPSNR: {metrics['bicubic']['psnr']:.2f}\nSSIM: {metrics['bicubic']['ssim']:.3f}")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(pred, cmap="viridis", extent=extent, origin=origin)
    axs[1, 1].set_title(f"Model\nPSNR: {metrics['model']['psnr']:.2f}\nSSIM: {metrics['model']['ssim']:.3f}")
    axs[1, 1].axis("off")

    fig.colorbar(im0, ax=axs, orientation="vertical", fraction=0.02, pad=0.04, label="Signal Strength (dB)")

    plt.show()


def main(results_dir: str) -> None:
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".pt"))
    for f in files:
        print(f"Showing {f}")
        show_sample(os.path.join(results_dir, f))
        input("Press Enter for next sample (Ctrl+C to quit)...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with result_*.pt files")
    args = parser.parse_args()
    main(args.results_dir)
