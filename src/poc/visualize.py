import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().cpu().numpy()
    return arr


def unnormalize(arr: np.ndarray, db_floor: float, db_ceiling: float) -> np.ndarray:
    return arr * (db_ceiling - db_floor) + db_floor


def show_sample(result_path: str, save: bool = False, db_floor: float = -150.0, db_ceiling: float = -50.0) -> None:
    data = torch.load(result_path, map_location="cpu")

    # Un-normalize to dBm
    lr = unnormalize(tensor_to_image(data["lr"]), db_floor, db_ceiling)
    hr = unnormalize(tensor_to_image(data["hr"]), db_floor, db_ceiling)
    pred = unnormalize(tensor_to_image(data["pred"]), db_floor, db_ceiling)
    interp = unnormalize(tensor_to_image(data["interp"]), db_floor, db_ceiling)

    metrics = data["metrics"]

    extent = None
    origin = "upper"

    # Common scaling for all plots
    vmin, vmax = db_floor, db_ceiling

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # Use same vmin/vmax for all
    im0 = axs[0, 0].imshow(lr, cmap="viridis", extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("LR (Input)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(hr, cmap="viridis", extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("HR (Ground Truth)")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(interp, cmap="viridis", extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title(
        f"Bicubic\nPSNR: {metrics['bicubic']['psnr']:.2f}\nSSIM: \
            {metrics['bicubic']['ssim']:.3f}\nRMSE: {metrics['bicubic']['rmse_db']:.2f} dB"
    )
    axs[1, 0].axis("off")

    axs[1, 1].imshow(pred, cmap="viridis", extent=extent, origin=origin, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title(
        f"Model\nPSNR: {metrics['model']['psnr']:.2f}\nSSIM: \
            {metrics['model']['ssim']:.3f}\nRMSE: {metrics['model']['rmse_db']:.2f} dB"
    )
    axs[1, 1].axis("off")

    fig.colorbar(im0, ax=axs, orientation="vertical", fraction=0.02, pad=0.04, label="Signal Strength (dBm)")

    if save:
        plt.savefig(result_path.replace(".pt", ".png"))
        plt.close(fig)
    else:
        plt.show()


def main(args: argparse.Namespace) -> None:
    files = sorted(f for f in os.listdir(args.results_dir) if f.endswith(".pt"))
    for f in files:
        print(f"Visualizing {f}")
        show_sample(os.path.join(args.results_dir, f), args.save, args.db_floor, args.db_ceiling)
        if not args.save:
            input("Press Enter for next sample (Ctrl+C to quit)...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with result_*.pt files")
    parser.add_argument("--save", action="store_true", help="Save images instead of showing them")
    parser.add_argument("--db_floor", type=float, default=-150.0, help="DB floor for normalization")
    parser.add_argument("--db_ceiling", type=float, default=-50.0, help="DB ceiling for normalization")
    args = parser.parse_args()
    main(args)
