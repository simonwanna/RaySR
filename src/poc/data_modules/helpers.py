import matplotlib.pyplot as plt
import torch


def standarize_img(img: torch.Tensor) -> torch.Tensor:
    img_mean, img_std = img.mean(), img.std()
    img = (img - img_mean) / (img_std + 1e-16)
    return img


def visualize_sample(sample_path: str) -> None:
    sample = torch.load(sample_path, weights_only=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Use grid_info for map bounds
    bounds = sample["grid_info"]["map_bounds"]
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]  # [xmin, xmax, ymin, ymax]

    ax1.imshow(sample["map_lr"].numpy(), cmap="viridis", extent=extent, origin="lower", aspect="equal")
    ax1.scatter(sample["tx_positions"][:, 0], sample["tx_positions"][:, 1], c="red", s=50, marker="x")
    ax1.set_title(f"LR Scale: {sample['scale']}x")

    ax2.imshow(sample["map_hr"].numpy(), cmap="viridis", extent=extent, origin="lower", aspect="equal")
    ax2.scatter(sample["tx_positions"][:, 0], sample["tx_positions"][:, 1], c="red", s=50, marker="x")
    ax2.set_title("HR")

    # add colorbar
    cbar = fig.colorbar(ax1.images[0], ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Signal Strength (dB)")

    plt.show()
