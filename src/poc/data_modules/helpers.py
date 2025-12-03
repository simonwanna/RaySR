import matplotlib.pyplot as plt
import torch


def standarize_img(img: torch.Tensor) -> torch.Tensor:
    img_mean, img_std = img.mean(), img.std()
    img = (img - img_mean) / (img_std + 1e-16)
    return img


def normalize_img(img: torch.Tensor, img_mean: float, img_std: float) -> torch.Tensor:
    img = img * (img_std + 1e-16) + img_mean
    return img


def visualize_sample(sample_path: str) -> None:
    visualize_metric(sample_path)
    visualize_masks(sample_path)


def visualize_masks(sample_path: str) -> None:
    sample = torch.load(sample_path, weights_only=False)
    num_cols = 3
    fig, (ax1, ax2, ax3) = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))

    has_hmap = "height_map" in sample
    has_bmask = "building_mask" in sample
    has_los = "los_mask" in sample

    if not (has_hmap or has_bmask or has_los):
        print("No masks to display in the sample.")
        return

    bounds = sample["grid_info"]["map_bounds"]
    extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]

    if has_hmap:
        ax1.imshow(sample["height_map"].numpy(), cmap="viridis", extent=extent, origin="lower", aspect="equal")
        ax1.scatter(sample["tx_positions"][:, 0], sample["tx_positions"][:, 1], c="red", s=50, marker="x")
        ax1.set_title("Height Map")
        fig.colorbar(
            ax1.images[0],
            ax=ax1,
            label="Height (m)",
            ticks=[sample["height_map"].min().item(), sample["height_map"].max().item()],
        )
    else:
        ax1.axis("off")

    if has_bmask:
        ax2.imshow(sample["building_mask"].numpy(), cmap="gray", extent=extent, origin="lower", aspect="equal")
        ax2.scatter(sample["tx_positions"][:, 0], sample["tx_positions"][:, 1], c="red", s=50, marker="x")
        ax2.set_title("Building Mask")
        fig.colorbar(ax2.images[0], ax=ax2, label="Building Presence", ticks=[0, 1])

    else:
        ax2.axis("off")

    if has_los:
        ax3.imshow(sample["los_mask"].numpy(), cmap="gray", extent=extent, origin="lower", aspect="equal")
        ax3.scatter(sample["tx_positions"][:, 0], sample["tx_positions"][:, 1], c="red", s=50, marker="x")
        ax3.set_title("LOS Mask")
        fig.colorbar(ax3.images[0], ax=ax3, label="LOS Presence", ticks=[0, 1])
    else:
        ax3.axis("off")

    plt.show()


def visualize_metric(sample_path: str) -> None:
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
