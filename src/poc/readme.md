# POC
## 🚀 Overview

Proof of concept code for generating data, training, testing, and visualizing super-resolution models for radio maps. 
It leverages [Hydra](https://hydra.cc/) for configuration and [Lightning](https://lightning.ai/) for model training.

---

## 📦 Setup

Same as project. \
Run `uv sync` to install/update environment.

---

## ⚙️ Configuration

- All scripts use Hydra configs from configs.
- Override any config at the command line:
    ```sh
    uv run train model.nf=64 data.n_samples=200
    ```
- To see the full config for a run, just run the script; it prints the config at startup.

---

## 🛠️ Data Generation

Generate a dataset of low-res/high-res radio map pairs:

```sh
uv run generate
```

- **Config:** `configs/data/radio_maps.yaml`
- **Output:** `.pt` files in the directory specified by `data.dataset_path`.

- You can loop over the scenes creating n_samples per scene in one go:
    ```bash
    for scene in munich florence san_francisco etoile; do
        uv run generate scene_name="$scene";
    done
    ```

---

## 🏋️ Training

Train the super-resolution model:

```sh
uv run train
```

- **Config:** `configs/model/pan.yaml`, `configs/trainer/default.yaml`
- **Output:** Checkpoints and logs in the Hydra run directory.

- To enable Weights & Biases logging (metrics + validation image triplets each epoch), use the WandB trainer config:
  ```sh
  uv run train trainer=wandb
  ```
  Optionally set `WANDB_PROJECT`/`WANDB_ENTITY` env vars. Images are logged as triples: `[LR (nearest-upsampled) | SR (pred) | HR]`.

---

## 🧪 Testing

Evaluate a trained model and save results:

```sh
uv run test
```

- **Config:** `configs/test/test.yaml`
- **Output:** For each sample, saves a `.pt` file with:
    - Low-res input
    - High-res ground truth
    - Model prediction
    - Bicubic upsampled baseline
    - PSNR/SSIM metrics for both model and baseline

---

## 📊 Visualization

Visualize results and compare model vs. baseline:

```sh
uv run src/poc/visualize.py --results_dir outputs/results
```

- Shows LR, HR, predicted, and bicubic images with PSNR/SSIM metrics.

---

## 🧩 Notes

- For experiments, edit or copy config files in `configs/`.
- Data generation places transmitters on a grid with possible horizontal/vertical randomization. The center of the grid is randomized within a range and placed in a scene. Change `scene_name` for diverse data. **Note** that the scale factor needs to be the same for the model later on.
- The model currently implemented is [PAN (Pixel Attention Network)](https://arxiv.org/abs/2010.01073). It uses a CNN backbone with pixel attention blocks to focus on important regions, and a residual connection with bilinear interpolation to learn only the missing details.   Input is a low-res radio map; output is a super-resolved map at *scale* higher resolution.

---

## 🛠️ TODO

- [x] Fix: Model output too similar to low-res input
- [ ] Add larger scenes (using Blender)
- [x] Add WandB logging (with image logging)
- [ ] Multiprocessing for data generation (maybe not needed? Pretty fast already)
- [ ] Data augmentation (flip/rotate, etc.)
- [ ] Warn if transmitters are placed too close/outside scene (current error handling enough?)
- [ ] Prevent transmitters from being placed inside buildings etc.
- [x] Save checkpoints in Hydra job folder
- [x] Fix image mirroring bug in generation
- [ ] Add multi-channel support (e.g., where buildings are / mask for where there are no signal)
