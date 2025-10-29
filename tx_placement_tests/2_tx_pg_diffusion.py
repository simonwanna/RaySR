from sionna.rt import load_scene, RadioMapSolver, Transmitter, PlanarArray, Scene
import numpy as np
import matplotlib.pyplot as plt

positions = np.arange(1, 50, 1)
    
def path_gain_loss_test(scene: Scene, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Test the scene with transmitters at specified positions and return high-resolution and low-resolution path gain maps."""
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization='V', pattern='iso')
    
    hr_results = np.zeros((len(positions), 512, 512))
    lr_results = np.zeros((len(positions), 256, 256))

    for i in range(len(positions)):
        scene.remove('tx1')
        scene.remove('tx2')

        tx1 = Transmitter('tx1', position=[float(positions[i]), 0.0, 10.0])
        tx2 = Transmitter('tx2', position=[-float(positions[i]), 0.0, 10.0])

        scene.add(tx1)
        scene.add(tx2)

        rm_solver = RadioMapSolver()

        hr_rm = rm_solver(
            scene, 
            max_depth=5, 
            samples_per_tx=10**6,
            cell_size=(160.0 / 512, 160.0 / 512),
            center=(0.0, 0.0, 0.0),
            size=(160.0, 160.0),
            orientation=(0.0, 0.0, 0.0)
        )

        lr_rm = rm_solver(
            scene, 
            max_depth=5, 
            samples_per_tx=10**6,
            cell_size=(160.0 / 256, 160.0 / 256),
            center=(0.0, 0.0, 0.0),
            size=(160.0, 160.0),
            orientation=(0.0, 0.0, 0.0)
        )

        hr_results[i] = np.max(hr_rm.path_gain, axis=0)
        lr_results[i] = np.max(lr_rm.path_gain, axis=0)

    return hr_results, lr_results


def heatmap(z: np.ndarray) -> None:
    """Plot the path gain map using matplotlib."""
    plt.imshow(z, cmap='viridis', origin='lower')
    plt.colorbar(label='Path Gain (dB)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Path Gain Map')
    plt.show()


hr_results, lr_results = path_gain_loss_test(load_scene(), positions)
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Fix color limits across frames
hr_vmin, hr_vmax = np.nanmin(hr_results), np.nanmax(hr_results)
lr_vmin, lr_vmax = np.nanmin(lr_results), np.nanmax(lr_results)

# Initialize the first frame
im0 = ax[0].imshow(hr_results[0], cmap='viridis', origin='lower',
                   vmin=hr_vmin, vmax=hr_vmax)
ax[0].set_title(f'HR Path Gain Map - Position {positions[0]}')
ax[0].set_xlabel('X-axis'); ax[0].set_ylabel('Y-axis')
cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
cbar0.set_label('Path Gain')

im1 = ax[1].imshow(lr_results[0], cmap='viridis', origin='lower',
                   vmin=lr_vmin, vmax=lr_vmax)
ax[1].set_title(f'LR Path Gain Map - Position {positions[0]}')
ax[1].set_xlabel('X-axis'); ax[1].set_ylabel('Y-axis')
cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
cbar1.set_label('Path Gain')

fig.tight_layout()

for i in range(len(positions)):
    im0.set_data(hr_results[i])
    im1.set_data(lr_results[i])

    # Compute approximate path gain in the center
    hr_pg = hr_results[i][hr_results[i].shape[0] // 2, hr_results[i].shape[1] // 2]
    lr_pg = lr_results[i][lr_results[i].shape[0] // 2, lr_results[i].shape[1] // 2]

    ax[0].set_title(f'HR Path Gain Map - Path Gain {hr_pg:.2e} dB')
    ax[1].set_title(f'LR Path Gain Map - Path Gain {lr_pg:.2e} dB')

    plt.pause(0.3)

plt.ioff()
plt.show()