import sionna
from sionna.rt import load_scene, Scene

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

def height_map(
        scene: Scene,
        step_size_exponent: int = 0,
        min_height: float = 15.0,
    ) -> np.ndarray:
    """Generate a full height map for the scene."""

    # Define step size
    step_size = 10 ** step_size_exponent    # e.g., exponent=-1 -> step_size=0.1
                                            # must be power of 10 for rounding
                                            # and height map indexing to work correctly
    # Full scene bounding box
    scene_bbox = scene.mi_scene.bbox()

    # Discretize full scene area
    x = np.arange(
        start=scene_bbox.min.x,
        stop=scene_bbox.max.x,
        step=step_size
    ).round(-step_size_exponent)

    y = np.arange(
        start=scene_bbox.min.y,
        stop=scene_bbox.max.y,
        step=step_size
    ).round(-step_size_exponent)

    ny, nx = len(y), len(x)
    z = np.full((ny, nx), np.nan)

    x_index = {x[i]: i for i in range(nx)}
    y_index = {y[j]: j for j in range(ny)}

    # Get valid objects bounding boxes
    object_bboxes = []
    for obj in tqdm(scene.objects.values(), desc="Validating scene objects"):
        if getattr(obj, "mi_mesh", None) is not None \
            and obj.name not in ['ground', 'Terrain', 'Plane', 'floor'] \
                and obj.mi_mesh.bbox().extents()[2] >= min_height:
            object_bboxes.append(obj.mi_mesh.bbox())

    # Process each bounding box and fill height map
    for obj_bbox in tqdm(object_bboxes, desc="Processing bounding boxes"):
        x_obj = np.arange(
            start=obj_bbox.min.x,
            stop=obj_bbox.max.x + step_size,  # Ensure inclusion of xmax due to rounding
            step=step_size
        ).round(-step_size_exponent)

        y_obj = np.arange(
            start=obj_bbox.min.y,
            stop=obj_bbox.max.y + step_size,  # Ensure inclusion of ymax due to rounding
            step=step_size
        ).round(-step_size_exponent)

        for xn in x_obj:
            for yn in y_obj:
                i = x_index.get(xn, None)
                j = y_index.get(yn, None)
                if i is not None and j is not None:
                    # If no value yet, or current bbox max.z is higher, update
                    if np.isnan(z[j, i]):
                        z[j, i] = obj_bbox.max.z
                    else:
                        z[j, i] = obj_bbox.max.z if obj_bbox.max.z > z[j, i] else z[j, i]

    # Something causes left-right flip, so correct here
    # z = np.fliplr(z)
    z_index = {(j, i): (x_index[x[i]], y_index[y[j]]) for i in range(nx) for j in range(ny)}

    return {"x": x, "y": y, "z": z, "x_index": x_index, "y_index": y_index, "z_index": z_index}

def test_height_map_plot(scene_name: str):
    # Load scene
    if scene_name:
        # plot one image
        scene = load_scene(scene_name, merge_shapes=False)
        height_map_data = height_map(
                scene,
                step_size_exponent=0,
                min_height=10.0,
            )
        
        name_str = scene_name.split('\\')[-2]
        print(f"Height map generation for {name_str} completed.")
        
        # Plot height map
        plt.figure(figsize=(7,7))
        plt.imshow(height_map_data["z"], cmap='viridis', origin='lower')
        plt.title(f"Height map for scene: {name_str}")
        plt.colorbar(label='Height (m)')

    else:
        scene_names = [
            sionna.rt.scene.etoile, 
            sionna.rt.scene.munich, 
            sionna.rt.scene.san_francisco, 
            sionna.rt.scene.florence
            ]

        _, ax = plt.subplots(1, len(scene_names), figsize=(5 * len(scene_names), 5))

        for i, name in enumerate(scene_names):
            scene = load_scene(name, merge_shapes=False)

            # Generate height map using ray casting
            start = timer()
            height_map_data = height_map(
                scene,
                step_size_exponent=0,
                min_height=10.0,
            )
            end = timer()

            name_str = name.split('\\')[-2]
            print(f"Height map generation for {name_str} "
                f"took {end - start:.2f} seconds.")

            # Plot height map
            ax[i].imshow(height_map_data["z"], cmap='viridis', origin='lower')
            ax[i].set_title(name_str)

            # Add colorbar for each subplot
            cbar = plt.colorbar(ax[i].images[0], ax=ax[i])
        
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def place_transmitters(z: np.ndarray,
                       steps_from_center: int = 100,
                       tx_margin: int = 30,
                       n_tx: int = 5,
                       rng: np.random.Generator | None = None):
    """
    z: (ny, nx) height map with NaN for invalid cells.
    All sizes are in *index cells* (not meters).
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = z.shape

    # Outer (TX) half-width and inner (camera) half-width, in indices
    r_cam  = steps_from_center // 2 if steps_from_center % 2 == 0 else steps_from_center/2
    r_tx   = (steps_from_center + tx_margin) // 2 if (steps_from_center + tx_margin) % 2 == 0 else (steps_from_center + tx_margin)/2

    # Ensure both squares fit within the image
    pad = int(np.ceil(r_tx))
    i_center = rng.integers(pad, nx - pad)   # x/col
    j_center = rng.integers(pad, ny - pad)   # y/row

    # Convert half-widths to integer index extents
    r_cam_i = int(np.floor(r_cam))
    r_cam_j = int(np.floor(r_cam))
    r_tx_i  = int(np.floor(r_tx))
    r_tx_j  = int(np.floor(r_tx))

    # Outer (TX) square bounds [inclusive]
    i0_tx = i_center - r_tx_i
    i1_tx = i_center + r_tx_i
    j0_tx = j_center - r_tx_j
    j1_tx = j_center + r_tx_j

    # Inner (camera) square bounds [inclusive]
    i0_cam = i_center - r_cam_i
    i1_cam = i_center + r_cam_i
    j0_cam = j_center - r_cam_j
    j1_cam = j_center + r_cam_j

    # Slice outer area and find valid (non-NaN) cells
    Z_cut = z[j0_tx:j1_tx+1, i0_tx:i1_tx+1]
    valid_local = np.argwhere(~np.isnan(Z_cut))  # (row, col) within the outer square

    if valid_local.size == 0:
        raise RuntimeError("No valid TX cells inside the TX area. Try different margins or center.")

    if n_tx > len(valid_local):
        n_tx = len(valid_local)  # cap to available cells

    # Sample without replacement
    pick = rng.choice(len(valid_local), size=n_tx, replace=False)
    tx_local = valid_local[pick]  # shape (n_tx, 2): (row_in_cut, col_in_cut)

    # Convert to global indices
    tx_global = np.stack([j0_tx + tx_local[:, 0], i0_tx + tx_local[:, 1]], axis=1)  # (y=row, x=col)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(z, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Height (m)')

    # Outer TX area (yellow dashed)
    ax.add_patch(Rectangle((i0_tx, j0_tx),
                           i1_tx - i0_tx + 1, j1_tx - j0_tx + 1,
                           fill=False, lw=2, ls='--', ec='yellow', label='TX area'))

    # Inner camera area (red solid)
    ax.add_patch(Rectangle((i0_cam, j0_cam),
                           i1_cam - i0_cam + 1, j1_cam - j0_cam + 1,
                           fill=False, lw=2.5, ls='-', ec='red', label='Camera area'))

    # Center marks
    ax.scatter(i_center, j_center, c='red', s=120, marker='x', label='Center')

    # TX points
    ax.scatter(tx_global[:,1], tx_global[:,0], c='lime', s=50, marker='o', label=f'TX (n={len(tx_global)})')

    ax.set_title('TX placement within valid area')
    ax.set_xlabel('X (index)'); ax.set_ylabel('Y (index)')
    ax.legend(loc='upper center', framealpha=0.85)
    plt.tight_layout()
    plt.show()

    return {
        "center_ij": (j_center, i_center),
        "camera_bounds_ij": (j0_cam, j1_cam, i0_cam, i1_cam),
        "tx_bounds_ij": (j0_tx, j1_tx, i0_tx, i1_tx),
        "tx_indices_ij": tx_global  # array of (row=y, col=x)
    }

def test_tx_placement():
    # Load scene
    scene = load_scene(sionna.rt.scene.etoile, merge_shapes=False)

    # Generate height map using ray casting
    x, y, z, x_index, y_index, z_index = height_map(
        scene,
        step_size_exponent=0,
        min_height=10.0,
    )

    # Place transmitters
    placement_info = place_transmitters(
        z,
        steps_from_center=100,
        tx_margin=30,
        n_tx=10,
        rng=np.random.default_rng(42)
    )
    print("Placement info:", placement_info)


if __name__ == "__main__":
    test_height_map_plot(sionna.rt.scene.etoile)
    # test_tx_placement()