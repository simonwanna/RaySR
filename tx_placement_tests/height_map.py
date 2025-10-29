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

    z = np.full((len(x), len(y)), np.nan)

    x_index = {x[i]: i for i in range(len(x))}
    y_index = {y[i]: i for i in range(len(y))}

    # Get valid objects bounding boxes
    object_bboxes = []
    for obj in tqdm(scene.objects.values(), desc="Validating scene objects"):
        if getattr(obj, "mi_mesh", None) is not None \
            and obj.name not in ['ground', 'Terrain', 'Plane', 'floor'] \
                and obj.mi_mesh.bbox().extents()[2] >= min_height:
            object_bboxes.append(obj.mi_mesh.bbox())

    # Process each bounding box and fill height map
    for obj_bbox in tqdm(object_bboxes, desc="Processing coordinates from bounding boxes"):
        xmin, xmax = obj_bbox.min.x, obj_bbox.max.x
        ymin, ymax = obj_bbox.min.y, obj_bbox.max.y

        x_obj = np.arange(
            start=xmin,
            stop=xmax + step_size,  # Ensure inclusion of xmax due to rounding
            step=step_size
        ).round(-step_size_exponent)

        y_obj = np.arange(
            start=ymin,
            stop=ymax + step_size,  # Ensure inclusion of ymax due to rounding
            step=step_size
        ).round(-step_size_exponent)

        for xi in x_obj:
            for yj in y_obj:
                i = x_index.get(xi, None)
                j = y_index.get(yj, None)
                if i is not None and j is not None:
                    # If no value yet, or current bbox max.z is higher, update
                    if np.isnan(z[i, j]):
                        z[i, j] = obj_bbox.max.z
                    else:
                        z[i, j] = obj_bbox.max.z if obj_bbox.max.z > z[i, j] else z[i, j]

    # Something causes left-right flip, so correct here
    z = np.fliplr(z)

    return x, y, z

if __name__ == "__main__":
    # Load scene
    scene_names = [
        sionna.rt.scene.etoile, 
        sionna.rt.scene.munich, 
        sionna.rt.scene.san_francisco, 
        sionna.rt.scene.florence
        ]

    fig, ax = plt.subplots(1, len(scene_names), figsize=(5 * len(scene_names), 5))

    for i, name in enumerate(scene_names):
        scene = load_scene(name, merge_shapes=False)

        # Generate height map using ray casting
        start = timer()
        x, y, z = height_map(
            scene,
            step_size_exponent=0,
            min_height=10.0,
        )
        end = timer()

        name_str = name.split('\\')[-2]
        print(f"Height map generation for {name_str} "
              f"took {end - start:.2f} seconds.")

        # Plot height map
        ax[i].imshow(z, cmap='viridis', origin='lower')
        ax[i].set_title(name_str)

        # Add colorbar for each subplot
        cbar = plt.colorbar(ax[i].images[0], ax=ax[i])
    
    plt.tight_layout()
    plt.show()