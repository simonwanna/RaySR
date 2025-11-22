import logging
import time

import sionna
import torch
from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

from poc.models.pan import PANLightningModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_scene() -> sionna.rt.Scene:
    """Load the Etoile scene and setup a default transmitter."""
    logger.info("Loading Munich scene...")
    scene = load_scene(sionna.rt.scene.munich)

    # Add a transmitter
    tx1 = Transmitter(name="tx1", position=[0, 0, 20], power_dbm=44.0)
    scene.add(tx1)
    tx2 = Transmitter(name="tx2", position=[50, 0, 20], power_dbm=44.0)
    scene.add(tx2)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    return scene


def measure_solver(scene: sionna.rt.Scene, cell_size: float, coverage_size: float = 500.0, samples: int = 10) -> float:
    """Measure the execution time of the Sionna RadioMapSolver."""
    solver = RadioMapSolver()

    # Warmup
    logger.info(f"Warming up solver for cell_size={cell_size}...")
    rm = solver(
        scene,
        max_depth=5,
        samples_per_tx=100_000,
        cell_size=cell_size,
        center=[0, 0, 0],
        size=[coverage_size, coverage_size],
        orientation=[0, 0, 0],
    )

    shape = rm.path_gain.shape
    h, w = shape[-2], shape[-1]
    num_pixels = h * w
    logger.info(f"Solver Output Shape: {shape}, Pixels: {num_pixels} ({h}x{w})")

    times = []
    logger.info(f"Running {samples} samples for solver...")
    for _ in range(samples):
        start_time = time.perf_counter()
        _ = solver(
            scene,
            max_depth=5,
            samples_per_tx=100_000,
            cell_size=cell_size,
            center=[0, 0, 0],
            size=[coverage_size, coverage_size],
            orientation=[0, 0, 0],
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    return avg_time, num_pixels


def measure_sr(model: PANLightningModule, input_shape: tuple, device: torch.device, samples: int = 10) -> float:
    """Measure the execution time of the SR model inference."""
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    logger.info(f"Warming up SR model for input_shape={input_shape}...")
    with torch.no_grad():
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    logger.info(f"Running {samples} samples for SR model...")
    for _ in range(samples):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    return avg_time


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    scene = setup_scene()

    # Experiment parameters
    coverage_size = 500.0
    samples = 5
    base_cell_size = 3.0
    scales = [2, 3, 4]

    results = []

    # Measure base resolution solver once
    logger.info(f"--- Measuring Base Solver ({base_cell_size}m) ---")
    time_base, pixels_base = measure_solver(
        scene, cell_size=base_cell_size, coverage_size=coverage_size, samples=samples
    )

    # Pre-calculate SR input shape (constant for all scales if base is constant)
    input_dim = int(coverage_size / base_cell_size)
    input_shape = (1, 1, input_dim, input_dim)

    for scale in scales:
        target_res = base_cell_size / scale
        logger.info(f"--- Experiment Scale {scale}x: Target {target_res:.2f}m ---")

        # 1. HR Solver
        time_hr, pixels_hr = measure_solver(scene, cell_size=target_res, coverage_size=coverage_size, samples=samples)
        results.append(
            {
                "Target": f"{target_res:.2f}m",
                "Method": f"HR Solver ({target_res:.2f}m)",
                "Time": time_hr,
                "Pixels": pixels_hr,
            }
        )

        # 2. LR Solver + SR
        # We already have time_base

        # Measure SR model
        model = PANLightningModule(scale=scale)
        time_sr = measure_sr(model, input_shape, device, samples=samples)

        total_time = time_base + time_sr
        results.append(
            {
                "Target": f"{target_res:.2f}m",
                "Method": f"LR Solver ({base_cell_size}m) + SR({scale}x)",
                "Time": total_time,
                "Pixels": pixels_hr,  # Target pixels
                "Details": f"Solver({pixels_base}px): {time_base:.4f}s, SR: {time_sr:.4f}s",
            }
        )

    # Print Results Table
    print("\n" + "=" * 100)
    print(f"{'Target Resolution':<20} | {'Method':<30} | {'Pixels':<10} | {'Time (s)':<15} | {'Details'}")
    print("-" * 100)
    for res in results:
        details = res.get("Details", "")
        print(
            f"{res['Target']:<20} | {res['Method']:<30} | {res['Pixels']:<10} | {res['Time']:.4f}          | {details}"
        )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
