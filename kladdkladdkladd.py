from sionna.rt import load_scene, RadioMapSolver, Transmitter, PlanarArray, Scene, \
                      DEFAULT_FREQUENCY, DEFAULT_BANDWIDTH, DEFAULT_TEMPERATURE
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c

from tqdm import tqdm

lambda_ = c / DEFAULT_FREQUENCY
step_size = lambda_ / 20
positions = np.arange(1, 2, step_size)

dx = dy = lambda_ / 10.0
GRID_N = 512 * 2
SIZE_X = GRID_N * dx
SIZE_Y = GRID_N * dy


def path_gain_loss_test(scene, positions: np.ndarray) -> np.ndarray:
    rm_solver = RadioMapSolver()

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, polarization='V', pattern='iso')
    scene.remove('tx1')
    scene.remove('tx2')
    tx1 = Transmitter('tx1', position=[0.0, 0.0, 10.0])
    scene.add(tx1)

    results = np.zeros((len(positions), GRID_N, GRID_N), dtype=float)

    for i, pos in tqdm(enumerate(positions), total=len(positions)):
        scene.remove('tx2')
        tx2 = Transmitter('tx2', position=[-float(pos), 0.0, 10.0])
        scene.add(tx2)

        rm = rm_solver(
            scene,
            max_depth=5,
            samples_per_tx=10**6,
            cell_size=(dx, dy),
            center=(0.0, 0.0, 0.0),
            size=(SIZE_X, SIZE_Y),
            orientation=(0.0, 0.0, 0.0)
        )

        results[i] = np.max(rm.path_gain, axis=0)

    return results


def heatmap(z: np.ndarray) -> None:
    """Plot the path gain map using matplotlib."""
    plt.imshow(z, cmap='viridis', origin='lower')
    plt.colorbar(label='Path Gain (dB)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Path Gain Map')
    plt.show()


def plot_movie(res: np.ndarray) -> None:
    # --- after you compute res (T, Ny, Nx) and know size/center ---
    SIZE_X, SIZE_Y = SIZE_X, SIZE_Y         # same as in RadioMapSolver
    CENTER_X, CENTER_Y = 0.0, 0.0           # same center as in solver
    extent = [CENTER_X - SIZE_X/2, CENTER_X + SIZE_X/2,
            CENTER_Y - SIZE_Y/2, CENTER_Y + SIZE_Y/2]

    plt.ion()
    fig, ax = plt.subplots(1, figsize=(8, 7))

    rmin, rmax = np.nanmin(res), np.nanmax(res)
    im0 = ax.imshow(res[0], cmap='viridis', origin='lower',
                    vmin=rmin, vmax=rmax, extent=extent)

    # initial TX positions in world coords
    tx1_xy = np.array([0.0, 0.0])                  # fixed
    tx2_xy = np.array([-float(positions[0]), 0.0]) # moving

    tx1_sc = ax.scatter([tx1_xy[0]], [tx1_xy[1]], c='red', s=80, label='TX1')
    tx2_sc = ax.scatter([tx2_xy[0]], [tx2_xy[1]], c='blue', s=80, label='TX2')
    ax.legend()

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Path Gain')
    fig.tight_layout()

    for i in range(len(positions)):
        # update heatmap
        im0.set_data(res[i])

        # update TX2 position
        tx2_xy = np.array([-float(positions[i]), 0.0])
        tx2_sc.set_offsets(tx2_xy[None, :])   # expects shape (N,2)

        # (if TX1 moved, also call tx1_sc.set_offsets(...))

        # optional: title with center cell value
        pg_center = res[i][res[i].shape[0]//2, res[i].shape[1]//2]
        ax.set_title(f'Path Gain (center): {pg_center:.2e}')

        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.ioff()
    plt.show()


def abs_diff_map(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Per-pixel absolute difference |A - B| with NaN where either input is NaN.
    Works for linear or dB-valued maps. Returns a 2D array.
    """
    A, B = np.asarray(A), np.asarray(B)
    mask = np.isnan(A) | np.isnan(B)
    D = np.abs(A - B)
    D[mask] = np.nan
    return D

def mad_mean(A: np.ndarray, B: np.ndarray) -> float:
    """
    Mean Absolute Difference over valid pixels.
    """
    D = abs_diff_map(A, B)
    return float(np.nanmean(D))

def mad_median(A: np.ndarray, B: np.ndarray) -> float:
    """
    Median Absolute Difference (robust) over valid pixels.
    """
    D = abs_diff_map(A, B)
    return float(np.nanmedian(D))

def nmad_by_range(A: np.ndarray, B: np.ndarray) -> float:
    """
    Normalized MAD = mean |A-B| / range(A,B) over valid pixels.
    Useful when maps have different absolute scales.
    """
    A, B = np.asarray(A), np.asarray(B)
    D = abs_diff_map(A, B)
    # compute global range over valid pixels from both maps
    m = ~np.isnan(A); n = ~np.isnan(B)
    valid_vals = np.concatenate([A[m], B[n]])
    rng = np.nanmax(valid_vals) - np.nanmin(valid_vals)
    if rng == 0 or not np.isfinite(rng):
        return np.nan
    return float(np.nanmean(D) / rng)

def summary_mad(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Convenience wrapper returning a few summaries at once.
    """
    D = abs_diff_map(A, B)
    return {
        "mad_mean": float(np.nanmean(D)),
        "mad_median": float(np.nanmedian(D)),
        "mad_std": float(np.nanstd(D)),
        "valid_fraction": float(np.isfinite(D).mean()),
    }

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import uniform_filter1d

IS_DB = True  # set False if your maps are linear

def map_mad(A, B):
    D = np.abs(A - B)
    D[np.isnan(A) | np.isnan(B)] = np.nan
    return float(np.nanmean(D))

def mad_vs_displacement(res, positions):
    """MAD between consecutive maps and associated Δx."""
    T = len(res)
    mads = [map_mad(res[i], res[i+1]) for i in range(T-1)]
    dx   = np.abs(np.diff(positions))
    return np.asarray(dx), np.asarray(mads)

def centerline_series(res):
    """Extract a 1-D series at the map center (can also average a small stripe)."""
    T, H, W = res.shape
    j = H // 2
    # average a small vertical band for robustness
    band = res[:, max(0, j-2):min(H, j+3), :]
    s = np.nanmean(band, axis=(1,2))  # one value per map
    return s  # shape (T,)

def detrend_small_scale(x, win=11):
    """Remove large-scale trend from a 1D series (linear or dB)."""
    # work in linear for detrending if original is dB
    if IS_DB:
        x_lin = 10**(x/10)
        trend = uniform_filter1d(x_lin, size=win, mode='nearest')
        small = x_lin / np.maximum(trend, 1e-30)
        amp = np.sqrt(small)  # amplitude ~ sqrt(power)
        return amp
    else:
        trend = uniform_filter1d(x, size=win, mode='nearest')
        small = x / np.maximum(trend, 1e-30)
        amp = np.sqrt(np.maximum(small, 0))
        return amp

def spatial_autocorr(series):
    """Normalized autocorrelation for non-NaN 1D series."""
    x = series.copy()
    x = x[~np.isnan(x)]
    x = x - np.mean(x)
    if len(x) < 3: return np.array([1.0])
    c = correlate(x, x, mode='full')
    c = c[c.size//2:]
    c = c / (c[0] + 1e-30)
    return c

def plot_small_scale_suite(res, positions, lambda_):
    # 1) MAD vs displacement
    dx, mads = mad_vs_displacement(res, positions)
    plt.figure(figsize=(5,3))
    plt.plot(dx, mads, 'o-')
    plt.axvline(lambda_/2, color='r', ls='--', label='~ λ/2')
    plt.xlabel('ΔTX (m)'); plt.ylabel('MAD (dB)' if IS_DB else 'MAD (linear)')
    plt.title('MAD between consecutive maps')
    plt.legend(); plt.tight_layout(); plt.show()

    # 2) 1-D time/position series at center and its autocorrelation
    s = centerline_series(res)  # per-position power
    amp = detrend_small_scale(s, win=max(5, len(s)//20))
    # distances from first position
    d = np.abs(positions - positions[0])

    plt.figure(figsize=(6,3))
    plt.plot(d, 20*np.log10(amp+1e-12) if IS_DB else amp, '-')
    plt.xlabel('TX displacement (m)')
    plt.ylabel('Normalized envelope (dB)' if IS_DB else 'Normalized envelope')
    plt.title('Normalized small-scale envelope along TX path')
    plt.tight_layout(); plt.show()

    R = spatial_autocorr(amp)
    dd = np.arange(len(R)) * np.median(np.diff(d))
    plt.figure(figsize=(5,3))
    plt.plot(dd, R, '-')
    plt.axvline(lambda_/2, color='r', ls='--', label='~ λ/2')
    plt.xlabel('Lag distance (m)'); plt.ylabel('Autocorrelation')
    plt.title('Spatial autocorrelation of envelope')
    plt.legend(); plt.tight_layout(); plt.show()

    # 3) Envelope histogram vs Rayleigh (qualitative)
    try:
        from scipy.stats import rayleigh, rice
        a = amp / (np.sqrt(np.mean(amp**2)) + 1e-12)  # normalize RMS=1
        xs = np.linspace(0, np.percentile(a, 99.5), 200)
        plt.figure(figsize=(5,3))
        plt.hist(a, bins=40, density=True, alpha=0.5, label='Measured')
        plt.plot(xs, rayleigh.pdf(xs), 'r--', label='Rayleigh')
        # simple Rician fit parameter (method-of-moments-ish)
        v = max(0.0, np.sqrt(max(0.0, np.mean(a)**2 - (1 - np.pi/4))))
        plt.plot(xs, rice.pdf(xs, b=v), 'g--', label='Rician (rough)')
        plt.ylim(0, 1)
        plt.xlabel('Normalized amplitude'); plt.ylabel('PDF')
        plt.title('Envelope statistics')
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    res = path_gain_loss_test(load_scene(), positions)
    # plot_small_scale_suite(res, positions, lambda_)

    MADS = []
    for i in range(1, len(positions)):
        mad = map_mad(res[i-1], res[i])
        MADS.append(mad)

    # plot movie of MADS without plot_movie function
    plt.ion()
    fig, ax = plt.subplots(1, figsize=(12, 6))
    rmin, rmax = np.nanmin(min(m for m in MADS if m is not None)), np.nanmax(max(m for m in MADS if m is not None))
    im0 = ax.imshow(MADS[0], cmap='viridis', origin='lower',
                    vmin=rmin, vmax=rmax)
    ax.set_title(f'HR Path Gain Map - Position {positions[0]}')
    ax.set_xlabel('X-axis'); ax.set_ylabel('Y-axis')
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar0.set_label('Path Gain')
    fig.tight_layout()
    for i in range(len(positions)):
        im0.set_data(MADS[i])
        hr_pg = MADS[i][MADS[i].shape[0] // 2, MADS[i].shape[1] // 2]
        ax.set_title(f'HR Path Gain Map - Path Gain {hr_pg:.2e} dB')
        plt.pause(0.3)
    plt.ioff()
    plt.show()



    
