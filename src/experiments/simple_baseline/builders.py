from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import sionna
from sionna.rt import PlanarArray, Transmitter, Receiver
from typing import Optional, Sequence, List, Tuple


@dataclass(frozen=True)
class TransceiverConfig:
    n_tx: int
    n_rx: int
    # Explicit positions (list of [x,y,z]); if None -> random (if range provided)
    tx_positions: Optional[Sequence[Sequence[float]]] = None
    rx_positions: Optional[Sequence[Sequence[float]]] = None
    # Ranges for random placement: ((xmin,xmax),(ymin,ymax), z_fixed)
    random_tx_range: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], float]
    ] = None
    random_rx_range: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], float]
    ] = None
    # Antenna / power parameters
    tx_power_dbm: float = 44.0
    tx_array_pattern: str = "tr38901"
    rx_array_pattern: Optional[str] = None  # Defaults to tx_array_pattern if None
    polarization: str = "V"
    # Pointing [x, y, z] / orientation (either one global look_at or per-TX list)
    look_at: Optional[Sequence[float]] = None
    per_tx_look_at: Optional[Sequence[Sequence[float]]] = None
    # Naming + reproducibility
    base_tx_name: str = "tx"
    base_rx_name: str = "rx"
    starting_index: int = 1
    seed: Optional[int] = None
    # Planar array size
    num_rows: int = 1
    num_cols: int = 1

    def validate(self) -> TransceiverConfig:
        if self.n_tx <= 0 or self.n_rx <= 0:
            raise ValueError("n_tx and n_rx must be > 0.")
        if self.tx_positions is not None and len(self.tx_positions) != self.n_tx:
            raise ValueError("Length of tx_positions must equal n_tx.")
        if self.rx_positions is not None and len(self.rx_positions) != self.n_rx:
            raise ValueError("Length of rx_positions must equal n_rx.")
        if self.per_tx_look_at is not None and len(self.per_tx_look_at) != self.n_tx:
            raise ValueError("Length of per_tx_look_at must equal n_tx.")
        if self.tx_positions is None and self.random_tx_range is None:
            raise ValueError("Provide tx_positions or random_tx_range.")
        if self.rx_positions is None and self.random_rx_range is None:
            raise ValueError("Provide rx_positions or random_rx_range.")
        return self

    @staticmethod
    def positioned(
        *,
        n_tx: int,
        n_rx: int,
        tx_positions: Sequence[Sequence[float]],
        rx_positions: Sequence[Sequence[float]],
        **kwargs,
    ) -> TransceiverConfig:
        """
        Factory for explicit positions.
        """
        cfg = TransceiverConfig(
            n_tx=n_tx,
            n_rx=n_rx,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            **kwargs,
        )
        return cfg.validate()

    @staticmethod
    def randomized(
        *,
        n_tx: int,
        n_rx: int,
        random_tx_range: Tuple[Tuple[float, float], Tuple[float, float], float],
        random_rx_range: Tuple[Tuple[float, float], Tuple[float, float], float],
        **kwargs,
    ) -> TransceiverConfig:
        """
        Factory for randomized placement.
        """
        cfg = TransceiverConfig(
            n_tx=n_tx,
            n_rx=n_rx,
            random_tx_range=random_tx_range,
            random_rx_range=random_rx_range,
            **kwargs,
        )
        return cfg.validate()


class SceneTransceiverBuilder:
    """
    Single-responsibility helper: clears previous TX/RX (matching naming pattern)
    and (re)creates them on an existing scene based on a TransceiverConfig.
    """

    def __init__(self, scene: "sionna.rt.Scene"):
        self.scene = scene

    def _safe_remove(self, name: str):
        try:
            self.scene.remove(name)
        except Exception:
            pass  # Ignore if not present

    def _clear_previous(self, cfg: TransceiverConfig, max_scan: int = 500):
        """
        Clears any previously created transmitters and receivers
        to avoid conflicts when creating new ones.
        """
        for prefix in (cfg.base_tx_name, cfg.base_rx_name):
            for i in range(cfg.starting_index, cfg.starting_index + max_scan):
                self._safe_remove(f"{prefix}_{i}")

    @staticmethod
    def _generate_random_positions(count: int, rng_def: Tuple) -> List[List[float]]:
        (xmin, xmax), (ymin, ymax), z = rng_def
        xs = np.random.uniform(xmin, xmax, size=count)
        ys = np.random.uniform(ymin, ymax, size=count)
        return [[float(x), float(y), float(z)] for x, y in zip(xs, ys)]

    def build(self, config: TransceiverConfig = None, /, **kwargs) -> None:
        """
        Build transceivers on the scene (in-place).
        Either pass a TransceiverConfig or kwargs (must include n_tx, n_rx).
        """
        if config is None:
            if "n_tx" not in kwargs or "n_rx" not in kwargs:
                raise ValueError(
                    "Must provide n_tx and n_rx when not passing a TransceiverConfig."
                )
            config = TransceiverConfig(**kwargs).validate()
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    config = replace(config, **{k: v})
                else:
                    raise ValueError(f"Unknown config field: {k}")
            config.validate()

        if config.seed is not None:
            np.random.seed(config.seed)

        # Clear the scene
        self._clear_previous(config)

        self.scene.tx_array = PlanarArray(
            num_rows=config.num_rows,
            num_cols=config.num_cols,
            pattern=config.tx_array_pattern,
            polarization=config.polarization,
        )
        self.scene.rx_array = PlanarArray(
            num_rows=config.num_rows,
            num_cols=config.num_cols,
            pattern=config.rx_array_pattern or config.tx_array_pattern,
            polarization=config.polarization,
        )

        # Transmitters (TX) positions
        if config.tx_positions is not None:
            tx_positions = [list(p) for p in config.tx_positions]
        else:
            tx_positions = self._generate_random_positions(
                config.n_tx, config.random_tx_range
            )

        # Receivers (RX) positions
        if config.rx_positions is not None:
            rx_positions = [list(p) for p in config.rx_positions]
        else:
            rx_positions = self._generate_random_positions(
                config.n_rx, config.random_rx_range
            )

        # Look-at handling
        if config.per_tx_look_at is not None:
            look_ats: List[Optional[Sequence[float]]] = config.per_tx_look_at
        else:
            look_ats = [config.look_at] * config.n_tx

        # Create TX
        for i, (pos, la) in enumerate(
            zip(tx_positions, look_ats), start=config.starting_index
        ):
            name = f"{config.base_tx_name}_{i}"
            tx_kwargs = dict(
                name=name, position=pos, power_dbm=config.tx_power_dbm, color=(0, 0, 1)
            )
            if la is not None:
                tx_kwargs["look_at"] = la
            self.scene.add(Transmitter(**tx_kwargs))

        # Create RX
        for i, pos in enumerate(rx_positions, start=config.starting_index):
            name = f"{config.base_rx_name}_{i}"
            self.scene.add(Receiver(name=name, position=pos))

        return None
