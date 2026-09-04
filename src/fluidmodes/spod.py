"""Standard serial batch SPOD using PySPOD."""

from __future__ import annotations

import contextlib
import io
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyspod.spod.standard import Standard

from fluidmodes.preprocessing import subtract_temporal_mean
from fluidmodes.snapshots import SnapshotData, uniform_time_step


@dataclass(frozen=True)
class SPODResult:
    """Results of standard one-sided SPOD for a real snapshot matrix.

    ``eigenvalues`` has shape ``(n_frequency, n_blocks)``. ``modes`` has
    shape ``(n_frequency, n_dof, n_modes_saved)``; modes are complex and have
    an arbitrary global phase at each frequency. ``mean`` is ``None`` when
    long-time mean subtraction was disabled.
    """

    frequencies: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    modes: NDArray[np.complex128]
    block_size: int
    overlap: float
    n_blocks: int
    time_step: float
    mean: NDArray[np.float64] | None
    spatial_shape: tuple[int, ...] | None


def compute_spod(
    data: SnapshotData,
    *,
    block_size: int,
    overlap: float = 0.5,
    modes: int = 1,
    subtract_mean: bool = True,
) -> SPODResult:
    """Compute one-sided standard SPOD of uniformly sampled real snapshots.

    ``block_size`` is the number of snapshots in each Welch block, giving
    frequency resolution ``1 / (block_size * dt)``. A Hamming window reduces
    block-edge leakage. FluidModes analyzes fluctuations about the long-time
    mean by default. It converts its column-per-snapshot ``X[n_dof, n_time]``
    convention to PySPOD's time-first, single-variable layout; real input
    produces a one-sided non-negative frequency spectrum.
    """
    if not isinstance(subtract_mean, (bool, np.bool_)):
        raise ValueError("subtract_mean must be a boolean.")

    time_step = uniform_time_step(data.t, analysis_name="SPOD")
    n_blocks = _validate_spectral_settings(data.t.size, block_size, overlap, modes)

    if subtract_mean:
        # PySPOD removes this long-time mean internally; retain it for output
        # without centering the same snapshots a second time.
        mean = subtract_temporal_mean(data).mean
        mean_type = "longtime"
    else:
        mean = None
        mean_type = "zero"

    # PySPOD takes time first and a trailing variable dimension. Preserve image
    # dimensions when available; tabular data has one spatial dimension.
    if data.spatial_shape is not None and len(data.spatial_shape) == 2:
        backend_data = data.X.T.reshape((data.t.size, *data.spatial_shape, 1))
        n_space_dims = 2
    else:
        backend_data = data.X.T[..., None]
        n_space_dims = 1

    # FluidModes exposes overlap as a fraction, while PySPOD expects a percent.
    overlap_percent = 100.0 * float(overlap)
    # PySPOD persists standard-SPOD modes to disk. Read the required modes
    # back from isolated temporary storage and return FluidModes' convention.
    with tempfile.TemporaryDirectory(prefix="fluidmodes-spod-") as temporary_directory:
        # Standard PySPOD uses its Hamming window when no window is supplied.
        # ``weights=None`` selects uniform spatial weighting; ``comm=None`` is
        # the serial execution path.
        params = {
            "n_dft": int(block_size),
            "time_step": time_step,
            "n_space_dims": n_space_dims,
            "n_variables": 1,
            "overlap": overlap_percent,
            "mean_type": mean_type,
            "fullspectrum": False,
            "normalize_weights": False,
            "normalize_data": False,
            "n_modes_save": int(modes),
            "dtype": "double",
            "savedir": temporary_directory,
            "savefreq_disk": True,
        }
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.filterwarnings(
                "ignore",
                message="No mean subtracted. Consider using longtime mean.",
                category=UserWarning,
                module="pyspod.spod.base",
            )
            warnings.filterwarnings(
                "ignore",
                message="Parameter `weights` not equal to a `numpy.ndarray`.Using default uniform weighting",
                category=UserWarning,
                module="pyspod.spod.base",
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in sqrt",
                category=RuntimeWarning,
                module="pyspod.spod.standard",
            )
            model = Standard(params=params, weights=None, comm=None).fit([backend_data])

        frequencies = np.asarray(model.freq, dtype=float)
        eigenvalues = np.asarray(model.eigs, dtype=float)
        saved_modes = np.empty((frequencies.size, data.X.shape[0], int(modes)), dtype=complex)
        for frequency_index in range(frequencies.size):
            backend_modes = np.asarray(model.get_modes_at_freq(frequency_index), dtype=complex)
            # Flatten spatial dimensions back to FluidModes' DOF row ordering.
            saved_modes[frequency_index] = backend_modes.reshape(data.X.shape[0], int(modes))

    return SPODResult(
        frequencies=frequencies,
        eigenvalues=eigenvalues,
        modes=saved_modes,
        block_size=int(block_size),
        overlap=float(overlap),
        n_blocks=n_blocks,
        time_step=time_step,
        mean=mean,
        spatial_shape=data.spatial_shape,
    )


def _validate_spectral_settings(
    n_snapshots: int, block_size: int, overlap: float, modes: int
) -> int:
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, (bool, np.bool_)):
        raise ValueError("SPOD block size must be an integer.")
    if block_size < 4:
        raise ValueError("SPOD block size must be at least 4 snapshots.")
    if block_size > n_snapshots:
        raise ValueError("SPOD block size must not exceed the available number of snapshots.")
    if not isinstance(overlap, (int, float, np.integer, np.floating)) or isinstance(overlap, (bool, np.bool_)):
        raise ValueError("SPOD overlap must be a fraction from 0 (inclusive) to 1 (exclusive).")
    overlap_fraction = float(overlap)
    if not np.isfinite(overlap_fraction) or not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("SPOD overlap must be a fraction from 0 (inclusive) to 1 (exclusive).")

    n_overlap = int(np.ceil(int(block_size) * overlap_fraction))
    n_blocks = (n_snapshots - n_overlap) // (int(block_size) - n_overlap)
    if n_blocks < 2:
        raise ValueError("SPOD block size and overlap must produce at least two blocks.")
    if not isinstance(modes, (int, np.integer)) or isinstance(modes, (bool, np.bool_)):
        raise ValueError("Number of SPOD modes must be an integer.")
    if modes < 1 or modes > n_blocks:
        raise ValueError(f"Number of SPOD modes must be between 1 and {n_blocks}.")
    return int(n_blocks)
