"""Standard exact dynamic mode decomposition using PyDMD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydmd import DMD

from fluidmodes.preprocessing import subtract_temporal_mean
from fluidmodes.snapshots import SnapshotData, uniform_time_step


@dataclass(frozen=True)
class DMDResult:
    """Results of standard exact DMD for one uniformly sampled snapshot matrix.

    ``modes`` has shape ``(n_dof, rank)`` and ``dynamics`` has shape
    ``(rank, n_time)``.  Complex-valued quantities are retained because DMD
    modes and their finite-rank reconstruction are generally complex.
    """

    eigenvalues: NDArray[np.complex128]
    continuous_eigenvalues: NDArray[np.complex128]
    growth_rates: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    modes: NDArray[np.complex128]
    amplitudes: NDArray[np.complex128]
    dynamics: NDArray[np.complex128]
    reconstruction: NDArray[np.complex128]
    rank: int
    time: NDArray[np.float64]
    mean: NDArray[np.float64] | None
    spatial_shape: tuple[int, ...] | None


def compute_dmd(
    data: SnapshotData, *, rank: int | None = None, subtract_mean: bool = False
) -> DMDResult:
    """Fit exact DMD to successive, uniformly sampled snapshot columns.

    FluidModes passes ``X[n_dof, n_time]`` directly in PyDMD's expected
    orientation. ``rank`` selects the SVD truncation used to fit the exact-DMD
    map from ``X[:, :-1]`` to ``X[:, 1:]``. PyDMD returns discrete eigenvalues
    ``λ``; FluidModes reports ``ω = log(λ) / dt``, with growth rate
    ``Re(ω)`` and signed cyclic frequency ``Im(ω)/(2π)``. Complex quantities
    are retained because conjugate modal pairs represent real oscillations.
    Mean subtraction is optional and disabled by default because it changes
    the dynamical system being approximated.
    """
    if not isinstance(subtract_mean, (bool, np.bool_)):
        raise ValueError("subtract_mean must be a boolean.")

    time_step = uniform_time_step(data.t, analysis_name="DMD")
    max_rank = min(data.X.shape[0], data.X.shape[1] - 1)
    retained_rank = _validate_rank(rank, max_rank)

    if subtract_mean:
        mean_result = subtract_temporal_mean(data)
        analyzed = mean_result.data.X
        mean: NDArray[np.float64] | None = mean_result.mean
    else:
        analyzed = data.X
        mean = None

    # PyDMD uses ``-1`` for no SVD truncation; ``0`` instead requests automatic
    # rank selection. Exact modes are lifted from the reduced operator to the
    # full snapshot space.
    model = DMD(
        svd_rank=-1 if rank is None else retained_rank,
        exact=True,
        opt=False,
        tlsq_rank=0,
    )
    model.fit(analyzed)

    eigenvalues = np.asarray(model.eigs, dtype=complex)
    if np.any(eigenvalues == 0):
        raise ValueError(
            "DMD produced a zero discrete eigenvalue. A zero eigenvalue has no finite "
            "continuous-time logarithm, so its growth rate and frequency are undefined. "
            "Reduce the requested DMD rank or inspect the input data."
        )
    # A discrete zero has no finite complex logarithm, so it was rejected above.
    continuous_eigenvalues = np.log(eigenvalues) / time_step
    growth_rates = np.asarray(continuous_eigenvalues.real, dtype=float)
    frequencies = np.asarray(continuous_eigenvalues.imag / (2.0 * np.pi), dtype=float)

    modes = np.asarray(model.modes, dtype=complex)
    amplitudes = np.asarray(model.amplitudes, dtype=complex)
    dynamics = np.asarray(model.dynamics, dtype=complex)
    analyzed_reconstruction = np.asarray(model.reconstructed_data, dtype=complex)
    reconstruction = analyzed_reconstruction if mean is None else analyzed_reconstruction + mean[:, None]

    return DMDResult(
        eigenvalues=eigenvalues,
        continuous_eigenvalues=continuous_eigenvalues,
        growth_rates=growth_rates,
        frequencies=frequencies,
        modes=modes,
        amplitudes=amplitudes,
        dynamics=dynamics,
        reconstruction=reconstruction,
        rank=eigenvalues.size,
        time=data.t.copy(),
        mean=mean,
        spatial_shape=data.spatial_shape,
    )


def _validate_rank(rank: int | None, max_rank: int) -> int:
    if rank is None:
        return max_rank
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)):
        raise ValueError("DMD rank must be an integer.")
    if rank < 1 or rank > max_rank:
        raise ValueError(f"DMD rank must be between 1 and {max_rank}.")
    return int(rank)
