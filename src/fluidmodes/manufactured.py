"""Deterministic manufactured datasets for modal-analysis verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fluidmodes.snapshots import SnapshotData


@dataclass(frozen=True)
class PODCase:
    """Three orthogonal spatial modes with prescribed POD energy fractions."""

    data: SnapshotData
    spatial_modes: NDArray[np.float64]
    temporal_coefficients: NDArray[np.float64]
    relative_energies: NDArray[np.float64]
    expected_rank: int


@dataclass(frozen=True)
class DMDCase:
    """Real snapshots from conjugate eigenvalue pairs with known dynamics."""

    data: SnapshotData
    modes: NDArray[np.complex128]
    amplitudes: NDArray[np.complex128]
    frequencies: NDArray[np.float64]
    growth_rates: NDArray[np.float64]
    eigenvalues: NDArray[np.complex128]
    time_step: float


@dataclass(frozen=True)
class SPODCase:
    """Coherent standing-wave structures at two prescribed frequencies."""

    data: SnapshotData
    spatial_modes: NDArray[np.float64]
    frequencies: NDArray[np.float64]


def make_pod_case() -> PODCase:
    """Create a POD case with three orthogonal modes at 0.6, 0.3, and 0.1 energy."""
    n_time = 60
    t = np.arange(n_time, dtype=float) / 10.0
    spatial_modes = np.eye(4, 3)
    relative_energies = np.array([0.6, 0.3, 0.1])
    frequencies = np.array([1.0, 2.0, 3.0])
    # Integer numbers of cycles make these temporal rows mutually orthogonal;
    # the squared amplitudes therefore set the prescribed POD energies.
    temporal_coefficients = np.sqrt(relative_energies)[:, None] * np.sin(
        2.0 * np.pi * frequencies[:, None] * np.arange(n_time) / n_time
    )
    data = SnapshotData(
        X=spatial_modes @ temporal_coefficients,
        t=t,
        labels=("q1", "q2", "q3", "q4"),
        time_label="time",
        spatial_shape=(2, 2),
    )
    return PODCase(data, spatial_modes, temporal_coefficients, relative_energies, expected_rank=3)


def make_dmd_case() -> DMDCase:
    """Create real snapshots with prescribed frequencies and growth rates."""
    time_step = 0.05
    n_time = 80
    t = np.arange(n_time, dtype=float) * time_step
    frequencies = np.array([1.5, -1.5, 3.5, -3.5])
    growth_rates = np.array([-0.10, -0.10, 0.05, 0.05])
    # λ = exp[(σ + i 2πf) Δt] gives the exact discrete-time evolution.
    eigenvalues = np.exp((growth_rates + 2j * np.pi * frequencies) * time_step)
    modes = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1j, 1j, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1j, 1j],
        ],
        dtype=complex,
    ) / np.sqrt(2.0)
    amplitudes = np.array([0.8 + 0.3j, 0.8 - 0.3j, 0.5 - 0.2j, 0.5 + 0.2j])
    dynamics = amplitudes[:, None] * eigenvalues[:, None] ** np.arange(n_time)
    X = np.real_if_close(modes @ dynamics).real
    data = SnapshotData(X=X, t=t, labels=("q1", "q2", "q3", "q4"), time_label="time")
    return DMDCase(data, modes, amplitudes, frequencies, growth_rates, eigenvalues, time_step)


def make_spod_case() -> SPODCase:
    """Create coherent noise-free standing waves at 5 and 12 cycles per time unit."""
    n_dof = 16
    time_step = 0.01
    n_time = 200
    coordinate = 2.0 * np.pi * np.arange(n_dof) / n_dof
    t = np.arange(n_time, dtype=float) * time_step
    spatial_modes = np.column_stack((np.sin(coordinate), np.cos(2.0 * coordinate)))
    spatial_modes /= np.linalg.norm(spatial_modes, axis=0)
    frequencies = np.array([5.0, 12.0])
    # These coherent standing waves fall exactly on the 1-cycle-per-time-unit
    # bins used by the manufactured SPOD test.
    coefficients = np.array([1.0, 0.6])[:, None] * np.cos(2.0 * np.pi * frequencies[:, None] * t)
    data = SnapshotData(
        X=spatial_modes @ coefficients,
        t=t,
        labels=tuple(f"q{index + 1}" for index in range(n_dof)),
        time_label="time",
    )
    return SPODCase(data, spatial_modes, frequencies)
