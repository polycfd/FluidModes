"""Plots of POD energy, spatial structures, and temporal coefficients."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fluidmodes.pod import PODResult


def plot_energy_spectrum(result: PODResult, path: str | Path) -> None:
    """Save modal energy fraction versus one-based POD mode number."""
    figure, axis = plt.subplots()
    mode_numbers = np.arange(1, result.rank + 1)
    axis.plot(mode_numbers, result.energy_fractions, marker="o")
    axis.set_xlabel("Mode number")
    axis.set_ylabel("Modal energy fraction")
    axis.set_xticks(mode_numbers)
    axis.set_title("POD modal energy spectrum")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_spatial_mode(result: PODResult, mode_number: int, path: str | Path) -> None:
    """Save one spatial POD mode as an image or DOF-index line plot."""
    mode = _mode(result, mode_number)
    figure, axis = plt.subplots()
    if result.spatial_shape is not None and len(result.spatial_shape) == 2:
        image = axis.imshow(mode.reshape(result.spatial_shape), origin="lower", aspect="auto")
        figure.colorbar(image, ax=axis, label="Mode amplitude")
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
    else:
        axis.plot(np.arange(mode.size), mode)
        axis.set_xlabel("Degree-of-freedom index")
        axis.set_ylabel("Mode amplitude")
    axis.set_title(f"POD mode {mode_number}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_temporal_coefficient(result: PODResult, mode_number: int, path: str | Path) -> None:
    """Save the coefficient multiplying one spatial POD mode versus time."""
    coefficient = _mode(result, mode_number, result.coefficients)
    figure, axis = plt.subplots()
    axis.plot(result.time, coefficient)
    axis.set_xlabel("Time")
    axis.set_ylabel("Temporal coefficient")
    axis.set_title(f"POD coefficient {mode_number}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _mode(
    result: PODResult, mode_number: int, values: np.ndarray | None = None
) -> np.ndarray:
    if not isinstance(mode_number, int) or not 1 <= mode_number <= result.rank:
        raise ValueError(f"Mode number must be between 1 and {result.rank}.")
    matrix = result.modes if values is None else values
    return matrix[:, mode_number - 1] if values is None else matrix[mode_number - 1]
