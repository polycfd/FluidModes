"""Plots of DMD eigenvalues and complex spatial modes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fluidmodes.dmd import DMDResult


def plot_eigenvalues(result: DMDResult, path: str | Path) -> None:
    """Save discrete DMD eigenvalues with the neutral-magnitude unit circle."""
    figure, axis = plt.subplots()
    angle = np.linspace(0.0, 2.0 * np.pi, 256)
    axis.plot(np.cos(angle), np.sin(angle), color="0.6", linestyle="--", label="Unit circle")
    axis.scatter(result.eigenvalues.real, result.eigenvalues.imag)
    axis.set_xlabel("Real part")
    axis.set_ylabel("Imaginary part")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("DMD discrete eigenvalues")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_spatial_mode(result: DMDResult, mode_number: int, path: str | Path) -> None:
    """Save quadrature real and imaginary components of one complex DMD mode."""
    mode = _mode(result, mode_number)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, values, part in zip(axes, (mode.real, mode.imag), ("real", "imaginary"), strict=True):
        if result.spatial_shape is not None and len(result.spatial_shape) == 2:
            image = axis.imshow(values.reshape(result.spatial_shape), origin="lower", aspect="auto")
            figure.colorbar(image, ax=axis, label="Mode amplitude")
            axis.set_xlabel("x index")
            axis.set_ylabel("y index")
        else:
            axis.plot(np.arange(values.size), values)
            axis.set_xlabel("Degree-of-freedom index")
            axis.set_ylabel("Mode amplitude")
        axis.set_title(f"DMD mode {mode_number}: {part} part")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _mode(result: DMDResult, mode_number: int) -> np.ndarray:
    if not isinstance(mode_number, int) or not 1 <= mode_number <= result.rank:
        raise ValueError(f"Mode number must be between 1 and {result.rank}.")
    return result.modes[:, mode_number - 1]
