"""Plots of SPOD spectral energies and complex spatial structures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fluidmodes.spod import SPODResult


def plot_eigenvalue_spectrum(result: SPODResult, path: str | Path) -> None:
    """Save the leading SPOD spectral energy versus frequency."""
    figure, axis = plt.subplots()
    leading_eigenvalues = result.eigenvalues[:, 0]
    axis.plot(result.frequencies, leading_eigenvalues)
    if np.all(leading_eigenvalues > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel("Frequency [cycles / input-time unit]")
    axis.set_ylabel("Leading SPOD eigenvalue")
    axis.set_title("Leading SPOD spectral energy")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_spatial_mode(
    result: SPODResult, frequency: float, mode_number: int, path: str | Path
) -> float:
    """Save one complex SPOD mode at the nearest available frequency bin.

    A requested continuous frequency is mapped to its nearest discrete Welch
    bin because SPOD modes exist only at the block's resolved frequencies.
    """
    if not np.isfinite(frequency):
        raise ValueError("Requested SPOD plot frequency must be finite.")
    if not isinstance(mode_number, int) or not 1 <= mode_number <= result.modes.shape[2]:
        raise ValueError(f"Mode number must be between 1 and {result.modes.shape[2]}.")
    frequency_index = int(np.argmin(np.abs(result.frequencies - frequency)))
    selected_frequency = float(result.frequencies[frequency_index])
    mode = result.modes[frequency_index, :, mode_number - 1]

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
        axis.set_title(f"SPOD mode {mode_number}, {part} part\nf = {selected_frequency:.6g}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return selected_frequency
