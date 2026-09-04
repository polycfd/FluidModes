"""Regenerate the two curated jet-SPOD figures after comparison is complete."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import EXAMPLE_DIRECTORY, DEFAULT_MAT_PATH, load_jet_data
from reference_spod import SELECTED_FREQUENCY_INDICES
from run_fluidmodes import SPOD_RESULTS


PLOTS_DIRECTORY = EXAMPLE_DIRECTORY / "plots"
REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "reference_spod.npz"


def main(mat_path=DEFAULT_MAT_PATH) -> None:
    """Plot reference/FluidModes spectra and FluidModes real mode components."""
    _, x, r, _, _ = load_jet_data(mat_path)
    reference = np.load(REFERENCE_PATH)
    fluidmodes_eigenvalues = np.load(SPOD_RESULTS / "eigenvalues.npy")
    fluidmodes_modes = np.load(SPOD_RESULTS / "modes.npy")
    frequencies = reference["frequencies"]
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for mode in range(4):
        axis.loglog(frequencies[1:], reference["eigenvalues"][1:, mode], "-", label=f"independent reference mode {mode + 1}" if mode < 2 else None)
        axis.loglog(frequencies[1:], fluidmodes_eigenvalues[1:, mode] / float(reference["scale_to_pyspod"]), "--", label=f"FluidModes mode {mode + 1} (converted)" if mode < 2 else None)
    axis.set(xlabel="frequency [cycles per input-time unit]", ylabel="SPOD modal energy", title="Mach 0.9 jet: unweighted batch-SPOD spectrum")
    axis.legend(fontsize="small")
    figure.savefig(PLOTS_DIRECTORY / "spod_spectrum.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    for row, frequency_index in enumerate(SELECTED_FREQUENCY_INDICES):
        for mode in range(2):
            field = fluidmodes_modes[frequency_index, :, mode].reshape(x.shape, order="C").real
            amplitude = np.max(np.abs(field))
            contour = axes[row, mode].contourf(x, r, field, levels=21, cmap="RdBu_r", vmin=-amplitude, vmax=amplitude)
            axes[row, mode].set(xlabel="x", ylabel="r", xlim=(0, 10), ylim=(0, 2), title=f"FluidModes real part: f = {frequencies[frequency_index]:.5f}, mode {mode + 1}")
            axes[row, mode].set_aspect("equal", adjustable="box")
            figure.colorbar(contour, ax=axes[row, mode], shrink=0.82, label="real mode amplitude")
    figure.savefig(PLOTS_DIRECTORY / "spod_modes.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", nargs="?", type=Path, default=DEFAULT_MAT_PATH)
    args = parser.parse_args()
    main(args.mat_path)
