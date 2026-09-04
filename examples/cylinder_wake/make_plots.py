"""Regenerate the curated cylinder-wake figures from prepared/reference/results data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_results import COMPARISON_PATH
from prepare_data import EXAMPLE_DIRECTORY, load_cylinder_data, mirror_vorticity_snapshot
from run_fluidmodes import DMD_RESULTS


PLOTS_DIRECTORY = EXAMPLE_DIRECTORY / "plots"
POD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"


def plot_field(axis: plt.Axes, field: np.ndarray, title: str) -> None:
    """Plot a vorticity field in its MATLAB spatial ordering."""
    maximum = np.max(np.abs(field))
    image = axis.imshow(field, origin="lower", cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    axis.set(title=title, xlabel="streamwise grid index", ylabel="cross-stream grid index")
    plt.colorbar(image, ax=axis, shrink=0.8, label="vorticity")


def main() -> None:
    """Create figures illustrating raw data, reference POD, and FluidModes DMD."""
    snapshots, spatial_shape, _ = load_cylinder_data()
    pod_reference = np.load(POD_REFERENCE)
    dmd_eigenvalues = np.load(DMD_RESULTS / "eigenvalues.npy")
    dmd_frequencies = np.loadtxt(DMD_RESULTS / "eigenvalues.csv", delimiter=",", skiprows=1)[:, 4]
    dmd_modes = np.load(DMD_RESULTS / "modes.npy")
    shedding_index = json.loads(COMPARISON_PATH.read_text())["shedding_mode_index"]
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    snapshot_index = min(100, snapshots.shape[1] - 1)
    raw_field = snapshots[:, snapshot_index].reshape(spatial_shape, order="F")
    mirrored_field = mirror_vorticity_snapshot(snapshots[:, snapshot_index], spatial_shape).reshape(spatial_shape, order="F")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_field(axes[0], raw_field, f"Raw vorticity snapshot {snapshot_index}")
    plot_field(axes[1], mirrored_field, "POD mirror: cross-stream reflection and sign reversal")
    figure.savefig(PLOTS_DIRECTORY / "pod_augmentation.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.semilogy(np.arange(1, 21), pod_reference["energy_fractions"][:20], "o-")
    axis.set(xlabel="POD mode", ylabel="modal energy fraction", title="Leading POD energy spectrum")
    axis.grid(True, which="both", alpha=0.3)
    figure.savefig(PLOTS_DIRECTORY / "pod_energy_spectrum.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for number, axis in enumerate(axes.flat, start=1):
        plot_field(axis, pod_reference["modes"][:, number - 1].reshape(spatial_shape, order="F"), f"Reference POD mode {number}")
    figure.savefig(PLOTS_DIRECTORY / "pod_leading_modes.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.scatter(dmd_eigenvalues.real, dmd_eigenvalues.imag, s=35, label="FluidModes DMD")
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(np.cos(angle), np.sin(angle), "k--", alpha=0.5, label="unit circle")
    axis.set(xlabel="real($\\lambda$)", ylabel="imaginary($\\lambda$)", title="FluidModes rank-21 DMD eigenvalues")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.savefig(PLOTS_DIRECTORY / "dmd_eigenvalues.png", dpi=160)
    plt.close(figure)

    shedding_mode = dmd_modes[:, shedding_index]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plot_field(axes[0], shedding_mode.real.reshape(spatial_shape, order="F"), f"FluidModes DMD shedding mode: real part (f = {dmd_frequencies[shedding_index]:.5f})")
    plot_field(axes[1], shedding_mode.imag.reshape(spatial_shape, order="F"), f"FluidModes DMD shedding mode: imaginary part (f = {dmd_frequencies[shedding_index]:.5f})")
    figure.savefig(PLOTS_DIRECTORY / "dmd_shedding_mode.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
