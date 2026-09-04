"""Make compact physical PIV POD/DMD validation figures."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_results import COMPARISON_PATH
from prepare_data import EXAMPLE_DIRECTORY, SPATIAL_SHAPE, load_velocity_snapshots
from run_fluidmodes import DMD_RESULTS


PLOTS_DIRECTORY = EXAMPLE_DIRECTORY / "plots"
POD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"
LITERATURE_WAKE_FREQUENCY = 0.889


def _field(mode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_field_dof = int(np.prod(SPATIAL_SHAPE))
    return (
        mode[:n_field_dof].reshape(SPATIAL_SHAPE, order="C"),
        mode[n_field_dof:].reshape(SPATIAL_SHAPE, order="C"),
    )


def _vorticity(mode: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, v = _field(mode)
    return np.gradient(v, x[:, 0], axis=0) - np.gradient(u, y[0, :], axis=1)


def _plot_scalar(axis: plt.Axes, x: np.ndarray, y: np.ndarray, scalar: np.ndarray, title: str, label: str) -> None:
    limit = float(np.nanmax(np.abs(scalar)))
    image = axis.pcolormesh(x.T, y.T, scalar.T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
    plt.colorbar(image, ax=axis, shrink=0.82, label=label)


def main() -> None:
    """Visualize raw experimental data, POD energy/pair, and DMD shedding mode."""
    snapshots, x, y, _ = load_velocity_snapshots()
    pod = np.load(POD_REFERENCE)
    comparison = json.loads(COMPARISON_PATH.read_text())
    dmd_table = np.loadtxt(DMD_RESULTS / "eigenvalues.csv", delimiter=",", skiprows=1)
    dmd_modes = np.load(DMD_RESULTS / "modes.npy")
    PLOTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    snapshot_u, snapshot_v = _field(snapshots[:, min(100, snapshots.shape[1] - 1)])
    magnitude = np.hypot(snapshot_u, snapshot_v)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    image = axis.pcolormesh(x.T, y.T, magnitude.T, shading="auto", cmap="viridis")
    axis.set(xlabel="x (m)", ylabel="y (m)", title="Experimental PIV velocity magnitude (snapshot 101)", aspect="equal")
    plt.colorbar(image, ax=axis, label="velocity magnitude (m s$^{-1}$)")
    figure.savefig(PLOTS_DIRECTORY / "experimental_snapshot.png", dpi=160)
    plt.close(figure)

    energy = pod["energy_fractions"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    modes = np.arange(1, 21)
    axes[0].semilogy(modes, energy[:20], "o-")
    axes[0].axvspan(0.75, 2.25, color="C1", alpha=0.15, label="leading pair")
    axes[0].axvline(12, color="k", linestyle="--", label="mode 12")
    axes[0].set(xlabel="POD mode", ylabel="energy fraction", title="POD energy fractions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(1, energy.size + 1), np.cumsum(energy), "o-", markevery=20)
    axes[1].axvline(12, color="k", linestyle="--")
    axes[1].axhline(0.75, color="C3", linestyle=":", label="paper context: 75%")
    axes[1].set(xlabel="POD mode", ylabel="cumulative energy", xlim=(0, 30), ylim=(0, 1.02), title="Cumulative POD energy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.savefig(PLOTS_DIRECTORY / "pod_energy.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, axis in enumerate(axes):
        _plot_scalar(axis, x, y, _vorticity(pod["modes"][:, index], x, y), f"POD mode {index + 1} vorticity (sign arbitrary)", "$\\omega_z$ (visualization only)")
    figure.savefig(PLOTS_DIRECTORY / "pod_leading_pair.png", dpi=160)
    plt.close(figure)

    wake_index = comparison["fluidmodes_wake_mode_index_one_based"] - 1
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    positive = dmd_table[:, 4] > 0.0
    axes[0].stem(dmd_table[positive, 4], dmd_table[positive, 5], basefmt=" ")
    axes[0].axvline(LITERATURE_WAKE_FREQUENCY, color="C3", linestyle="--", label="literature 0.889 Hz")
    axes[0].axvline(dmd_table[wake_index, 4], color="C1", linestyle=":", label="selected coherent pair")
    axes[0].set(xlabel="frequency (Hz)", ylabel="FluidModes |amplitude|", title="Rank-12 DMD frequency representation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    _plot_scalar(axes[1], x, y, _vorticity(dmd_modes[:, wake_index].real, x, y), f"DMD mode near {dmd_table[wake_index, 4]:.3f} Hz", "$\\omega_z$ (real part; visualization only)")
    figure.savefig(PLOTS_DIRECTORY / "dmd_shedding.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
