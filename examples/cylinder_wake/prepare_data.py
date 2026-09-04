"""Prepare raw-DMD and symmetry-augmented-POD CSV inputs for the cylinder wake."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = EXAMPLE_DIRECTORY / "data"
MAT_PATH = DATA_DIRECTORY / "CYLINDER_ALL.mat"
DMD_CSV_PATH = DATA_DIRECTORY / "cylinder_dmd.csv"
POD_CSV_PATH = DATA_DIRECTORY / "cylinder_pod.csv"
SNAPSHOT_SPACING = 0.2


def load_cylinder_data() -> tuple[np.ndarray, tuple[int, int], dict[str, tuple[tuple[int, ...], str]]]:
    """Load the third-party ``VORTALL[n_dof, n_time]`` snapshots and shape."""
    if not MAT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MAT_PATH}. Download DATA.zip from http://dmdbook.com/DATA.zip "
            "and extract DATA/FLUIDS/CYLINDER_ALL.mat here."
        )
    contents = loadmat(MAT_PATH)
    if not {"VORTALL", "nx", "ny"}.issubset(contents):
        raise ValueError("CYLINDER_ALL.mat must contain VORTALL, nx, and ny.")
    snapshots = np.asarray(contents["VORTALL"], dtype=float)
    spatial_shape = (int(np.asarray(contents["nx"]).squeeze()), int(np.asarray(contents["ny"]).squeeze()))
    if (int(np.asarray(contents["m"]).squeeze()), int(np.asarray(contents["n"]).squeeze())) != spatial_shape:
        raise ValueError("MATLAB m, n dimensions must agree with nx, ny.")
    if snapshots.shape[0] != np.prod(spatial_shape):
        raise ValueError("VORTALL degrees of freedom must equal nx * ny.")
    variables = {name: (tuple(value.shape), str(value.dtype)) for name, value in contents.items() if not name.startswith("__")}
    return snapshots, spatial_shape, variables


def mirror_vorticity_snapshot(snapshot: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Reflect a MATLAB-order vorticity field cross-stream and reverse its sign."""
    field = snapshot.reshape(spatial_shape, order="F")
    return (-np.flip(field, axis=0)).ravel(order="F")


def build_pod_ensemble(snapshots: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Return the symmetry-augmented ensemble used for POD."""
    mirrored_snapshots = np.column_stack(
        [mirror_vorticity_snapshot(snapshot, spatial_shape) for snapshot in snapshots.T]
    )
    return np.column_stack((snapshots, mirrored_snapshots))


def write_csv(path: Path, snapshots: np.ndarray, time: np.ndarray) -> None:
    """Write ``time, q1, ...`` rows with one FluidModes snapshot per row."""
    header = ",".join(("time", *(f"q{index}" for index in range(1, snapshots.shape[0] + 1))))
    np.savetxt(path, np.column_stack((time, snapshots.T)), delimiter=",", header=header, comments="", fmt="%.17g")


def prepare_inputs() -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, tuple[tuple[int, ...], str]]]:
    """Write raw DMD data and the symmetry-augmented POD ensemble.

    DMD receives raw ``VORTALL`` snapshots. POD receives ``[X, X_mirror]``:
    each MATLAB-order field is reshaped, cross-stream reflected, sign-reversed
    as vorticity, flattened in MATLAB order, and concatenated. Mean subtraction
    is deliberately left to the default ``fluidmodes pod`` calculation.
    """
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    snapshots, spatial_shape, variables = load_cylinder_data()
    pod_snapshots = build_pod_ensemble(snapshots, spatial_shape)
    write_csv(DMD_CSV_PATH, snapshots, SNAPSHOT_SPACING * np.arange(snapshots.shape[1]))
    write_csv(POD_CSV_PATH, pod_snapshots, SNAPSHOT_SPACING * np.arange(pod_snapshots.shape[1]))
    return snapshots, pod_snapshots, spatial_shape, variables


if __name__ == "__main__":
    prepare_inputs()
