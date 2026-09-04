"""Prepare the published Re=413 cylinder-PIV velocity snapshots for FluidModes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = EXAMPLE_DIRECTORY / "data"
MAT_PATH = DATA_DIRECTORY / "cylinder_vel.mat"
CSV_PATH = DATA_DIRECTORY / "piv_velocity.csv"
INSPECTION_PATH = DATA_DIRECTORY / "inspection.json"
SAMPLING_FREQUENCY = 20.0
TIME_STEP = 1.0 / SAMPLING_FREQUENCY
N_SNAPSHOTS = 2000
SPATIAL_SHAPE = (135, 80)


def _load_matlab_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load the four published arrays, including a v7.3/HDF5 fallback.

    The released file inspected for this validation is MATLAB v5.  The fallback
    keeps the example's error message useful if Zenodo later publishes a v7.3
    revision; neither reader is a FluidModes dependency.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download cylinder_vel.mat from DOI 10.5281/zenodo.20765567 "
            "and place it in this example's data directory."
        )
    try:
        from scipy.io import loadmat

        contents = loadmat(path, variable_names=("u", "v", "x", "y"))
        required = {"u", "v", "x", "y"}
        if not required.issubset(contents):
            raise ValueError(f"{path} must contain u, v, x, and y.")
        arrays = tuple(np.asarray(contents[name]) for name in ("u", "v", "x", "y"))
        variables = {
            name: {"shape": list(np.asarray(contents[name]).shape), "dtype": str(np.asarray(contents[name]).dtype)}
            for name in ("u", "v", "x", "y")
        }
        return (*arrays, {"mat_format": "MATLAB v5 (scipy.io.loadmat)", "variables": variables})
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as contents:
            required = {"u", "v", "x", "y"}
            if not required.issubset(contents):
                raise ValueError(f"{path} must contain u, v, x, and y.")
            # MATLAB v7.3 stores dimensions reversed in HDF5.  Reversing every
            # axis restores MATLAB array dimensions before the explicit checks.
            arrays = tuple(np.asarray(contents[name]).transpose(tuple(range(contents[name].ndim - 1, -1, -1))) for name in ("u", "v", "x", "y"))
            variables = {
                name: {"shape": list(contents[name].shape[::-1]), "dtype": str(contents[name].dtype)}
                for name in ("u", "v", "x", "y")
            }
        return (*arrays, {"mat_format": "MATLAB v7.3/HDF5 (h5py)", "variables": variables})


def _validate_and_orient(
    u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Validate the documented MATLAB layout and make time the final axis."""
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError(f"u and v must be equal three-dimensional arrays; received {u.shape} and {v.shape}.")
    time_axes = [axis for axis, size in enumerate(u.shape) if size == 8000]
    if len(time_axes) != 1:
        raise ValueError(f"Expected exactly one 8000-snapshot time axis in u/v; received shape {u.shape}.")
    time_axis = time_axes[0]
    oriented_u = np.moveaxis(u, time_axis, -1)
    oriented_v = np.moveaxis(v, time_axis, -1)
    if oriented_u.shape[:2] != SPATIAL_SHAPE:
        raise ValueError(
            f"Expected spatial dimensions {SPATIAL_SHAPE} after moving time axis {time_axis}; "
            f"received {oriented_u.shape}."
        )
    if x.shape != SPATIAL_SHAPE or y.shape != SPATIAL_SHAPE:
        raise ValueError(f"Expected x and y shape {SPATIAL_SHAPE}; received {x.shape} and {y.shape}.")
    if not (np.isrealobj(u) and np.isrealobj(v) and np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("u and v must be finite real velocity arrays.")
    zero_u = oriented_u == 0.0
    zero_v = oriented_v == 0.0
    # The published description identifies zero-valued masked regions.  The
    # released arrays contain a nearly fixed common core but also a small
    # number of component- and time-specific exact zeros.  Preserve every
    # value as published; this example must not manufacture a replacement mask
    # or reject valid experimental data merely because those diagnostics differ.
    common_zero_mask = zero_u & zero_v
    x_is_axis0 = np.allclose(x, x[:, :1], rtol=0.0, atol=1.0e-14)
    y_is_axis1 = np.allclose(y, y[:1, :], rtol=0.0, atol=1.0e-14)
    if not (x_is_axis0 and y_is_axis1):
        raise ValueError("Expected a rectilinear x(axis 0), y(axis 1) coordinate grid; do not silently reinterpret this grid.")
    dx = np.diff(x[:, 0])
    dy = np.diff(y[0, :])
    if not (np.allclose(dx, dx[0], rtol=1.0e-10, atol=1.0e-14) and np.allclose(dy, dy[0], rtol=1.0e-10, atol=1.0e-14)):
        raise ValueError("Expected a regular coordinate grid; do not interpolate this experimental data.")
    inspection = {
        "time_axis_in_released_u_v": time_axis,
        "oriented_velocity_shape": list(oriented_u.shape),
        "spatial_shape": list(SPATIAL_SHAPE),
        "n_snapshots_released": int(oriented_u.shape[-1]),
        "u_dtype": str(u.dtype),
        "v_dtype": str(v.dtype),
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "finite_velocities": True,
        "u_v_zero_masks_equal": bool(np.array_equal(zero_u, zero_v)),
        "u_zero_mask_constant_in_time": bool(
            np.array_equal(zero_u, np.broadcast_to(zero_u[..., :1], zero_u.shape))
        ),
        "v_zero_mask_constant_in_time": bool(
            np.array_equal(zero_v, np.broadcast_to(zero_v[..., :1], zero_v.shape))
        ),
        "common_zero_spatial_dofs_all_times": int(np.count_nonzero(np.all(common_zero_mask, axis=-1))),
        "u_zero_entries": int(np.count_nonzero(zero_u)),
        "v_zero_entries": int(np.count_nonzero(zero_v)),
        "u_only_zero_entries": int(np.count_nonzero(zero_u & ~zero_v)),
        "v_only_zero_entries": int(np.count_nonzero(zero_v & ~zero_u)),
        "grid_regular": True,
        "x_increases_with_axis_0": bool(dx[0] > 0.0),
        "y_increases_with_axis_1": bool(dy[0] > 0.0),
        "dx": float(dx[0]),
        "dy": float(dy[0]),
    }
    return oriented_u, oriented_v, x, y, inspection


def load_velocity_snapshots(path: Path = MAT_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return first-2000 raw [u; v] snapshots, coordinates, and inspection data.

    The array mapping is deliberately explicit: each MATLAB-layout field has
    shape ``(n_x, n_y)`` and is flattened in C order (``y`` varies fastest).
    Thus the first ``N = 135*80`` rows are u and the next N rows are v.
    """
    u, v, x, y, file_info = _load_matlab_arrays(path)
    u, v, x, y, inspection = _validate_and_orient(u, v, x, y)
    if u.shape[-1] < N_SNAPSHOTS:
        raise ValueError(f"Expected at least {N_SNAPSHOTS} snapshots; received {u.shape[-1]}.")
    selected_u = u[..., :N_SNAPSHOTS]
    selected_v = v[..., :N_SNAPSHOTS]
    n_field_dof = int(np.prod(SPATIAL_SHAPE))
    states = np.vstack((
        selected_u.reshape(n_field_dof, N_SNAPSHOTS, order="C"),
        selected_v.reshape(n_field_dof, N_SNAPSHOTS, order="C"),
    )).astype(float, copy=False)
    inspection.update(file_info)
    inspection.update({
        "subset": "first 2000 consecutive snapshots (MATLAB indices 1--2000)",
        "n_snapshots_selected": N_SNAPSHOTS,
        "dt_seconds": TIME_STEP,
        "n_dof": int(states.shape[0]),
        "state_order": "u.ravel(order='C'), then v.ravel(order='C')",
        "state_matrix_mib": float(states.nbytes / 1024**2),
    })
    return states, x.astype(float, copy=False), y.astype(float, copy=False), inspection


def _write_csv(path: Path, states: np.ndarray, time: np.ndarray) -> None:
    labels = ",".join(("time", *(f"q{index}" for index in range(1, states.shape[0] + 1))))
    np.savetxt(path, np.column_stack((time, states.T)), delimiter=",", header=labels, comments="", fmt="%.17g")


def prepare_csv(path: Path = MAT_PATH) -> dict[str, object]:
    """Write the raw-state CSV and verify it through FluidModes' public reader."""
    from fluidmodes.tabular import read_csv

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    states, x, y, inspection = load_velocity_snapshots(path)
    time = TIME_STEP * np.arange(states.shape[1], dtype=float)
    _write_csv(CSV_PATH, states, time)
    reconstructed = read_csv(CSV_PATH)
    if not np.array_equal(reconstructed.t, time) or reconstructed.X.shape != states.shape:
        raise ValueError("FluidModes CSV round trip changed time coordinates or state dimensions.")
    if not np.array_equal(reconstructed.X, states):
        difference = float(np.max(np.abs(reconstructed.X - states)))
        raise ValueError(f"FluidModes CSV round trip changed velocity values; maximum difference {difference:.3e}.")
    inspection.update({
        "mat_bytes": int(path.stat().st_size),
        "csv_bytes": int(CSV_PATH.stat().st_size),
        "csv_roundtrip_exact": True,
        "x_min_max": [float(np.min(x)), float(np.max(x))],
        "y_min_max": [float(np.min(y)), float(np.max(y))],
    })
    INSPECTION_PATH.write_text(json.dumps(inspection, indent=2) + "\n")
    return inspection


def main() -> None:
    inspection = prepare_csv()
    print(f"Prepared {CSV_PATH}")
    print(f"MAT size: {inspection['mat_bytes'] / 1024**3:.2f} GiB; CSV size: {inspection['csv_bytes'] / 1024**3:.2f} GiB")
    print(f"Selected snapshots: {inspection['n_snapshots_selected']}; DOFs: {inspection['n_dof']}")
    print(f"State matrix: {inspection['state_matrix_mib']:.1f} MiB; exact CSV round trip: yes")


if __name__ == "__main__":
    main()
