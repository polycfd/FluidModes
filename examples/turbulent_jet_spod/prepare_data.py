"""Prepare the Mach 0.9 jet LES data for the FluidModes CSV reader."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = EXAMPLE_DIRECTORY / "data"
DEFAULT_MAT_PATH = DATA_DIRECTORY / "jetLES.mat"
CSV_PATH = DATA_DIRECTORY / "jet_spod.csv"


def load_jet_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, tuple[tuple[int, ...], str]]]:
    """Load MATLAB-order ``p[t, x, r]``, coordinates, and ``dt``.

    The official file is MATLAB v7.3, so current SciPy deliberately rejects it.
    The SciPy path remains useful for an earlier MATLAB-file variant; h5py is the
    required fallback for this HDF5-backed authoritative dataset.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download jet_data/jetLES.mat from "
            "https://github.com/SpectralPOD/spod_matlab and place it there."
        )

    try:
        from scipy.io import loadmat

        contents = loadmat(path)
        required = {"p", "x", "r", "dt"}
        if not required.issubset(contents):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))}.")
        p = np.asarray(contents["p"])
        x = np.asarray(contents["x"])
        r = np.asarray(contents["r"])
        dt = float(np.asarray(contents["dt"]).squeeze())
        variables = {
            name: (tuple(np.asarray(value).shape), str(np.asarray(value).dtype))
            for name, value in contents.items()
            if not name.startswith("__")
        }
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as contents:
            required = {"p", "x", "r", "dt"}
            if not required.issubset(contents):
                raise ValueError(f"{path} must contain {', '.join(sorted(required))}.")
            # MATLAB v7.3 HDF5 dimensions are stored in reverse order.  Reverse
            # all axes to recover MATLAB's p[n_time, n_x, n_r] convention.
            p = np.asarray(contents["p"]).transpose(2, 1, 0)
            x = np.asarray(contents["x"]).T
            r = np.asarray(contents["r"]).T
            dt = float(np.asarray(contents["dt"])[0, 0])
            variables = {
                name: (tuple(dataset.shape[::-1]), str(dataset.dtype))
                for name, dataset in contents.items()
                if hasattr(dataset, "shape")
            }

    if p.ndim != 3:
        raise ValueError(f"p must have MATLAB dimensions (time, x, r); received {p.shape}.")
    if p.shape[1:] != x.shape or x.shape != r.shape:
        raise ValueError(
            f"p spatial shape {p.shape[1:]}, x shape {x.shape}, and r shape {r.shape} disagree."
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a positive finite scalar; received {dt!r}.")
    if not np.isrealobj(p):
        raise ValueError("The unweighted real-data validation requires real-valued pressure snapshots.")
    return np.asarray(p, dtype=float), np.asarray(x), np.asarray(r), dt, variables


def flatten_matlab_field(field: np.ndarray) -> np.ndarray:
    """Flatten a MATLAB-layout ``(x, r)`` pressure field in C order.

    This creates CSV columns with ``r`` varying fastest.  The inverse mapping
    used by validation is ``mode.reshape((n_x, n_r), order='C')``.
    """
    return np.ravel(field, order="C")


def write_csv(path: Path, snapshots: np.ndarray, time: np.ndarray) -> None:
    """Write ``time, q1, ...`` in the released FluidModes CSV convention."""
    labels = ",".join(["time", *(f"q{index}" for index in range(1, snapshots.shape[1] + 1))])
    np.savetxt(path, np.column_stack((time, snapshots)), delimiter=",", header=labels, comments="", fmt="%.17g")


def prepare_csv(mat_path: Path = DEFAULT_MAT_PATH) -> dict[str, object]:
    """Convert one authoritative ``jetLES.mat`` file and verify CSV fidelity."""
    from fluidmodes.tabular import read_csv

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    p, x, r, dt, variables = load_jet_data(mat_path)
    flattened = p.reshape(p.shape[0], -1, order="C")
    time = dt * np.arange(p.shape[0], dtype=float)
    write_csv(CSV_PATH, flattened, time)

    reconstructed = read_csv(CSV_PATH)
    if not np.array_equal(reconstructed.t, time):
        raise ValueError("CSV round trip changed the prescribed time vector.")
    if reconstructed.X.shape != flattened.T.shape or not np.allclose(
        reconstructed.X, flattened.T, rtol=1.0e-14, atol=1.0e-14
    ):
        difference = float(np.max(np.abs(reconstructed.X - flattened.T)))
        raise ValueError(f"CSV round trip changed the pressure data; maximum difference {difference:.3e}.")
    return {
        "mat_path": mat_path,
        "csv_path": CSV_PATH,
        "p_shape": tuple(p.shape),
        "spatial_shape": tuple(p.shape[1:]),
        "dt": dt,
        "variables": variables,
        "csv_bytes": CSV_PATH.stat().st_size,
        "x": x,
        "r": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", nargs="?", type=Path, default=DEFAULT_MAT_PATH)
    args = parser.parse_args()
    summary = prepare_csv(args.mat_path)
    print(f"Prepared {summary['csv_path']}")
    print(f"p shape (time, x, r): {summary['p_shape']}; dt: {summary['dt']:.17g}")
    print(f"CSV size: {summary['csv_bytes'] / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
