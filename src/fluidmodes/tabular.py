"""Translate between time-row tabular files and FluidModes snapshot matrices."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from fluidmodes.snapshots import SnapshotData


def read_csv(path: str | Path) -> SnapshotData:
    """Read ``time, q1, q2, ...`` CSV rows into ``X[n_dof, n_time]``.

    Each input row is one snapshot, so the state portion is transposed when
    creating FluidModes' column-per-snapshot representation. Header labels
    after ``time`` are retained as degree-of-freedom labels.
    """
    source = Path(path)
    with source.open(newline="") as file:
        rows = list(csv.reader(file))
    return _snapshot_from_rows(rows, source)


def read_txt(path: str | Path) -> SnapshotData:
    """Read a whitespace table with the same ``time, q1, q2, ...`` layout."""
    source = Path(path)
    with source.open() as file:
        rows = [line.split() for line in file]
    return _snapshot_from_rows(rows, source)


def write_csv(data: SnapshotData, path: str | Path) -> None:
    """Write snapshots as time rows in the CSV layout accepted by :func:`read_csv`."""
    destination = Path(path)
    with destination.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_table_rows(data))


def write_txt(data: SnapshotData, path: str | Path) -> None:
    """Write snapshots as time rows in the TXT layout accepted by :func:`read_txt`."""
    destination = Path(path)
    with destination.open("w") as file:
        for row in _table_rows(data):
            file.write(" ".join(row) + "\n")


def _snapshot_from_rows(rows: list[list[str]], source: Path) -> SnapshotData:
    if not rows or not rows[0]:
        raise ValueError(f"{source}: expected a non-empty header row.")

    header = [entry.strip() for entry in rows[0]]
    if len(header) < 2:
        raise ValueError(f"{source}: expected a time column and at least one state column.")
    if any(not entry for entry in header):
        raise ValueError(f"{source}: header labels must be non-empty.")
    if len(rows) == 1:
        raise ValueError(f"{source}: expected at least one data row.")

    values: list[list[float]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"{source}: row {line_number} has {len(row)} columns; expected {len(header)}.")
        try:
            values.append([float(value) for value in row])
        except ValueError as error:
            raise ValueError(f"{source}: row {line_number} contains non-numeric data.") from error

    table = np.asarray(values, dtype=float)
    # File rows are snapshots; FluidModes stores the same snapshots as columns.
    return SnapshotData(X=table[:, 1:].T, t=table[:, 0], labels=header[1:], time_label=header[0])


def _table_rows(data: SnapshotData) -> Iterable[list[str]]:
    labels = data.labels or tuple(f"q{index + 1}" for index in range(data.X.shape[0]))
    yield [data.time_label or "time", *labels]
    # Seventeen significant digits preserve a float64 CSV/TXT round trip.
    for time, state in zip(data.t, data.X.T, strict=True):
        yield [format(time, ".17g"), *(format(value, ".17g") for value in state)]
