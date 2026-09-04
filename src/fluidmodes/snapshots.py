"""Common representation for time-resolved fields and state snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SnapshotData:
    """A real-valued snapshot matrix and its time and spatial metadata.

    FluidModes uses ``X`` with shape ``(n_dof, n_time)``: row ``i`` is the
    time history of spatial or state degree of freedom ``i``, while column
    ``k`` is the field at time ``t[k]``. ``labels``, when present, name the
    rows of ``X``; ``time_label`` preserves the source table's time heading.

    ``spatial_shape`` records how one column may be reshaped for plotting.
    Its dimensions must multiply to ``n_dof`` so that flattening and
    reshaping retain a one-to-one correspondence with the rows of ``X``.
    """

    X: NDArray[np.float64]
    t: NDArray[np.float64]
    labels: Sequence[str] | None = None
    time_label: str | None = None
    spatial_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        t = np.asarray(self.t)

        if X.ndim != 2:
            raise ValueError("Snapshot matrix X must be two-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Snapshot matrix X must contain at least one degree of freedom and one time snapshot.")
        if np.iscomplexobj(X) or not np.issubdtype(X.dtype, np.number):
            raise ValueError("Snapshot matrix X must contain real numeric values.")
        if not np.isfinite(X).all():
            raise ValueError("Snapshot matrix X must contain only finite values.")

        if t.ndim != 1:
            raise ValueError("Time coordinates t must be one-dimensional.")
        if np.iscomplexobj(t) or not np.issubdtype(t.dtype, np.number):
            raise ValueError("Time coordinates t must contain real numeric values.")
        if t.size != X.shape[1]:
            raise ValueError("Number of time coordinates must equal the number of snapshot columns.")
        if not np.isfinite(t).all():
            raise ValueError("Time coordinates t must contain only finite values.")
        if not np.all(np.diff(t) > 0):
            raise ValueError("Time coordinates t must be strictly increasing.")

        labels = None if self.labels is None else tuple(self.labels)
        if labels is not None and len(labels) != X.shape[0]:
            raise ValueError("Number of degree-of-freedom labels must equal the number of rows in X.")
        if labels is not None and any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("Degree-of-freedom labels must be non-empty strings.")
        if self.time_label is not None and (
            not isinstance(self.time_label, str) or not self.time_label
        ):
            raise ValueError("Time-column label must be a non-empty string when provided.")

        spatial_shape = None if self.spatial_shape is None else tuple(self.spatial_shape)
        if spatial_shape is not None and (
            not spatial_shape
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in spatial_shape)
        ):
            raise ValueError("Spatial shape must contain positive integer dimensions.")
        if spatial_shape is not None and prod(spatial_shape) != X.shape[0]:
            raise ValueError("Product of spatial shape dimensions must equal the number of rows in X.")

        object.__setattr__(self, "X", X.astype(float, copy=False))
        object.__setattr__(self, "t", t.astype(float, copy=False))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "spatial_shape", spatial_shape)


def uniform_time_step(time: NDArray[np.float64], *, analysis_name: str) -> float:
    """Validate uniform sampling and return the existing snapshot spacing.

    This helper checks consecutive coordinates; it does not interpolate or
    resample data. Analyses using a temporal evolution or spectral frequency
    therefore retain the sampling represented by the input snapshots.
    """
    if time.size < 2:
        raise ValueError(f"{analysis_name} requires at least two snapshots.")
    time_steps = np.diff(time)
    time_step = float(time_steps[0])
    if not np.allclose(time_steps, time_step, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"{analysis_name} requires uniformly sampled snapshots; consecutive time steps must match."
        )
    return time_step
