"""Preprocessing operations for FluidModes snapshot data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fluidmodes.snapshots import SnapshotData


@dataclass(frozen=True)
class MeanSubtractionResult:
    """Fluctuation snapshots and the time-mean field removed from them."""

    data: SnapshotData
    mean: NDArray[np.float64]


def subtract_temporal_mean(data: SnapshotData) -> MeanSubtractionResult:
    """Return fluctuations about the per-DOF temporal mean.

    For each degree of freedom, this forms
    ``X'[i, k] = X[i, k] - (1 / n_time) sum_l X[i, l]``. The returned mean
    permits interpretation of the base field and reconstruction of the
    corresponding original-field approximation.
    """
    mean = np.mean(data.X, axis=1)
    centered = SnapshotData(
        X=data.X - mean[:, None],
        t=data.t.copy(),
        labels=data.labels,
        time_label=data.time_label,
        spatial_shape=data.spatial_shape,
    )
    return MeanSubtractionResult(data=centered, mean=mean)
