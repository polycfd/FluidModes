"""Proper orthogonal decomposition from the economical singular-value decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fluidmodes.preprocessing import subtract_temporal_mean
from fluidmodes.snapshots import SnapshotData


@dataclass(frozen=True)
class PODResult:
    """Results of a standard POD of one snapshot matrix.

    ``modes`` has shape ``(n_dof, rank)`` and ``coefficients`` has shape
    ``(rank, n_time)``.  Energy fractions are relative to the full analyzed
    snapshot matrix, including singular values beyond the retained rank.
    ``mean`` is ``None`` when no temporal mean was removed.
    """

    modes: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    energy_fractions: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    rank: int
    mean: NDArray[np.float64] | None
    time: NDArray[np.float64]
    spatial_shape: tuple[int, ...] | None


def compute_pod(
    data: SnapshotData, *, rank: int | None = None, subtract_mean: bool = True
) -> PODResult:
    """Compute POD of the analyzed snapshot matrix ``X_a = U Σ Vᵀ``.

    ``X_a`` is either the input matrix or its temporal fluctuations. Because
    its columns are snapshots, columns of ``U`` are spatial POD modes and
    ``A = Σ_r V_rᵀ`` contains their temporal coefficients. Squared singular
    values are modal energies. Retained energy fractions are normalized by
    the full singular-value spectrum, so a rank-truncated result need not sum
    to one. When a mean is removed, add the returned mean field to ``Φ A`` to
    reconstruct the original field approximation.
    """
    if not isinstance(subtract_mean, (bool, np.bool_)):
        raise ValueError("subtract_mean must be a boolean.")

    if subtract_mean:
        mean_result = subtract_temporal_mean(data)
        analyzed = mean_result.data.X
        mean: NDArray[np.float64] | None = mean_result.mean
    else:
        analyzed = data.X
        mean = None

    # With snapshots stored as columns, U contains spatial modes and ΣVᵀ
    # contains their temporal coefficients.
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
        analyzed, full_matrices=False
    )

    retained_rank = _validate_rank(rank, singular_values.size)
    spatial_modes = left_vectors[:, :retained_rank]
    temporal_coefficients = (
        singular_values[:retained_rank, None] * right_vectors_transposed[:retained_rank]
    )

    # Energy fractions describe the full analyzed field, not only retained modes.
    total_energy = float(np.sum(singular_values**2))
    if total_energy == 0.0:
        raise ValueError("POD requires an analyzed snapshot matrix with non-zero energy.")
    energy_fractions = singular_values[:retained_rank] ** 2 / total_energy

    return PODResult(
        modes=spatial_modes,
        singular_values=singular_values[:retained_rank],
        energy_fractions=energy_fractions,
        coefficients=temporal_coefficients,
        rank=retained_rank,
        mean=mean,
        time=data.t.copy(),
        spatial_shape=data.spatial_shape,
    )


def _validate_rank(rank: int | None, max_rank: int) -> int:
    if rank is None:
        return max_rank
    if not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)):
        raise ValueError("POD rank must be an integer.")
    if rank < 1 or rank > max_rank:
        raise ValueError(f"POD rank must be between 1 and {max_rank}.")
    return int(rank)
