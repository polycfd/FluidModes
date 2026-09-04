import numpy as np
import pytest

from fluidmodes.manufactured import make_pod_case
from fluidmodes.pod import compute_pod
from fluidmodes.pod_plots import (
    plot_energy_spectrum,
    plot_spatial_mode,
    plot_temporal_coefficient,
)
from fluidmodes.snapshots import SnapshotData


def test_pod_recovers_manufactured_modes_and_energies() -> None:
    """Recover the three orthogonal modes with energies 0.6, 0.3, and 0.1."""
    case = make_pod_case()
    result = compute_pod(case.data, rank=3, subtract_mean=False)
    expected_energies = np.array([0.6, 0.3, 0.1])

    assert result.rank == 3
    assert result.modes.shape == (4, 3)
    assert result.coefficients.shape == (3, 60)
    assert np.all(np.diff(result.singular_values) <= 0.0)
    assert np.allclose(case.relative_energies, expected_energies)
    assert np.allclose(result.energy_fractions, expected_energies)
    assert np.isclose(result.energy_fractions.sum(), 1.0)

    # The three retained modes span the complete rank-three manufactured field.
    assert np.allclose(result.modes @ result.coefficients, case.data.X)
    # POD modes may independently change sign without changing the reconstruction.
    assert np.allclose(np.abs(result.modes.T @ case.spatial_modes), np.eye(3))
    assert result.mean is None
    assert np.array_equal(result.time, case.data.t)
    assert result.spatial_shape == (2, 2)


def test_pod_truncation_uses_the_full_singular_value_energy() -> None:
    case = make_pod_case()
    result = compute_pod(case.data, rank=2, subtract_mean=False)

    assert result.modes.shape == (4, 2)
    assert result.coefficients.shape == (2, 60)
    assert np.allclose(result.energy_fractions, [0.6, 0.3])
    assert np.isclose(result.energy_fractions.sum(), 0.9)
    assert np.allclose(
        result.modes @ result.coefficients,
        case.spatial_modes[:, :2] @ case.temporal_coefficients[:2],
    )


def test_pod_mean_subtraction_reconstructs_the_original_field() -> None:
    fluctuation = np.array([[1.0, -1.0, 1.0, -1.0], [2.0, 0.0, -2.0, 0.0]])
    data = SnapshotData(
        X=fluctuation + np.array([[10.0], [20.0]]),
        t=np.arange(4, dtype=float),
        spatial_shape=(1, 2),
    )

    centered = compute_pod(data, subtract_mean=True)
    raw = compute_pod(data, subtract_mean=False)

    assert np.allclose(centered.mean, [10.0, 20.0])
    assert np.allclose(centered.modes @ centered.coefficients + centered.mean[:, None], data.X)
    assert raw.mean is None
    assert np.allclose(raw.modes @ raw.coefficients, data.X)


@pytest.mark.parametrize("rank", [0, -1, 5, 1.5, True])
def test_pod_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(ValueError, match="rank"):
        compute_pod(make_pod_case().data, rank=rank, subtract_mean=False)  # type: ignore[arg-type]


def test_pod_plots_are_created(tmp_path) -> None:
    result = compute_pod(make_pod_case().data, rank=1, subtract_mean=False)
    energy = tmp_path / "energy.png"
    mode = tmp_path / "mode_1.png"
    coefficient = tmp_path / "coefficient_1.png"

    plot_energy_spectrum(result, energy)
    plot_spatial_mode(result, 1, mode)
    plot_temporal_coefficient(result, 1, coefficient)

    assert energy.is_file()
    assert mode.is_file()
    assert coefficient.is_file()
