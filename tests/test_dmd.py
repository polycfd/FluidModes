import warnings

import numpy as np
import pytest

from fluidmodes.dmd import compute_dmd
from fluidmodes.dmd_plots import plot_eigenvalues, plot_spatial_mode
from fluidmodes.manufactured import make_dmd_case
from fluidmodes.snapshots import SnapshotData


def match_eigenvalues(recovered: np.ndarray, prescribed: np.ndarray) -> np.ndarray:
    """Return indices matching each prescribed discrete eigenvalue to recovery.

    DMD does not prescribe an eigenvalue ordering. The manufactured eigenvalues
    are well-separated, so nearest-neighbour matching gives a unique pairing.
    """
    indices = [int(np.argmin(np.abs(recovered - expected))) for expected in prescribed]
    assert len(set(indices)) == len(prescribed)
    return np.asarray(indices)


def test_dmd_recovers_the_manufactured_linear_dynamics() -> None:
    """Recover prescribed conjugate eigenvalue pairs and their real snapshots."""
    case = make_dmd_case()
    frequencies = np.array([1.5, -1.5, 3.5, -3.5])
    growth_rates = np.array([-0.10, -0.10, 0.05, 0.05])
    expected_eigenvalues = np.exp(
        (growth_rates + 2j * np.pi * frequencies) * case.time_step
    )

    assert np.allclose(case.frequencies, frequencies)
    assert np.allclose(case.growth_rates, growth_rates)
    assert np.allclose(case.eigenvalues, expected_eigenvalues)

    result = compute_dmd(case.data)
    # PyDMD may return the same discrete spectrum in a different order.
    matched_indices = match_eigenvalues(result.eigenvalues, expected_eigenvalues)

    assert result.rank == 4
    assert result.modes.shape == (4, 4)
    assert result.amplitudes.shape == (4,)
    assert result.dynamics.shape == (4, 80)
    assert result.reconstruction.shape == case.data.X.shape
    assert np.allclose(result.eigenvalues[matched_indices], expected_eigenvalues)
    assert np.allclose(result.frequencies[matched_indices], frequencies)
    assert np.allclose(result.growth_rates[matched_indices], growth_rates)
    assert np.allclose(result.reconstruction.real, case.data.X, atol=1e-11)
    assert np.max(np.abs(result.reconstruction.imag)) < 1e-12
    assert result.mean is None


def test_dmd_continuous_time_convention_uses_the_principal_logarithm() -> None:
    case = make_dmd_case()
    result = compute_dmd(case.data)
    expected_omega = np.log(result.eigenvalues) / case.time_step

    assert np.allclose(result.continuous_eigenvalues, expected_omega)
    assert np.allclose(result.growth_rates, np.log(np.abs(result.eigenvalues)) / case.time_step)
    assert np.allclose(result.frequencies, np.angle(result.eigenvalues) / (2.0 * np.pi * case.time_step))


def test_dmd_mean_subtraction_is_opt_in_and_restores_the_mean_to_reconstruction() -> None:
    case = make_dmd_case()
    offset = np.array([[2.0], [-1.0], [3.0], [4.0]])
    offset_data = SnapshotData(X=case.data.X + offset, t=case.data.t)

    raw = compute_dmd(offset_data)
    centered = compute_dmd(offset_data, subtract_mean=True)

    assert raw.mean is None
    assert np.allclose(centered.mean, np.mean(offset_data.X, axis=1))
    assert np.allclose(raw.reconstruction, raw.modes @ raw.dynamics)
    assert np.allclose(centered.reconstruction - centered.mean[:, None], centered.modes @ centered.dynamics)


def test_dmd_rank_validation_and_time_requirements() -> None:
    case = make_dmd_case()
    assert compute_dmd(case.data, rank=2).rank == 2

    for rank in (0, -1, 5, 1.5, True):
        with pytest.raises(ValueError, match="rank"):
            compute_dmd(case.data, rank=rank)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least two snapshots"):
        compute_dmd(SnapshotData(X=[[1.0]], t=[0.0]))
    with pytest.raises(ValueError, match="uniformly sampled"):
        compute_dmd(SnapshotData(X=[[1.0, 2.0, 3.0]], t=[0.0, 0.1, 0.3]))


def test_dmd_rejects_exactly_zero_discrete_eigenvalues_without_log_warning() -> None:
    data = SnapshotData(
        X=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
        t=[0.0, 1.0, 2.0],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="zero discrete eigenvalue"):
            compute_dmd(data)


def test_dmd_plots_are_created(tmp_path) -> None:
    result = compute_dmd(make_dmd_case().data)
    eigenvalues = tmp_path / "eigenvalues.png"
    mode = tmp_path / "mode_1.png"

    plot_eigenvalues(result, eigenvalues)
    plot_spatial_mode(result, 1, mode)

    assert eigenvalues.is_file()
    assert mode.is_file()
