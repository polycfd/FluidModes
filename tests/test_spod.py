import numpy as np
import pytest

import fluidmodes.spod as spod_module
from fluidmodes.manufactured import make_spod_case
from fluidmodes.snapshots import SnapshotData
from fluidmodes.spod import compute_spod
from fluidmodes.spod_plots import plot_eigenvalue_spectrum, plot_spatial_mode


def complex_mode_correlation(reference: np.ndarray, recovered: np.ndarray) -> float:
    """Return the normalized inner-product magnitude for complex SPOD modes.

    A complex mode has arbitrary global phase, so its phase-invariant overlap
    with a reference mode is the magnitude of the normalized inner product.
    """
    return float(
        abs(np.vdot(reference, recovered))
        / (np.linalg.norm(reference) * np.linalg.norm(recovered))
    )


def test_spod_recovers_the_manufactured_spectrum_and_coherent_modes() -> None:
    case = make_spod_case()
    block_size = 100
    time_step = 0.01
    frequencies = np.array([5.0, 12.0])
    frequency_resolution = 1.0 / (block_size * time_step)

    assert case.data.t[1] - case.data.t[0] == time_step
    assert np.allclose(case.frequencies, frequencies)
    assert frequency_resolution == 1.0

    result = compute_spod(case.data, block_size=block_size)

    assert result.time_step == time_step
    assert result.n_blocks == 3
    assert result.eigenvalues.shape == (51, 3)
    assert result.modes.shape == (51, 16, 1)
    assert np.allclose(result.frequencies, np.arange(51, dtype=float))
    assert np.all(np.diff(result.eigenvalues, axis=1) <= 1e-14)
    assert np.allclose(result.mean, np.mean(case.data.X, axis=1))

    expected_indices = np.rint(frequencies / frequency_resolution).astype(int)
    peak_indices = np.argsort(result.eigenvalues[:, 0])[-2:]
    assert np.array_equal(np.sort(peak_indices), expected_indices)
    assert np.allclose(result.frequencies[expected_indices], frequencies)

    for frequency_index, reference_mode in zip(expected_indices, case.spatial_modes.T, strict=True):
        recovered_mode = result.modes[frequency_index, :, 0]
        assert complex_mode_correlation(reference_mode, recovered_mode) > 0.99


def test_spod_mean_subtraction_can_be_disabled() -> None:
    case = make_spod_case()
    offset_data = SnapshotData(X=case.data.X + 2.0, t=case.data.t)

    centered = compute_spod(offset_data, block_size=100)
    raw = compute_spod(offset_data, block_size=100, subtract_mean=False)

    assert np.allclose(centered.mean, np.mean(offset_data.X, axis=1))
    assert raw.mean is None


def test_spod_rejects_nonuniform_time_and_invalid_spectral_settings() -> None:
    case = make_spod_case()
    nonuniform = SnapshotData(X=case.data.X[:, :4], t=[0.0, 0.1, 0.2, 0.4])
    with pytest.raises(ValueError, match="uniformly sampled"):
        compute_spod(nonuniform, block_size=4)

    for block_size in (3, 201, 4.5, True):
        with pytest.raises(ValueError, match="block size"):
            compute_spod(case.data, block_size=block_size)  # type: ignore[arg-type]
    for overlap in (-0.1, 1.0, float("nan"), True):
        with pytest.raises(ValueError, match="overlap"):
            compute_spod(case.data, block_size=100, overlap=overlap)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two blocks"):
        compute_spod(case.data, block_size=150, overlap=0.0)
    for modes in (0, 4, 1.5, True):
        with pytest.raises(ValueError, match="SPOD modes"):
            compute_spod(case.data, block_size=100, modes=modes)  # type: ignore[arg-type]


def test_spod_passes_time_first_data_and_explicit_settings_to_pyspod(
    monkeypatch,
) -> None:
    """Check the FluidModes-to-PySPOD boundary without re-testing PySPOD."""
    captured: dict[str, object] = {}

    class RecordingStandard:
        def __init__(self, *, params, weights, comm) -> None:
            captured["params"] = params
            captured["weights"] = weights
            captured["comm"] = comm

        def fit(self, snapshots):
            captured["snapshots"] = snapshots
            self.freq = np.array([0.0])
            self.eigs = np.array([[1.0]])
            return self

        def get_modes_at_freq(self, frequency_index: int) -> np.ndarray:
            assert frequency_index == 0
            return np.ones((16, 1), dtype=complex)

    monkeypatch.setattr(spod_module, "Standard", RecordingStandard)
    result = compute_spod(make_spod_case().data, block_size=100, overlap=0.25)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["n_dft"] == 100
    assert params["overlap"] == 25.0
    assert params["mean_type"] == "longtime"
    assert params["fullspectrum"] is False
    assert params["normalize_weights"] is False
    assert params["normalize_data"] is False
    assert params["n_modes_save"] == 1
    assert captured["weights"] is None
    assert captured["comm"] is None
    assert captured["snapshots"][0].shape == (200, 16, 1)
    assert result.modes.shape == (1, 16, 1)


def test_spod_plots_are_created_and_select_the_nearest_frequency(tmp_path) -> None:
    result = compute_spod(make_spod_case().data, block_size=100)
    spectrum = tmp_path / "spectrum.png"
    mode = tmp_path / "mode.png"

    plot_eigenvalue_spectrum(result, spectrum)
    selected_frequency = plot_spatial_mode(result, 5.4, 1, mode)

    assert spectrum.is_file()
    assert mode.is_file()
    assert selected_frequency == 5.0
