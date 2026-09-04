import numpy as np

from fluidmodes.manufactured import make_dmd_case, make_pod_case, make_spod_case


def test_pod_case_has_orthogonal_rank_three_reference_data() -> None:
    case = make_pod_case()

    assert np.linalg.matrix_rank(case.data.X) == case.expected_rank
    assert np.allclose(case.spatial_modes.T @ case.spatial_modes, np.eye(case.expected_rank))
    assert np.allclose(
        np.sum(case.temporal_coefficients**2, axis=1) / np.sum(case.temporal_coefficients**2),
        case.relative_energies,
    )


def test_dmd_case_follows_declared_discrete_time_evolution() -> None:
    case = make_dmd_case()
    reconstructed = case.modes @ (
        case.amplitudes[:, None] * case.eigenvalues[:, None] ** np.arange(case.data.t.size)
    )

    assert np.allclose(case.eigenvalues, np.exp((case.growth_rates + 2j * np.pi * case.frequencies) * case.time_step))
    assert np.allclose(reconstructed.imag, 0.0)
    assert np.allclose(case.data.X, reconstructed.real)


def test_spod_case_has_the_prescribed_frequency_peaks() -> None:
    case = make_spod_case()
    coefficients = case.spatial_modes.T @ case.data.X
    frequency_bins = np.fft.rfftfreq(case.data.t.size, d=case.data.t[1] - case.data.t[0])
    peaks = frequency_bins[np.argmax(np.abs(np.fft.rfft(coefficients, axis=1)), axis=1)]

    assert np.allclose(peaks, case.frequencies)
