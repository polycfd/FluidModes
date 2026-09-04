"""Compare independent method-of-snapshots SPOD with FluidModes CLI output."""

from __future__ import annotations

import json

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY
from reference_spod import SELECTED_FREQUENCY_INDICES
from run_fluidmodes import SPOD_RESULTS


REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "reference_spod.npz"
COMPARISON_PATH = EXAMPLE_DIRECTORY / "reference" / "comparison.json"


def complex_mode_correlation(reference: np.ndarray, recovered: np.ndarray) -> float:
    """Return phase-invariant complex-mode correlation |φ_FM* φ_ref|/(||φ_FM||||φ_ref||)."""
    return float(abs(np.vdot(recovered, reference)) / (np.linalg.norm(recovered) * np.linalg.norm(reference)))


def relative_error(recovered: np.ndarray, reference: np.ndarray) -> float:
    """Return the maximum relative error where the reference branch is significant."""
    mask = reference > np.max(reference) * 1.0e-10
    return float(np.max(np.abs(recovered[mask] - reference[mask]) / reference[mask]))


def main() -> None:
    """Load independent reference and CLI files, apply normalization, and compare."""
    reference = np.load(REFERENCE_PATH)
    # FluidModes outputs read here: eigenvalues.npy, eigenvalues.csv, and modes.npy.
    fluidmodes_eigenvalues = np.load(SPOD_RESULTS / "eigenvalues.npy")
    fluidmodes_modes = np.load(SPOD_RESULTS / "modes.npy")
    fluidmodes_frequencies = np.loadtxt(SPOD_RESULTS / "eigenvalues.csv", delimiter=",", skiprows=1)[:, 0]
    reference_frequencies = reference["frequencies"]
    reference_eigenvalues = reference["eigenvalues"]

    # PySPOD uses FFT(w q)/sum(w); the MATLAB reference uses a unit-power
    # Hamming window and sqrt(dt) FFT(w q). This squared amplitude ratio is
    # analytical, not a fitted spectral rescaling.
    scale_to_pyspod = float(reference["scale_to_pyspod"])
    converted_fluidmodes_eigenvalues = fluidmodes_eigenvalues / scale_to_pyspod
    correlations = []
    for position, frequency_index in enumerate(SELECTED_FREQUENCY_INDICES):
        correlations.append({
            "matlab_index": frequency_index + 1,
            "frequency": float(reference_frequencies[frequency_index]),
            "mode_1": complex_mode_correlation(reference["modes"][position, :, 0], fluidmodes_modes[frequency_index, :, 0]),
            "mode_2": complex_mode_correlation(reference["modes"][position, :, 1], fluidmodes_modes[frequency_index, :, 1]),
        })
    summary = {
        "n_frequency_bins": int(reference_frequencies.size),
        "frequency_resolution": float(reference_frequencies[1] - reference_frequencies[0]),
        "max_frequency_error": float(np.max(np.abs(fluidmodes_frequencies - reference_frequencies))),
        "normalization_factor_to_pyspod": scale_to_pyspod,
        "raw_leading_relative_error": relative_error(fluidmodes_eigenvalues[:, 0], reference_eigenvalues[:, 0]),
        "raw_second_relative_error": relative_error(fluidmodes_eigenvalues[:, 1], reference_eigenvalues[:, 1]),
        "leading_relative_error_after_conversion": relative_error(converted_fluidmodes_eigenvalues[:, 0], reference_eigenvalues[:, 0]),
        "second_relative_error_after_conversion": relative_error(converted_fluidmodes_eigenvalues[:, 1], reference_eigenvalues[:, 1]),
        "mode_correlations": correlations,
    }
    COMPARISON_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
