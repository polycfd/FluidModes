"""Compare independent cylinder-wake references with released CLI outputs."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from prepare_data import EXAMPLE_DIRECTORY
from reference_dmd import DMD_RANK
from run_fluidmodes import DMD_RESULTS, POD_RESULTS


POD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"
DMD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "dmd_reference.npz"
COMPARISON_PATH = EXAMPLE_DIRECTORY / "reference" / "comparison.json"


def match_eigenvalues(reference: np.ndarray, recovered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match unordered DMD spectra by minimum total complex-plane distance."""
    reference_indices, recovered_indices = linear_sum_assignment(
        np.abs(reference[:, None] - recovered[None, :])
    )
    order = np.argsort(reference_indices)
    return reference_indices[order], recovered_indices[order]


def mode_correlation(reference: np.ndarray, recovered: np.ndarray) -> np.ndarray:
    """Return sign/phase-invariant normalized spatial-mode correlations."""
    return np.abs(np.sum(recovered.conj() * reference, axis=0)) / (
        np.linalg.norm(recovered, axis=0) * np.linalg.norm(reference, axis=0)
    )


def main() -> None:
    """Load reference and FluidModes files, then calculate every reported error."""
    pod_reference = np.load(POD_REFERENCE)
    dmd_reference = np.load(DMD_REFERENCE)

    # FluidModes POD outputs: singular_values.csv, energy.csv, and modes.npy.
    pod_singular_values = np.loadtxt(POD_RESULTS / "singular_values.csv", delimiter=",", skiprows=1)[:, 1]
    pod_energies = np.loadtxt(POD_RESULTS / "energy.csv", delimiter=",", skiprows=1)[:, 1]
    pod_modes = np.load(POD_RESULTS / "modes.npy")
    n_compare = min(20, pod_reference["singular_values"].size)
    pod_mode_correlations = mode_correlation(pod_reference["modes"][:, :6], pod_modes[:, :6])
    pod_subspace_correlations = [
        float(np.min(np.linalg.svd(pod_modes[:, start:start + 2].T @ pod_reference["modes"][:, start:start + 2], compute_uv=False)))
        for start in (0, 2, 4)
    ]

    # FluidModes DMD outputs: eigenvalues.npy, eigenvalues.csv, modes.npy, amplitudes.npy.
    dmd_eigenvalues = np.load(DMD_RESULTS / "eigenvalues.npy")
    dmd_table = np.loadtxt(DMD_RESULTS / "eigenvalues.csv", delimiter=",", skiprows=1)
    dmd_frequencies = dmd_table[:, 4]
    dmd_growth_rates = dmd_table[:, 3]
    dmd_modes = np.load(DMD_RESULTS / "modes.npy")
    dmd_amplitudes = np.load(DMD_RESULTS / "amplitudes.npy")
    reference_indices, recovered_indices = match_eigenvalues(dmd_reference["eigenvalues"], dmd_eigenvalues)
    dmd_mode_correlations = mode_correlation(dmd_reference["modes"][:, reference_indices], dmd_modes[:, recovered_indices])
    shedding_candidates = np.flatnonzero((dmd_frequencies > 0.05) & (dmd_frequencies < 0.30))
    shedding_index = int(shedding_candidates[np.argmax(np.linalg.norm(dmd_modes[:, shedding_candidates], axis=0) * np.abs(dmd_amplitudes[shedding_candidates]))])
    shedding_frequency = float(dmd_frequencies[shedding_index])

    summary = {
        "pod_max_singular_relative_error": float(np.max(np.abs(pod_singular_values[:n_compare] - pod_reference["singular_values"][:n_compare]) / pod_reference["singular_values"][:n_compare])),
        "pod_max_energy_absolute_error": float(np.max(np.abs(pod_energies[:n_compare] - pod_reference["energy_fractions"][:n_compare]))),
        "pod_mode_correlations": [float(value) for value in pod_mode_correlations],
        "pod_subspace_correlations": pod_subspace_correlations,
        "dmd_max_eigenvalue_error": float(np.max(np.abs(dmd_reference["eigenvalues"][reference_indices] - dmd_eigenvalues[recovered_indices]))),
        "dmd_max_frequency_error": float(np.max(np.abs(dmd_reference["frequencies"][reference_indices] - dmd_frequencies[recovered_indices]))),
        "dmd_max_growth_rate_error": float(np.max(np.abs(dmd_reference["growth_rates"][reference_indices] - dmd_growth_rates[recovered_indices]))),
        "dmd_mode_correlations": [float(value) for value in dmd_mode_correlations[:6]],
        "shedding_mode_index": shedding_index,
        "shedding_frequency": shedding_frequency,
        "shedding_difference_from_st_016": abs(shedding_frequency - 0.16),
        "dmd_rank": DMD_RANK,
    }
    COMPARISON_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
