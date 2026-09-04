"""Compare independent PIV POD/DMD references with released CLI outputs."""

from __future__ import annotations

import json

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY
from reference_dmd import DMD_RANK
from run_fluidmodes import DMD_RESULTS, POD_RESULTS


POD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"
DMD_REFERENCE = EXAMPLE_DIRECTORY / "reference" / "dmd_reference.npz"
COMPARISON_PATH = EXAMPLE_DIRECTORY / "reference" / "comparison.json"
LITERATURE_WAKE_FREQUENCY = 0.889


def _greedy_eigenvalue_match(reference: np.ndarray, recovered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Explicit order-independent one-to-one matching for this fixed rank-12 spectrum."""
    remaining = set(range(recovered.size))
    recovered_indices: list[int] = []
    for value in reference:
        recovered_index = min(remaining, key=lambda index: abs(value - recovered[index]))
        remaining.remove(recovered_index)
        recovered_indices.append(recovered_index)
    return np.arange(reference.size), np.asarray(recovered_indices)


def _mode_correlations(reference: np.ndarray, recovered: np.ndarray) -> np.ndarray:
    """Return phase/sign-invariant normalized spatial-mode correlations."""
    correlations = np.abs(np.sum(recovered.conj() * reference, axis=0)) / (
        np.linalg.norm(reference, axis=0) * np.linalg.norm(recovered, axis=0)
    )
    return np.clip(correlations, 0.0, 1.0)


def _nearest_positive_frequency_index(frequencies: np.ndarray, target: float) -> int:
    candidates = np.flatnonzero(frequencies > 0.0)
    if candidates.size == 0:
        raise ValueError("DMD spectrum contains no positive frequencies.")
    return int(candidates[np.argmin(np.abs(frequencies[candidates] - target))])


def main() -> None:
    """Load reference/CLI outputs and write direct numerical and physical comparisons."""
    pod_reference = np.load(POD_REFERENCE)
    dmd_reference = np.load(DMD_REFERENCE)
    pod_singular_values = np.loadtxt(POD_RESULTS / "singular_values.csv", delimiter=",", skiprows=1)[:, 1]
    pod_energy = np.loadtxt(POD_RESULTS / "energy.csv", delimiter=",", skiprows=1)[:, 1]
    pod_modes = np.load(POD_RESULTS / "modes.npy")
    n_pod_compare = min(20, pod_reference["singular_values"].size, pod_singular_values.size)
    pod_correlations = _mode_correlations(
        pod_reference["modes"][:, :n_pod_compare], pod_modes[:, :n_pod_compare]
    )
    pod_subspaces = []
    for start in (0, 2, 4):
        singular_values = np.linalg.svd(
            pod_reference["modes"][:, start : start + 2].T @ pod_modes[:, start : start + 2],
            compute_uv=False,
        )
        pod_subspaces.append(float(np.min(singular_values)))
    coefficient = pod_reference["leading_coefficients"][0]
    coefficient = coefficient - np.mean(coefficient)
    fft_frequencies = np.fft.rfftfreq(coefficient.size, d=0.05)
    fft_power = np.abs(np.fft.rfft(coefficient)) ** 2
    pod_fft_index = int(np.argmax(fft_power[1:]) + 1)

    dmd_eigenvalues = np.load(DMD_RESULTS / "eigenvalues.npy")
    dmd_table = np.loadtxt(DMD_RESULTS / "eigenvalues.csv", delimiter=",", skiprows=1)
    dmd_frequencies = dmd_table[:, 4]
    dmd_growth_rates = dmd_table[:, 3]
    dmd_modes = np.load(DMD_RESULTS / "modes.npy")
    reference_indices, fluidmodes_indices = _greedy_eigenvalue_match(
        dmd_reference["eigenvalues"], dmd_eigenvalues
    )
    dmd_correlations = _mode_correlations(
        dmd_reference["modes"][:, reference_indices], dmd_modes[:, fluidmodes_indices]
    )
    reference_wake_index = _nearest_positive_frequency_index(
        dmd_reference["frequencies"], LITERATURE_WAKE_FREQUENCY
    )
    matched_wake_index = int(fluidmodes_indices[reference_wake_index])
    fluidmodes_wake_frequency = float(dmd_frequencies[matched_wake_index])
    reference_wake_frequency = float(dmd_reference["frequencies"][reference_wake_index])
    harmonic_candidates = {}
    for multiple in (2, 3):
        index = _nearest_positive_frequency_index(dmd_frequencies, multiple * LITERATURE_WAKE_FREQUENCY)
        harmonic_candidates[str(multiple)] = float(dmd_frequencies[index])

    summary = {
        "pod_max_singular_relative_error_first_20": float(np.max(np.abs(pod_singular_values[:n_pod_compare] - pod_reference["singular_values"][:n_pod_compare]) / pod_reference["singular_values"][:n_pod_compare])),
        "pod_max_energy_absolute_error_first_20": float(np.max(np.abs(pod_energy[:n_pod_compare] - pod_reference["energy_fractions"][:n_pod_compare]))),
        "pod_max_cumulative_energy_absolute_error_first_20": float(np.max(np.abs(np.cumsum(pod_energy[:n_pod_compare]) - pod_reference["cumulative_energy"][:n_pod_compare]))),
        "pod_mode_correlations_first_20": [float(value) for value in pod_correlations],
        "pod_pair_subspace_correlations_1_2_3_4_5_6": pod_subspaces,
        "pod_energy_first_two": float(np.sum(pod_energy[:2])),
        "pod_cumulative_energy_mode_12": float(np.sum(pod_energy[:12])),
        "pod_modes_to_reach_75_percent_energy": int(np.searchsorted(np.cumsum(pod_energy), 0.75) + 1),
        "leading_pod_coefficient_fft_peak_hz": float(fft_frequencies[pod_fft_index]),
        "frequency_resolution_hz": float(1.0 / (coefficient.size * 0.05)),
        "dmd_rank": DMD_RANK,
        "dmd_max_eigenvalue_error": float(np.max(np.abs(dmd_reference["eigenvalues"][reference_indices] - dmd_eigenvalues[fluidmodes_indices]))),
        "dmd_max_frequency_error": float(np.max(np.abs(dmd_reference["frequencies"][reference_indices] - dmd_frequencies[fluidmodes_indices]))),
        "dmd_max_growth_rate_error": float(np.max(np.abs(dmd_reference["growth_rates"][reference_indices] - dmd_growth_rates[fluidmodes_indices]))),
        "dmd_mode_correlations": [float(value) for value in dmd_correlations],
        "reference_wake_frequency_hz": reference_wake_frequency,
        "fluidmodes_wake_frequency_hz": fluidmodes_wake_frequency,
        "reference_fluidmodes_wake_difference_hz": abs(reference_wake_frequency - fluidmodes_wake_frequency),
        "fluidmodes_wake_difference_from_literature_hz": abs(fluidmodes_wake_frequency - LITERATURE_WAKE_FREQUENCY),
        "fluidmodes_wake_relative_difference_from_literature": abs(fluidmodes_wake_frequency - LITERATURE_WAKE_FREQUENCY) / LITERATURE_WAKE_FREQUENCY,
        "fluidmodes_wake_mode_index_one_based": matched_wake_index + 1,
        "nearest_positive_frequency_to_literature_harmonics_hz": harmonic_candidates,
    }
    COMPARISON_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
