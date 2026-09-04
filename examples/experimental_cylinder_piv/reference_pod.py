"""Compute an independent NumPy-SVD POD reference for experimental PIV data."""

from __future__ import annotations

import time

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY, load_velocity_snapshots


REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"
N_SAVED_MODES = 20


def main() -> None:
    """Compute X' = X-Xbar = U Sigma V^T directly with NumPy."""
    snapshots, _, _, _ = load_velocity_snapshots()
    started = time.perf_counter()
    mean = np.mean(snapshots, axis=1)
    analyzed = snapshots - mean[:, None]
    modes, singular_values, right_vectors_transposed = np.linalg.svd(analyzed, full_matrices=False)
    energies = singular_values**2
    energy_fractions = energies / np.sum(energies)
    coefficients = singular_values[:2, None] * right_vectors_transposed[:2]
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        REFERENCE_PATH,
        mean=mean,
        modes=modes[:, :N_SAVED_MODES],
        singular_values=singular_values,
        energy_fractions=energy_fractions,
        cumulative_energy=np.cumsum(energy_fractions),
        leading_coefficients=coefficients,
        runtime_seconds=time.perf_counter() - started,
    )
    print(f"Wrote independent NumPy POD reference: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
