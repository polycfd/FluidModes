"""Compute an independent NumPy-SVD POD reference for prepared wake data."""

from __future__ import annotations

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY, build_pod_ensemble, load_cylinder_data


REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "pod_reference.npz"


def main() -> None:
    """Load prepared snapshots, perform POD directly, and save reference arrays."""
    snapshots, spatial_shape, _ = load_cylinder_data()
    pod_snapshots = build_pod_ensemble(snapshots, spatial_shape)
    mean = np.mean(pod_snapshots, axis=1)
    analyzed = pod_snapshots - mean[:, None]  # X_a = X - X_bar
    modes, singular_values, right_vectors_transposed = np.linalg.svd(analyzed, full_matrices=False)  # X_a = U Σ V^T
    energies = singular_values**2
    energy_fractions = energies / np.sum(energies)
    coefficients = singular_values[:, None] * right_vectors_transposed
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE_PATH, mean=mean, modes=modes, singular_values=singular_values, energies=energies, energy_fractions=energy_fractions, coefficients=coefficients)
    print(f"Wrote independent POD reference: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
