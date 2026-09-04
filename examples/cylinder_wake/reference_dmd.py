"""Compute an independent rank-21 textbook exact-DMD wake reference."""

from __future__ import annotations

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY, SNAPSHOT_SPACING, load_cylinder_data


DMD_RANK = 21
REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "dmd_reference.npz"


def main() -> None:
    """Apply exact DMD directly to the raw prepared ``VORTALL`` snapshots."""
    snapshots, _, _ = load_cylinder_data()
    X_1 = snapshots[:, :-1]
    X_2 = snapshots[:, 1:]
    U, singular_values, Vh = np.linalg.svd(X_1, full_matrices=False)  # X_1 = U Σ V*
    U_r = U[:, :DMD_RANK]
    sigma_r = singular_values[:DMD_RANK]
    V_r = Vh[:DMD_RANK].conj().T
    reduced_operator = U_r.conj().T @ X_2 @ V_r / sigma_r[None, :]  # A_tilde = U_r* X_2 V_r Σ_r^-1
    eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
    modes = X_2 @ V_r / sigma_r[None, :] @ eigenvectors  # Φ = X_2 V_r Σ_r^-1 W
    continuous_eigenvalues = np.log(eigenvalues) / SNAPSHOT_SPACING
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE_PATH, singular_values=sigma_r, eigenvalues=eigenvalues, continuous_eigenvalues=continuous_eigenvalues, growth_rates=continuous_eigenvalues.real, frequencies=continuous_eigenvalues.imag / (2.0 * np.pi), modes=modes)
    print(f"Wrote independent exact-DMD reference: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
