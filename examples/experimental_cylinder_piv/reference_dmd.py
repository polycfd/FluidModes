"""Compute an independent rank-12 textbook exact-DMD reference for PIV data."""

from __future__ import annotations

import time

import numpy as np

from prepare_data import EXAMPLE_DIRECTORY, TIME_STEP, load_velocity_snapshots


# The paper selects 12 POD modes as its low-dimensional basis (about 75% of
# energy), but it does not state a DMD rank.  This fixed rank is therefore the
# transparent FluidModes benchmark convention, not a claimed paper setting.
DMD_RANK = 12
REFERENCE_PATH = EXAMPLE_DIRECTORY / "reference" / "dmd_reference.npz"


def main() -> None:
    """Implement exact DMD: X1 -> SVD -> Atilde -> eig -> exact modes."""
    snapshots, _, _, _ = load_velocity_snapshots()
    started = time.perf_counter()
    x_1 = snapshots[:, :-1]
    x_2 = snapshots[:, 1:]
    left_vectors, singular_values, right_vectors_conjugate_transposed = np.linalg.svd(x_1, full_matrices=False)
    u_r = left_vectors[:, :DMD_RANK]
    sigma_r = singular_values[:DMD_RANK]
    v_r = right_vectors_conjugate_transposed[:DMD_RANK].conj().T
    reduced_operator = u_r.conj().T @ x_2 @ v_r / sigma_r[None, :]
    eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
    modes = x_2 @ v_r / sigma_r[None, :] @ eigenvectors
    continuous_eigenvalues = np.log(eigenvalues) / TIME_STEP
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        REFERENCE_PATH,
        singular_values=sigma_r,
        eigenvalues=eigenvalues,
        continuous_eigenvalues=continuous_eigenvalues,
        frequencies=continuous_eigenvalues.imag / (2.0 * np.pi),
        growth_rates=continuous_eigenvalues.real,
        modes=modes,
        runtime_seconds=time.perf_counter() - started,
    )
    print(f"Wrote independent NumPy exact-DMD reference: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
