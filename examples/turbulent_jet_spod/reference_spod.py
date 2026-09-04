"""Independent unweighted batch-SPOD reference following MATLAB ``spod.m``."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from prepare_data import DEFAULT_MAT_PATH, EXAMPLE_DIRECTORY, load_jet_data


BLOCK_SIZE = 256
OVERLAP = 128
SELECTED_FREQUENCY_INDICES = (9, 14, 19)  # MATLAB example_2.m indices 10, 15, 20.


def hamming_window(n_dft: int) -> np.ndarray:
    """Return the standard Hamming window used by the original ``spod.m``."""
    indices = np.arange(n_dft, dtype=float)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * indices / (n_dft - 1))


def reference_scale_to_pyspod(window: np.ndarray, time_step: float) -> float:
    """Return the eigenvalue factor converting MATLAB-SPOD to PySPOD scaling.

    MATLAB normalizes the window to unit power and uses ``sqrt(dt) * fft``.
    PySPOD uses the raw Hamming window and its gain correction
    ``fft(window * q) / sum(window)``.  The returned squared amplitude ratio
    therefore maps MATLAB-reference eigenvalues to raw PySPOD eigenvalues.
    """
    amplitude_ratio = np.sqrt(np.sum(window**2)) / (np.sqrt(time_step) * np.sum(window))
    return float(amplitude_ratio**2)


def compute_reference(
    snapshots: np.ndarray,
    time_step: float,
    *,
    block_size: int = BLOCK_SIZE,
    overlap: int = OVERLAP,
    selected_frequency_indices: tuple[int, ...] = SELECTED_FREQUENCY_INDICES,
    cache_directory: Path | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Compute the standard one-sided snapshot-method SPOD reference.

    ``snapshots`` follows MATLAB's time-first ``(n_time, n_x, n_r)`` layout.
    The calculation retains only modes needed for the documented comparison.
    """
    if snapshots.ndim != 3:
        raise ValueError("Reference SPOD expects snapshots with shape (time, x, r).")
    n_time = snapshots.shape[0]
    n_blocks = (n_time - overlap) // (block_size - overlap)
    if n_blocks < 2:
        raise ValueError("Block size and overlap must provide at least two Welch blocks.")
    n_frequency = block_size // 2 + 1
    n_dof = int(np.prod(snapshots.shape[1:]))
    window = hamming_window(block_size)
    unit_power_window = window / np.sqrt(np.sum(window**2))
    mean = np.mean(snapshots, axis=0)
    eigenvalues = np.empty((n_frequency, n_blocks), dtype=float)
    selected_modes = np.empty(
        (len(selected_frequency_indices), n_dof, 2), dtype=complex
    )

    # Cache all Welch-block Fourier realizations. Each block is mean-centered,
    # tapered with the unit-power Hamming window, and transformed with the
    # MATLAB reference's ``sqrt(dt)`` convention.
    with tempfile.TemporaryDirectory(prefix="jet-spod-reference-", dir=cache_directory) as temporary_directory:
        q_hats = np.memmap(
            Path(temporary_directory) / "q_hats.dat",
            dtype=np.complex128,
            mode="w+",
            shape=(n_blocks, n_frequency, n_dof),
        )
        for block_index in range(n_blocks):
            offset = min(block_index * (block_size - overlap) + block_size, n_time) - block_size
            block = snapshots[offset:offset + block_size] - mean
            transformed = np.sqrt(time_step) * np.fft.rfft(block * unit_power_window[:, None, None], axis=0)
            q_hats[block_index] = transformed.reshape(n_frequency, n_dof, order="C")
        q_hats.flush()

        # The method of snapshots diagonalizes the n_blocks-by-n_blocks
        # realization correlation instead of forming the much larger spatial
        # cross-spectral-density matrix. The nonzero eigenvalues are shared.
        for frequency_index in range(n_frequency):
            q_hat = np.asarray(q_hats[:, frequency_index, :]).T
            correlation = q_hat.conj().T @ q_hat / n_blocks
            values, vectors = np.linalg.eigh(correlation)
            order = np.argsort(values)[::-1]
            values = np.maximum(values[order].real, 0.0)
            vectors = vectors[:, order]
            eigenvalues[frequency_index] = values

            selected_position = selected_frequency_indices.index(frequency_index) if frequency_index in selected_frequency_indices else None
            if selected_position is not None:
                positive = values[:2] > np.finfo(float).eps * values[0]
                modes = np.zeros((n_dof, 2), dtype=complex)
                modes[:, positive] = q_hat @ vectors[:, :2][:, positive] / np.sqrt(n_blocks * values[:2][positive])
                selected_modes[selected_position] = modes
        del q_hats

    # Folded negative frequencies contribute equal energy to each nonzero,
    # non-Nyquist bin of this real-data one-sided spectrum. Normalized modes
    # themselves are unchanged by this eigenvalue correction.
    eigenvalues[1:-1] *= 2.0
    frequencies = np.arange(n_frequency, dtype=float) / (block_size * time_step)
    return {
        "frequencies": frequencies,
        "eigenvalues": eigenvalues,
        "modes": selected_modes,
        "mean": mean.reshape(-1, order="C"),
        "window": window,
        "n_blocks": n_blocks,
        "scale_to_pyspod": reference_scale_to_pyspod(window, time_step),
    }


def write_reference(mat_path: Path = DEFAULT_MAT_PATH) -> None:
    """Load pressure snapshots, compute the independent reference, and save it."""
    snapshots, _, _, time_step, _ = load_jet_data(mat_path)
    reference_directory = EXAMPLE_DIRECTORY / "reference"
    reference_directory.mkdir(parents=True, exist_ok=True)
    reference = compute_reference(snapshots, time_step, cache_directory=reference_directory)
    np.savez(reference_directory / "reference_spod.npz", **reference)
    print(f"Wrote independent method-of-snapshots SPOD reference: {reference_directory / 'reference_spod.npz'}")


def main() -> None:
    """Run the independent SPOD reference stage from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", nargs="?", type=Path, default=DEFAULT_MAT_PATH)
    args = parser.parse_args()
    write_reference(args.mat_path)


if __name__ == "__main__":
    main()
