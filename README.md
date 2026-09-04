# FluidModes

FluidModes analyzes time-resolved experimental and numerical fluid-dynamics
data with proper orthogonal decomposition (POD), spectral proper orthogonal
decomposition (SPOD), and dynamic mode decomposition (DMD). The command-line
tool provides standard POD, serial batch SPOD, and exact DMD for tabular data
and grayscale PNG sequences.

## Snapshot and tabular data

FluidModes stores data as a snapshot matrix `X[n_dof, n_time]`: rows are
spatial or state degrees of freedom and columns are time snapshots. CSV and
whitespace-delimited TXT input use a headered table with time in the first
column and one degree of freedom per remaining column:

```text
time q1 q2
0.0  1.0 10.0
0.5  2.0 20.0
```

## Verification

FluidModes' POD, DMD, and SPOD paths are tested against deterministic
manufactured datasets. POD recovers prescribed modal energies; DMD recovers
prescribed eigenvalues, signed frequencies, and growth rates; and SPOD
recovers prescribed spectral peaks and coherent spatial modes.

## Validation examples

The [cylinder-wake example](examples/cylinder_wake/) analyzes POD and DMD of
a canonical `Re = 100` numerical cylinder wake. It compares FluidModes with
independent reference calculations and the expected vortex-shedding dynamics.

The [turbulent-jet SPOD example](examples/turbulent_jet_spod/) analyzes the
Mach 0.9 turbulent-jet LES dataset with SPOD and compares FluidModes with an
independent method-of-snapshots SPOD implementation.

The [experimental cylinder-PIV example](examples/experimental_cylinder_piv/)
analyzes POD and DMD of 2,000 time-resolved `Re = 413` PIV snapshots. It uses
two-component `[u; v]` velocity states and independent NumPy POD/exact-DMD
references. The leading POD coefficient peaks at `0.890 Hz`, close to the
published `0.889 Hz` vortex-shedding frequency, and the fixed rank-12 DMD
result contains the coherent wake mode at `0.89748 Hz`.

## PNG sequences and preprocessing

PNG sequences are read as 8-bit grayscale scalar fields (Pillow mode `L`),
with one pixel per degree of freedom. Frames must have matching dimensions and
are naturally sorted; `--dt` defines `t[k] = k * dt`. FluidModes analyzes
pixel values directly as `float64` values from 0 to 255. POD and SPOD subtract
the temporal mean by default; `--no-subtract-mean` analyzes the supplied
snapshots directly.

## POD

FluidModes computes standard economical-SVD POD. For the analyzed snapshot
matrix (raw data or fluctuations) `X_a`, it uses

```text
X_a = U Σ Vᵀ
```

For retained rank `r`, spatial modes are `Φ = U[:, :r]` and temporal
coefficients are `A = Σ_r V_rᵀ`, so `X_a,r = Φ A`. Modal energy is the squared
singular value. Each reported fractional energy is relative to the full
analyzed matrix, so a truncated result need not sum to one.

POD subtracts the temporal mean by default. Use `--no-subtract-mean` to
analyze the raw snapshot matrix. When a mean is removed, `mean.npy` can be
added to a retained analyzed reconstruction to recover the corresponding
original-field reconstruction.

Run POD on one tabular file, choosing a retained rank if desired:

```bash
fluidmodes pod snapshots.csv --rank 10 --output pod_output
fluidmodes pod snapshots.txt --no-subtract-mean --output pod_output
```

Or run POD on an explicit PNG sequence; PNG input always requires `--dt`:

```bash
fluidmodes pod frames/*.png --dt 0.001 --rank 5 --plot-modes 1 2
```

A POD run writes retained singular values and energy fractions
(`singular_values.csv`, `energy.csv`), time-indexed coefficients
(`coefficients.csv`), spatial modes (`modes.npy`), and an energy plot
(`energy.png`). It also writes `mean.npy` when mean subtraction is used.
Optional `--plot-modes` and `--plot-coefficients` write the selected one-based
mode plots.

`modes.npy` has shape `(n_dof, r)`, where `r` is the retained rank. Each column
is one spatial POD mode, and rows follow the input DOF/state-column order. For
image data, `spatial_shape` defines how each mode is reshaped. When present,
`mean.npy` contains one value per DOF.

The deterministic manufactured POD case has three orthonormal spatial modes
with energy fractions `0.6`, `0.3`, and `0.1`; it is used by the test suite to
verify POD reconstruction and truncation.

## SPOD

FluidModes runs serial batch SPOD through
[PySPOD](https://github.com/MathEXLab/PySPOD). SPOD is intended for
statistically stationary data and requires uniformly sampled snapshots. For a
block size `N_block` and time step `dt`, its frequency resolution is

```text
Δf = 1 / (N_block * dt)
```

`--block-size` is required and is the number of snapshots in each spectral
block (PySPOD's `n_dft`). `--overlap` is a FluidModes fractional overlap from
0 (inclusive) to 1 (exclusive), defaulting to `0.5`; FluidModes converts it to
the percentage expected by PySPOD. SPOD uses a fixed Hamming window.
Long-time temporal mean subtraction is enabled by default; use
`--no-subtract-mean` to disable it. Real-valued input uses a one-sided,
non-negative spectrum. Frequencies are reported in cycles per input-time unit,
which are Hz only when the input time unit is seconds.

`--modes` selects the number of leading spatial SPOD modes saved at every
frequency (default `1`); it is not a global rank. SPOD eigenvalues are modal
energies at each frequency and are not global energy fractions.

```bash
fluidmodes spod snapshots.csv --block-size 100 --output spod_output
fluidmodes spod snapshots.txt --block-size 100 --overlap 0.5 --modes 2
fluidmodes spod frames/*.png --dt 0.01 --block-size 100 --plot-frequency 5 --plot-modes 1
```

An SPOD run writes `eigenvalues.csv` and `eigenvalues.npy` with one row per
frequency and a leading
`frequency_cycles_per_input_time` column, `modes.npy`, and `spectrum.png`.
`mean.npy` is written only if mean subtraction was used. `modes.npy` has shape
`(n_frequency, n_dof, n_modes_saved)`; its modes are complex and therefore
selected plots show their real and imaginary parts. Requested plot frequencies
are matched to the nearest available spectral bin.

The deterministic manufactured SPOD case contains coherent standing-wave
structures at 5 and 12 cycles per input-time unit. With `--block-size 100` and
`dt = 0.01`, both lie exactly on bins with `Δf = 1` and are used for numerical
verification.

## DMD

FluidModes runs [PyDMD](https://github.com/PyDMD/PyDMD)'s standard exact-DMD
implementation. It approximates the snapshot evolution as `x[k+1] ≈ A x[k]`.
For each discrete eigenvalue `λ`, FluidModes
reports the continuous-time eigenvalue

```text
ω = log(λ) / dt
```

using the principal complex logarithm, along with growth rate
`σ = real(ω)` (inverse input-time units) and frequency
`f = imag(ω) / (2π)` (cycles per input-time unit). Frequencies retain their
sign; when the input time unit is seconds, they are in Hz.

DMD requires at least two uniformly sampled snapshots. For CSV and TXT input,
time is read from the first column and must be uniformly spaced. For PNG input,
provide `--dt`. Mean subtraction is disabled by default because it changes the
linear dynamics being approximated; enable it explicitly with
`--subtract-mean`. A positive `--rank` requests fixed SVD truncation; without
it, DMD uses all available DMD snapshot-matrix directions.

```bash
fluidmodes dmd snapshots.csv --output dmd_output
fluidmodes dmd snapshots.txt --rank 10 --subtract-mean --output dmd_output
fluidmodes dmd frames/*.png --dt 0.001 --plot-modes 1 2 --output dmd_output
```

Each DMD run writes `eigenvalues.csv`, `eigenvalues.npy`, `modes.npy`,
`amplitudes.npy`, `dynamics.npy`, and `reconstruction.npy`, plus an eigenvalue
plot (`eigenvalues.png`). These NPY arrays are generally
complex-valued. `eigenvalues.npy` has shape `(n_modes,)` and contains the
discrete eigenvalue array itself. `modes.npy` has shape `(n_dof, n_modes)`,
`amplitudes.npy` has shape `(n_modes,)`, and both `dynamics.npy` and
`reconstruction.npy` have shape `(n_modes, n_time)` and `(n_dof, n_time)`,
respectively. `mean.npy` is written only when mean subtraction was used.
`eigenvalues.csv` reports the discrete eigenvalue real and imaginary
components, `growth_rate_per_input_time`, signed
`frequency_cycles_per_input_time`, and amplitude. The growth-rate and
frequency summary is in this CSV file; it is not part of `eigenvalues.npy`.
Selected mode plots show the real and imaginary components separately. 

The deterministic manufactured DMD case comprises two real-valued conjugate
mode pairs with prescribed eigenvalues, signed frequencies, and growth rates;
the test suite verifies their recovery and the full-rank reconstruction.

## Installation

FluidModes requires Python 3.10 or later. Clone the repository and install
FluidModes with pip:

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e '.[dev]'
```

## Command line

```bash
fluidmodes --help
fluidmodes pod --help
fluidmodes dmd --help
fluidmodes spod --help
```

`pod`, `spod`, and `dmd` are available for CSV, TXT, and grayscale PNG
sequences.

## Author

[Fabian Denner](https://www.polymtl.ca/expertises/en/denner-fabian)

## License

FluidModes is released under the [MIT License](LICENSE).
