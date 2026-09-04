# Cylinder-wake validation results

Generated with:

```bash
python examples/cylinder_wake/validate.py
```

This records a completed external validation run. The third-party dataset and
the large generated CSV, reference, and result files are not bundled with the
public example.

## Dataset

- Source: `DATA/FLUIDS/CYLINDER_ALL.mat` from the companion `DATA.zip` archive at http://dmdbook.com/DATA.zip.
- Associated reference: J. N. Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor, *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*, SIAM, 2016.
- Case: two-dimensional cylinder wake at `Re = 100`; vorticity variable `VORTALL`.
- Spatial dimensions: `m = nx = 199`, `n = ny = 449` (`89,351` DOFs); snapshots: `151`; spacing: `0.2`.
- MATLAB variables inspected: `UALL` (89351, 151) (float64); `UEXTRA` (89351, 1) (float64); `VALL` (89351, 151) (float64); `VEXTRA` (89351, 1) (float64); `VORTALL` (89351, 151) (float64); `VORTEXTRA` (89351, 1) (float64); `m` (1, 1) (uint8); `n` (1, 1) (uint16); `nx` (1, 1) (uint8); `ny` (1, 1) (uint16).

## POD

### Reference calculation and CLI invocation

`reference_pod.py` builds the symmetry-augmented ensemble directly from raw
`VORTALL` and computes its mean-subtracted economical NumPy SVD.
`run_fluidmodes.py` invokes
`fluidmodes pod data/cylinder_pod.csv --output results/pod`; no option disables
the CLI's default temporal-mean subtraction.

### Compared quantities and result

- Input: `Y = [X, X_mirror]`, with `X_mirror` formed by `reshape(..., (199, 449), order="F")`, cross-stream (`axis=0`) reflection, vorticity sign reversal, and MATLAB-order flattening. FluidModes then applies its default temporal mean subtraction once.
- Leading energy fractions: 0.414160, 0.400994, 0.053509, 0.052743, 0.032825, 0.031436, 0.005237, 0.005159.
- Maximum relative singular-value discrepancy (first 20): `6.445e-15`.
- Maximum absolute energy-fraction discrepancy (first 20): `1.527e-16`.
- Absolute spatial-mode correlations (modes 1--6): 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000.
- Paired-subspace correlations for modes (1, 2), (3, 4), and (5, 6): 1.000000000000, 1.000000000000, 1.000000000000.
- Discrepancies near `1e-14` are floating-point-level agreement with the
  independent SVD reference. Correlation `1.0` permits the arbitrary POD
  sign. Paired-subspace comparisons are reported because individual vectors
  within a nearly degenerate pair are not unique, whereas their two-dimensional
  subspace is. The leading fluctuation modes form near-energy-degenerate pairs
  and show alternating wake structures in the validation plots.

## DMD

### Reference calculation and CLI invocation

`reference_dmd.py` computes textbook exact DMD directly from raw `VORTALL`
with rank 21. `run_fluidmodes.py` invokes
`fluidmodes dmd data/cylinder_dmd.csv --rank 21 --output results/dmd`; no
mean-subtraction option is supplied.

### Compared quantities and result

- Input: raw `VORTALL`, no mean subtraction; exact DMD rank: `21`.
- Maximum matched discrete-eigenvalue difference: `3.637e-15`.
- Maximum matched signed-frequency difference: `5.551e-16`.
- Maximum matched growth-rate difference: `1.815e-14`.
- Representative phase-invariant mode correlations: 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000.
- Dominant positive fundamental-band DMD frequency: `0.165396`.
- Reference cylinder-wake shedding frequency: approximately `0.16`; absolute difference: `0.005396`.
- Clearly resolved positive-frequency harmonics: 2f = 0.330793, 3f = 0.496189, 4f = 0.661585, 5f = 0.826982, 6f = 0.992378, 7f = 1.157782, 8f = 1.323182, 9f = 1.488645, 10f = 1.654103.

The numerical discrepancies are again at floating-point precision. DMD modes
are complex, so correlation `1.0` permits arbitrary complex phase. The
comparison with `St ≈ 0.16` is a physical benchmark rather than a
machine-exact reference value.

## Figures

- `pod_energy_spectrum.png` shows the paired leading POD energies.
- `pod_leading_modes.png` shows signed vorticity patterns; a sign reversal of
  an individual POD mode is equivalent.
- `dmd_eigenvalues.png` places the rank-21 spectrum against the unit circle,
  and `dmd_shedding_mode.png` shows the real and imaginary quadrature
  components of the dominant shedding mode.

## Reproducibility

- FluidModes: `0.1.0`; NumPy: `1.26.4`; PyDMD: `2025.8.1`; SciPy: `1.11.3`.
- Raw `.mat`: 287.2 MiB; DMD CSV: 239.4 MiB; POD CSV: 478.1 MiB; POD results: 208.6 MiB; DMD results: 234.6 MiB; complete example: 1685.0 MiB.
