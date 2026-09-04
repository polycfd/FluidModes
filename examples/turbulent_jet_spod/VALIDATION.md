# Turbulent-jet SPOD validation results

Generated with:

```bash
python examples/turbulent_jet_spod/validate.py
```

This is a completed external validation run. Third-party data, reference code,
the generated CSV, and numerical caches are ignored and are not distributed by
FluidModes.

## Dataset

- Source: [`SpectralPOD/spod_matlab`](https://github.com/SpectralPOD/spod_matlab), `jet_data/jetLES.mat`; its Caltech license permits academic and other non-commercial use subject to its conditions.
- Case: Mach 0.9 turbulent round-jet LES, symmetric azimuthal component `m = 0`; pressure field `p`.
- MATLAB variables inspected: `dt` (1, 1) (float64); `m` (1, 1) (float64); `nr` (1, 1) (float64); `nt` (1, 1) (float64); `nx` (1, 1) (float64); `p` (5000, 39, 175) (float32); `p_mean` (39, 175) (float32); `r` (39, 175) (float32); `x` (39, 175) (float32).
- MATLAB layout: `p` is `(5000, 39, 175)` = `(time, x, r)`; `x` and `r` are `(39, 175)`. There are `5,000` real-valued snapshots and `6,825` spatial DOFs; `dt = 0.20000000000000001`.
- CSV mapping: each `p[k, :, :]` is flattened in C order (`r` fastest) into one CSV row; modes reshape with the inverse `reshape((39, 175), order='C')`.

## SPOD parameters

- `n_dft = 256`; standard Hamming window; 50% overlap = `128` snapshots; `38` Welch blocks.
- Long-time temporal mean subtraction; uniform spatial weights; real one-sided spectrum; `129` frequency bins.
- `Δf = 0.01953125`; maximum frequency-vector discrepancy: `0.000e+00`, so every FluidModes frequency bin is identical to its independent-reference counterpart.

## Quantitative comparison

### Reference calculation and CLI invocation

`reference_spod.py` independently forms overlapping Hamming-windowed Welch
blocks, applies the method of snapshots, and saves the one-sided reference.
`run_fluidmodes.py` invokes
`fluidmodes spod data/jet_spod.csv --block-size 256 --overlap 0.5 --modes 2`.
`compare_results.py` then reads the separate reference and CLI output files.

### Compared quantities and result

- Raw PySPOD/FluidModes eigenvalues have a global scale factor `0.026692737402151247` relative to the original-MATLAB convention. PySPOD applies the raw Hamming window with gain correction `FFT(wq) / sum(w)`, whereas `spod.m` normalizes the window to unit power and uses `sqrt(dt) FFT(wq)`. The squared ratio of those Fourier amplitudes is the stated multiplicative factor, so dividing FluidModes eigenvalues by it is an analytical normalization conversion rather than an empirical fit; no spectra were peak-normalized.
- Raw leading/second-branch maximum relative differences: `9.733e-01`, `9.733e-01`.
- After the analytical scaling conversion, leading/second-branch maximum relative differences: `6.422e-15`, `7.694e-15`, which is floating-point-level spectral agreement.
- Representative frequency bins (MATLAB indices 10, 15, 20): `10`: f = 0.17578125, `15`: f = 0.27343750, `20`: f = 0.37109375. These sample low, intermediate, and higher resolved frequencies in the plotted jet wavepacket sequence.
- First/second mode phase-invariant correlations: f=0.17578125: (1.000000000000, 1.000000000000); f=0.27343750: (1.000000000000, 1.000000000000); f=0.37109375: (1.000000000000, 1.000000000000). A correlation of `1.0` permits an arbitrary global complex phase.
- Two-mode subspace correlations: f=0.17578125: 1.000000000000; f=0.27343750: 1.000000000000; f=0.37109375: 1.000000000000. Individual modes are unique enough here that no subspace substitution was needed.
- The retained real-part mode plot shows the expected wavepacket-like structures; their streamwise wavelength shortens as frequency increases.

## Figures

`spod_spectrum.png` compares the first two FluidModes and independent-reference
eigenvalue branches after the analytical normalization conversion. Its axes
are frequency and SPOD modal energy. `spod_modes.png` shows the real part of
the first two complex modes at the three listed frequency bins in physical
streamwise (`x`) and radial (`r`) coordinates.

## Reference implementation

- The independent Python reference uses the method of snapshots, explicit Hamming-window formula, global mean removal, `sqrt(dt)` FFT scaling, uniform weights, and one-sided doubling exactly as in the original `spod.m` formulation. It does not import PySPOD or FluidModes internals.
- Neither MATLAB nor GNU Octave was available; only the independent Python reference following spod.m was run.

## Reproducibility

- FluidModes `0.1.0`; PySPOD `2.0.0`; NumPy `1.26.4`; SciPy `1.11.3`.
- CLI command: `fluidmodes spod examples/turbulent_jet_spod/data/jet_spod.csv --block-size 256 --overlap 0.5 --modes 2 --output examples/turbulent_jet_spod/results/spod`.
- Generated CSV: 614.4 MiB; reference cache: 0.7 MiB; FluidModes output: 27.1 MiB; curated plots: 0.3 MiB; public example footprint: 0.3 MiB. The local source clone and all generated working artifacts are excluded by Git.
