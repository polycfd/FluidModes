# Experimental cylinder-PIV validation record

Generated with `python examples/experimental_cylinder_piv/validate.py`. Source
data and regenerated numerical outputs are ignored; the four figures in
[`plots/`](plots/) are retained for visual inspection.

## Dataset and benchmark

- Local source: `data/cylinder_vel.mat`; MD5
  `4cc876439c48f7970afa477ea17d217a`, matching Shang and Tu's Zenodo release,
  DOI [10.5281/zenodo.20765567](https://doi.org/10.5281/zenodo.20765567).
- MATLAB v5: `u`, `v` are `(135, 80, 8000)` `float64`; `x`, `y` are `(135, 80)`
  `float64`. Time is axis 2; all velocity values are finite.
- The grid is regular/rectilinear: `x` varies along axis 0, decreasing from
  `0.0827563` to `-0.0579908 m` (`dx=-0.00105035 m`); `y` varies along axis 1,
  increasing from `-0.0671121` to `0.0158657 m` (`dy=0.00105035 m`). The
  plotting reshape preserves this geometry, with the cylinder at low `x`.
- Published zero values are retained with no interpolation or synthetic mask.
  There are 365 joint `u`/`v` zero DOFs at all times; the component masks differ
  by 247 entries over the full record and are not individually time-constant.
- State is `[u.ravel(order="C"); v.ravel(order="C")]`, with `y` fastest:
  `(21600, 2000)`, 329.6 MiB. The first 2,000 frames use `dt=0.05 s`, cover
  100 s, and have `df=0.01 Hz` / 10-Hz Nyquist frequency.

## Paper audit

Tu, Rowley, Kutz, and Shang, “Spectral analysis of fluid flows using
sub-Nyquist-rate PIV data,” *Experiments in Fluids* 55, 1805 (2014), DOI
[10.1007/s00348-014-1805-6](https://doi.org/10.1007/s00348-014-1805-6), reports
8,000 acquisitions, 20 Hz, and `135 x 80` vectors (PDF p. 9). It explicitly
uses the **first 2,000** image pairs (p. 10); Sec. 2.4 concatenates velocity
components into the PIV state. It uses SVD-based DMD, reports a 0.889-Hz wake
peak and weak harmonics, and shows a leading POD pair with approximately 75%
energy by 12 modes (Figs. 8--11). It does not state DMD rank or explicit mean
treatment; its DMD spectrum uses a separate cited modal-energy scaling.

Scherl et al., “Robust principal component analysis for modal decomposition of
corrupt fluid flows,” *Physical Review Fluids* 5, 054401 (2020), DOI
[10.1103/PhysRevFluids.5.054401](https://doi.org/10.1103/PhysRevFluids.5.054401),
documents the same `135 x 80`, 20-Hz, `m=8000` record (PDF p. 9), but studies
RPCA and does not define a competing 2,000-frame standard-DMD benchmark. It
discusses fluctuation POD after mean removal (p. 7) and shows experimental
PCA/RPCA vorticity modes (Fig. 10).

The 2014 paper therefore establishes the selected window. POD removes the
temporal mean exactly once in each path; DMD uses raw velocity snapshots. Rank
12 is a fixed FluidModes convention motivated by the paper's 12-mode POD basis
for compressed sensing, **not** a claimed literature DMD rank.

## Numerical implementation validation

The independent POD is direct NumPy SVD of `X - mean(X)`; the CLI is:

```text
fluidmodes pod examples/experimental_cylinder_piv/data/piv_velocity.csv \
  --output examples/experimental_cylinder_piv/results/pod
```

- Leading-20 maximum singular-value relative error: `1.187e-15`.
- Leading-20 energy/cumulative-energy errors: `1.110e-16` / `2.220e-16`.
- Sign-invariant mode and paired-subspace correlations are 1.0 to displayed
  precision.

The independent DMD is NumPy textbook exact DMD (`X1`, `X2`, SVD, reduced
operator, eigenpairs, exact modes), rank 12; the CLI is:

```text
fluidmodes dmd examples/experimental_cylinder_piv/data/piv_velocity.csv \
  --rank 12 --output examples/experimental_cylinder_piv/results/dmd
```

- Maximum matched eigenvalue error: `1.776e-15`.
- Maximum signed-frequency / growth-rate errors: `1.332e-15 Hz` /
  `3.553e-14 s^-1`.
- All 12 phase-invariant complex-mode correlations are 1.0 to displayed
  precision; matching is explicit and order-independent.

## Physical interpretation

- `e1=0.326060`, `e2=0.316079`, `E2=0.642139`; the leading pair visibly
  represents quadrature phases of the alternating wake.
- `E12=0.764889`; 10 modes first reach 75%, versus the paper's approximate
  12-mode statement. Since NumPy and FluidModes agree at roundoff and the
  paper does not fully specify weighting/preprocessing, this is not an error.
- The leading POD-coefficient NumPy FFT peaks independently at `0.890000 Hz`,
  without using `0.889 Hz` as a selection criterion. This is the direct
  physical-frequency check against the published wake-shedding value.
- The predeclared DMD correspondence selects the positive-frequency exact-DMD
  mode nearest the literature shedding frequency of `0.889 Hz`; it is therefore
  a comparison with the known shedding mode, not an independent
  frequency-identification criterion, and is not selected by raw amplitude.
  Reference and FluidModes both place this mode at `0.897480 Hz` (mode 3),
  agreeing to `7.772e-16 Hz` and differing from the literature value by
  `0.008480 Hz` (0.954%). For context, the 100-s record has an ordinary
  Fourier-bin spacing of `0.01 Hz`; DMD frequencies are continuous values from
  the eigenvalue phase and are not restricted to that spacing.
- Rank 12 has no clearly supported harmonics: its nearest `2f` candidate is
  1.694146 Hz and no distinct `3f` candidate occurs. None are claimed.
- Derived-vorticity POD/DMD plots show the coherent alternating wake downstream
  of the cylinder, qualitatively consistent with 2014 Figs. 8--9. Vorticity is
  for plotting only; decompositions use `[u; v]`.

## Figures

- `experimental_snapshot.png`: measured velocity magnitude and mask.
- `pod_energy.png`: energy/cumulative energy, leading pair, mode 12, 75% line.
- `pod_leading_pair.png`: reference POD pair as derived vorticity (sign free).
- `dmd_shedding.png`: raw FluidModes amplitude, literature/selected frequency,
  and selected DMD mode's vorticity real part. Raw amplitude is not the paper's
  separately scaled modal energy.

All four figures were visually inspected for orientation, coordinates, labels,
mask location, and plausible wake structure.

## Footprint and reproducibility

- MAT: 1,082,404,799 bytes; CSV: 913,792,365 bytes; references: 7.5 MiB; CLI
  results: 1.1 GiB; retained plots: 336 KiB.
- Calculation-only runtimes: independent POD/DMD 9.18/9.56 s; FluidModes
  POD/DMD 34.29/41.03 s.
- MAT/CSV, references, and results are ignored; local PDFs remain untracked.
  Nothing under `src/fluidmodes/` or `tests/` changed.
