# Mach 0.9 turbulent-jet SPOD validation

This example analyzes the symmetric azimuthal (`m = 0`) pressure component of
the canonical Mach 0.9 turbulent round-jet LES distributed with the original
[MATLAB SPOD implementation](https://github.com/SpectralPOD/spod_matlab).
FluidModes processes 5,000 pressure snapshots with serial batch SPOD and is
compared with an independent Python method-of-snapshots implementation of the
standard unweighted calculation used by `example_2.m`.

The SPOD calculation uses 256-snapshot blocks (PySPOD's `n_dft`), a Hamming
window, 128-snapshot (50%) overlap, temporal-mean subtraction, uniform
spatial weights, and a real one-sided spectrum. These settings match the
unweighted `example_2.m` reference calculation.

## Obtain the data

Clone or download the official repository, then place `jet_data/jetLES.mat`
at:

```text
examples/turbulent_jet_spod/data/jetLES.mat
```

The preparation, reference, plotting, and wrapper stages also accept the file
path as an optional positional argument. The official file is MATLAB v7.3/HDF5.

## Reproduce

Prepare the data, run the independent reference and FluidModes, then compare
and plot the results:

```bash
python examples/turbulent_jet_spod/prepare_data.py
python examples/turbulent_jet_spod/reference_spod.py
python examples/turbulent_jet_spod/run_fluidmodes.py
python examples/turbulent_jet_spod/compare_results.py
python examples/turbulent_jet_spod/make_plots.py
```

For a clone kept outside `data/`, pass its exact path to the relevant commands:

```bash
python examples/turbulent_jet_spod/prepare_data.py /path/to/spod_matlab/jet_data/jetLES.mat
python examples/turbulent_jet_spod/reference_spod.py /path/to/spod_matlab/jet_data/jetLES.mat
python examples/turbulent_jet_spod/make_plots.py /path/to/spod_matlab/jet_data/jetLES.mat
python examples/turbulent_jet_spod/validate.py /path/to/spod_matlab/jet_data/jetLES.mat
```

Preparation writes ignored `data/jet_spod.csv` with one time snapshot per row:
`time, q1, ..., q6825`. Each MATLAB `p[k, :, :]` field is flattened in C
order, so `r` varies fastest; validation restores a mode with
`reshape((39, 175), order="C")` and uses the original `x` and `r` arrays for
plots. The script checks the prepared CSV through the released FluidModes
tabular reader before running SPOD.

`validate.py` computes the independent Python method-of-snapshots reference
following the mathematics and conventions of the original `spod.m`, then runs:

```bash
fluidmodes spod examples/turbulent_jet_spod/data/jet_spod.csv \
    --block-size 256 --overlap 0.5 --modes 2 \
    --output examples/turbulent_jet_spod/results/spod
```

It then compares frequency bins, eigenvalue branches, and phase-invariant
spatial modes. The independent reference is implemented in Python; see
[VALIDATION.md](VALIDATION.md) for the recorded result and references below.

The convenience command runs the same visible stages in order:

```bash
python examples/turbulent_jet_spod/validate.py
```

## What is compared?

```text
pressure snapshots p[time, x, r]
├── reference_spod.py: independent method-of-snapshots SPOD
└── run_fluidmodes.py: fluidmodes spod --block-size 256 --overlap 0.5 --modes 2
         ↓
compare_results.py: frequency bins, normalized eigenvalue branches, and complex modes
```

## Interpreting the results

At each resolved frequency, SPOD ranks coherent structures by spectral energy.
The first eigenvalue branch is the most energetic coherent structure at that
frequency and the second branch is the next most energetic; these values are
not global POD energy fractions. With a block size of 256 and the dataset time
step `dt`, the frequency resolution is `Δf = 1 / (256 dt)`.

Each SPOD mode is a complex coherent spatial structure associated with one
discrete frequency bin. The curated mode figure displays its real part; the
unshown imaginary part is the phase-shifted quadrature component of the same
mode, not a second physical field. The plots show the expected jet wavepacket
behaviour, with streamwise wavelength changing with frequency.

The independent reference uses a different Hamming-window normalization.
The comparison applies the analytic conversion between the two conventions.
Eigenvalue agreement at about `1e-14` and correlations of `1.0` show agreement
up to the arbitrary global complex phase of each SPOD mode.

## References

- Brès et al., “Importance of the nozzle-exit boundary-layer state in subsonic turbulent jets,” *Journal of Fluid Mechanics* 851, 83–124, 2018.
- Schmidt et al., “Spectral analysis of jet turbulence,” *Journal of Fluid Mechanics* 855, 953–982, 2018.
- Towne, Schmidt, and Colonius, “Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis,” *Journal of Fluid Mechanics* 847, 821–867, 2018.
