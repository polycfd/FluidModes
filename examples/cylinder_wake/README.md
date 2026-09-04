# Cylinder-wake POD and DMD validation

This example analyzes POD and DMD of the canonical two-dimensional vorticity
field behind a circular cylinder at `Re = 100`. The dataset accompanies J. N.
Kutz, S. L. Brunton, B. W. Brunton, and J. L. Proctor, *Dynamic Mode
Decomposition: Data-Driven Modeling of Complex Systems*, SIAM, 2016. FluidModes
results are compared with independent NumPy POD and exact-DMD calculations.

Download `CYLINDER_ALL.mat` from http://dmdbook.com/DATA.zip, specifically
`DATA/FLUIDS/CYLINDER_ALL.mat`, and place it at:

```text
examples/cylinder_wake/data/CYLINDER_ALL.mat
```

Create the destination directory if needed:

```bash
mkdir -p examples/cylinder_wake/data
```

The repository keeps this third-party MATLAB file outside the tracked example
files. The archive's `DATA/FLUIDS/README.txt` identifies it as a 151-snapshot
`Re = 100` cylinder-wake dataset generated with the Immersed Boundary
Projection Method (IBPM). It records the two-dimensional vorticity field as
the wake develops its familiar periodic vortex-shedding motion.

The supplied MATLAB file contains `VORTALL` with shape `(89351, 151)`,
`m = nx = 199`, and `n = ny = 449`, so each snapshot has
`199 * 449 = 89351` degrees of freedom. It contains no coordinate arrays. The data were simulated with a
time step of `0.02` and saved every 10 simulation steps, so the snapshot
spacing here is `0.2`.

## Analysis

POD analyzes a symmetry-augmented ensemble. Each vorticity snapshot is
reshaped as `(199, 449)` with `order="F"`, reflected across the cross-stream
direction (`axis=0`), sign-reversed, and flattened with `order="F"`. The
prepared ensemble is `[X, X_mirror]`; its reflected snapshots use a synthetic
strictly increasing time index with the same `0.2` spacing. FluidModes then
applies its standard temporal-mean subtraction once.

DMD analyzes raw `VORTALL` at rank 21. It approximates the evolution of the
supplied field, so the calculation uses no temporal-mean subtraction. The
dominant DMD pair has `f = 0.165396`, close to the expected fundamental
vortex-shedding frequency of `St ≈ 0.16`.

## What is compared?

```text
POD: symmetry-augmented wake snapshots
├── reference_pod.py: independent NumPy SVD
└── run_fluidmodes.py: fluidmodes pod
         ↓
compare_results.py: singular values, energy fractions, modes, and paired subspaces

DMD: raw VORTALL snapshots
├── reference_dmd.py: independent rank-21 exact DMD
└── run_fluidmodes.py: fluidmodes dmd --rank 21
         ↓
compare_results.py: eigenvalues, frequencies, growth rates, and complex modes
```

## Interpreting the results

The curated figures in [`plots/`](plots/) show the structures checked by the
quantitative validation.

The POD spectrum shows how fluctuation energy is distributed among spatial
POD modes. Near-equal energetic pairs are expected for a periodic travelling
or oscillatory wake: two spatial patterns are needed to represent the two
quadrature phases of the motion. The positive and negative regions in a POD
mode are opposite-signed vorticity structures. A POD mode may be multiplied
by `-1` without changing the decomposition, so its sign is not physical.

The DMD eigenvalue plot shows discrete eigenvalues in the complex plane.
Points near the unit circle describe weakly growing or decaying oscillatory
structures; conjugate pairs describe real-valued oscillation, and their angle
sets the frequency. The real and imaginary panels of the shedding mode are
quadrature components of one complex mode.

## Public contents

- `prepare_data.py`: creates and verifies the DMD/POD CSV inputs.
- `reference_pod.py` and `reference_dmd.py`: independent NumPy references.
- `run_fluidmodes.py`: invokes only the public FluidModes CLI.
- `compare_results.py`: reads reference and CLI outputs and reports metrics.
- `make_plots.py`: creates the curated figures after comparison.
- `validate.py`: runs those stages in order as a convenience wrapper.
- `VALIDATION.md`: quantitative record of the completed validation.
- `plots/`: curated POD and DMD figures from that validation.
- `.gitignore`: example-local rules for regenerated large artifacts.

Generated CSVs, independent reference arrays, full FluidModes result
directories, and non-curated plots are ignored by Git because they are large
and reproducible.

SciPy loads the MATLAB file and matches DMD eigenvalues in this example.

## Reproduce the validation

Prepare the data, run the independent references and FluidModes, then compare
and plot the results from the repository root:

```bash
python examples/cylinder_wake/prepare_data.py
python examples/cylinder_wake/reference_pod.py
python examples/cylinder_wake/reference_dmd.py
python examples/cylinder_wake/run_fluidmodes.py
python examples/cylinder_wake/compare_results.py
python examples/cylinder_wake/make_plots.py
```

Or run the same sequence with the convenience wrapper:

```bash
python examples/cylinder_wake/validate.py
```

The independent references are written to `reference/`, FluidModes outputs are
written to `results/`, and `compare_results.py` records a machine-readable
summary in `reference/comparison.json`. See `VALIDATION.md` for the recorded
quantitative result.
