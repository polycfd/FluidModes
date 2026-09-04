# Experimental cylinder-PIV POD and DMD validation

This example analyzes 2,000 time-resolved PIV snapshots of the wake behind a
cylinder at `Re = 413`. The data are the planar PIV measurements of Shang and
Tu. FluidModes POD and DMD results are compared with independent NumPy
calculations, and the recovered wake frequency is compared with the `0.889 Hz`
value reported by Tu, Rowley, Kutz, and Shang (2014).

Download `cylinder_vel.mat` from the official
[Zenodo DOI 10.5281/zenodo.20765567](https://doi.org/10.5281/zenodo.20765567)
and place it here:

```text
examples/experimental_cylinder_piv/data/cylinder_vel.mat
```

## Data and analysis

The release contains `u`, `v`, `x`, and `y`; it records 8,000 snapshots at
20 Hz (`dt = 0.05 s`). This example uses the first 2,000 consecutive snapshots
(MATLAB indices 1--2000), matching the window explicitly analyzed by Tu et al.
(2014, PDF p. 10). Scherl et al. (2020) separately documents the full
8,000-frame record for its RPCA study.

Each `135 x 80` field is flattened in C order (`y` varies fastest). With
`N = 135 * 80`, each CSV row contains

```text
time, u.ravel(order="C") [N values], v.ravel(order="C") [N values]
```

so the FluidModes snapshot matrix is `[u; v]` with `2N = 21,600` DOFs.
Zero-valued PIV mask regions remain zero throughout the analysis.

POD analyzes velocity fluctuations about the temporal mean: the NumPy
reference subtracts it explicitly, and `fluidmodes pod` uses its default. DMD
analyzes the raw velocity snapshots at fixed rank 12. Direct NumPy SVD and
exact-DMD calculations provide independent numerical references. Rank 12 is
the fixed benchmark truncation used here; the 2014 paper does not prescribe a
DMD rank. Its DMD spectrum uses a separate modal-energy scaling, so raw PyDMD
amplitudes are not equated with the published spectrum.

```text
experimental PIV data
        |
     preparation
      /       \
     /         \
reference      FluidModes CLI
     \         /
      \       /
       comparison
           |
physical/literature cross-check
```

## Run it

Prepare the data, run the independent references and FluidModes, then compare
and plot the results:

```bash
python examples/experimental_cylinder_piv/prepare_data.py
python examples/experimental_cylinder_piv/reference_pod.py
python examples/experimental_cylinder_piv/reference_dmd.py
python examples/experimental_cylinder_piv/run_fluidmodes.py
python examples/experimental_cylinder_piv/compare_results.py
python examples/experimental_cylinder_piv/make_plots.py
```

Or run the same sequence:

```bash
python examples/experimental_cylinder_piv/validate.py
```

`prepare_data.py` checks the MAT format, arrays, time axis, zero-mask behavior,
regular grid, and the full round trip through the FluidModes CSV reader.
`reference_pod.py` is a direct NumPy SVD, `reference_dmd.py` is the textbook
exact-DMD calculation, `run_fluidmodes.py` contains literal public CLI calls,
`compare_results.py` writes a compact JSON result, and `make_plots.py` creates
four retained figures. See [VALIDATION.md](VALIDATION.md) for the completed
numerical record, paper audit, and stated limitations.

## Interpretation

FluidModes and the independent calculations agree to floating-point precision.
The experimental physics provides a separate check: the leading POD coefficient
peaks at `0.890 Hz`, close to the published `0.889 Hz` shedding frequency. The
corresponding DMD wake mode occurs at `0.89748 Hz`.

POD measures fluctuation energy, so a leading near-quadrature pair is the
expected signature of a coherent oscillatory wake; a POD mode's sign is
arbitrary. DMD identifies temporally coherent structures, and its conjugate
pair near `0.889 Hz` provides the physical wake-shedding cross-check. The
paper's `12 modes ≈ 75%` result is contextual unless the exact paper subset
becomes known. Vorticity in the curated mode plots is derived from the two
velocity components for visualization and is not an input to either
decomposition.
