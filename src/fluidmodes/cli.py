"""Command-line interface for FluidModes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fluidmodes import __version__
from fluidmodes.dmd import DMDResult, compute_dmd
from fluidmodes.images import read_png_sequence
from fluidmodes.pod import PODResult, compute_pod
from fluidmodes.spod import SPODResult, compute_spod
from fluidmodes.tabular import read_csv, read_txt


def build_parser() -> argparse.ArgumentParser:
    """Create the FluidModes argument parser."""
    parser = argparse.ArgumentParser(
        prog="fluidmodes",
        description="FluidModes is a modal-analysis post-processing tool for fluid-mechanics data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    pod_parser = subparsers.add_parser("pod", help="Run POD analysis.")
    pod_parser.add_argument("inputs", nargs="+", type=Path, help="One CSV/TXT file or PNG frames.")
    pod_parser.add_argument("--rank", type=int, help="Number of POD modes to retain.")
    pod_parser.add_argument("--dt", type=float, help="Frame time step, required for PNG input.")
    mean_group = pod_parser.add_mutually_exclusive_group()
    mean_group.add_argument(
        "--subtract-mean",
        dest="subtract_mean",
        action="store_true",
        help="Subtract the temporal mean before POD (default).",
    )
    mean_group.add_argument(
        "--no-subtract-mean",
        dest="subtract_mean",
        action="store_false",
        help="Analyze the input snapshots without temporal mean subtraction.",
    )
    pod_parser.set_defaults(subtract_mean=True, handler=_run_pod)
    pod_parser.add_argument(
        "--output", type=Path, default=Path("pod_output"), help="New output directory (default: pod_output)."
    )
    pod_parser.add_argument(
        "--plot-modes", nargs="+", type=int, metavar="MODE", help="One-based mode numbers to plot."
    )
    pod_parser.add_argument(
        "--plot-coefficients",
        nargs="+",
        type=int,
        metavar="MODE",
        help="One-based coefficient numbers to plot.",
    )

    dmd_parser = subparsers.add_parser("dmd", help="Run standard exact DMD analysis.")
    dmd_parser.add_argument("inputs", nargs="+", type=Path, help="One CSV/TXT file or PNG frames.")
    dmd_parser.add_argument("--rank", type=int, help="SVD truncation rank for exact DMD.")
    dmd_parser.add_argument("--dt", type=float, help="Frame time step, required for PNG input.")
    dmd_parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract the temporal mean before DMD (disabled by default).",
    )
    dmd_parser.add_argument(
        "--output", type=Path, default=Path("dmd_output"), help="New output directory (default: dmd_output)."
    )
    dmd_parser.add_argument(
        "--plot-modes", nargs="+", type=int, metavar="MODE", help="One-based mode numbers to plot."
    )
    dmd_parser.set_defaults(handler=_run_dmd)

    spod_parser = subparsers.add_parser("spod", help="Run standard batch SPOD analysis.")
    spod_parser.add_argument("inputs", nargs="+", type=Path, help="One CSV/TXT file or PNG frames.")
    spod_parser.add_argument(
        "--block-size", required=True, type=int, help="Snapshots per SPOD block."
    )
    spod_parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fractional block overlap from 0 to less than 1 (default: 0.5).",
    )
    spod_parser.add_argument(
        "--modes", type=int, default=1, help="Leading SPOD spatial modes saved at each frequency (default: 1)."
    )
    spod_parser.add_argument("--dt", type=float, help="Frame time step, required for PNG input.")
    spod_parser.add_argument(
        "--no-subtract-mean",
        dest="subtract_mean",
        action="store_false",
        help="Analyze snapshots without long-time temporal mean subtraction.",
    )
    spod_parser.add_argument(
        "--output", type=Path, default=Path("spod_output"), help="New output directory (default: spod_output)."
    )
    spod_parser.add_argument(
        "--plot-frequency",
        action="append",
        type=float,
        metavar="FREQUENCY",
        help="Plot modes at the nearest resolved frequency bin; may be repeated.",
    )
    spod_parser.add_argument(
        "--plot-modes", nargs="+", type=int, metavar="MODE", help="One-based saved mode numbers to plot."
    )
    spod_parser.set_defaults(subtract_mean=True, handler=_run_spod)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FluidModes command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is not None:
        return args.handler(args, parser)
    return 0


def _run_pod(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = _read_input(args, parser, "POD")
    try:
        result = compute_pod(data, rank=args.rank, subtract_mean=args.subtract_mean)
    except ValueError as error:
        parser.error(str(error))

    _validate_plot_modes(args.plot_modes, result.rank, parser)
    _validate_plot_modes(args.plot_coefficients, result.rank, parser)
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    _write_pod_output(result, args.output, args.plot_modes, args.plot_coefficients)

    print(f"DOFs: {result.modes.shape[0]}")
    print(f"Snapshots: {result.time.size}")
    print(f"Retained rank: {result.rank}")
    print(f"Temporal mean subtracted: {'yes' if result.mean is not None else 'no'}")
    print(f"Retained energy fraction: {np.sum(result.energy_fractions):.6g}")
    print(f"Output directory: {args.output}")
    return 0


def _run_dmd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = _read_input(args, parser, "DMD")
    try:
        result = compute_dmd(data, rank=args.rank, subtract_mean=args.subtract_mean)
    except ValueError as error:
        parser.error(str(error))

    _validate_plot_modes(args.plot_modes, result.rank, parser)
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    _write_dmd_output(result, args.output, args.plot_modes)

    time_step = result.time[1] - result.time[0]
    print(f"DOFs: {result.modes.shape[0]}")
    print(f"Snapshots: {result.time.size}")
    print(f"Retained DMD modes: {result.rank}")
    print(f"Time step: {time_step:.17g}")
    print(f"Temporal mean subtracted: {'yes' if result.mean is not None else 'no'}")
    print(f"Output directory: {args.output}")
    return 0


def _run_spod(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = _read_input(args, parser, "SPOD")
    if args.plot_modes is not None and args.plot_frequency is None:
        parser.error("--plot-modes requires at least one --plot-frequency.")
    try:
        result = compute_spod(
            data,
            block_size=args.block_size,
            overlap=args.overlap,
            modes=args.modes,
            subtract_mean=args.subtract_mean,
        )
    except ValueError as error:
        parser.error(str(error))

    _validate_plot_modes(args.plot_modes, result.modes.shape[2], parser)
    if args.plot_frequency is not None and not np.isfinite(args.plot_frequency).all():
        parser.error("requested SPOD plot frequencies must be finite.")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    _write_spod_output(result, args.output, args.plot_frequency, args.plot_modes)

    print(f"DOFs: {result.modes.shape[1]}")
    print(f"Snapshots: {data.t.size}")
    print(f"Time step: {result.time_step:.17g}")
    print(f"Block size: {result.block_size}")
    print(f"Overlap fraction: {result.overlap:.17g}")
    print(f"Blocks: {result.n_blocks}")
    print(f"Frequency resolution: {1.0 / (result.block_size * result.time_step):.17g}")
    print(f"Frequencies: {result.frequencies.size}")
    print(f"Spatial modes saved per frequency: {result.modes.shape[2]}")
    print(f"Long-time mean subtracted: {'yes' if result.mean is not None else 'no'}")
    print(f"Output directory: {args.output}")
    return 0


def _read_input(args: argparse.Namespace, parser: argparse.ArgumentParser, analysis_name: str):
    suffixes = {path.suffix.lower() for path in args.inputs}
    try:
        if suffixes == {".csv"}:
            if len(args.inputs) != 1:
                parser.error(f"{analysis_name} CSV input requires exactly one file.")
            if args.dt is not None:
                parser.error("--dt is only valid for PNG input.")
            return read_csv(args.inputs[0])
        if suffixes == {".txt"}:
            if len(args.inputs) != 1:
                parser.error(f"{analysis_name} TXT input requires exactly one file.")
            if args.dt is not None:
                parser.error("--dt is only valid for PNG input.")
            return read_txt(args.inputs[0])
        if suffixes == {".png"}:
            if args.dt is None:
                parser.error(f"PNG {analysis_name} input requires --dt.")
            return read_png_sequence(args.inputs, dt=args.dt)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    parser.error(f"{analysis_name} input must be one CSV/TXT file or only PNG frames.")


def _validate_plot_modes(
    mode_numbers: list[int] | None, rank: int, parser: argparse.ArgumentParser
) -> None:
    if mode_numbers is not None and any(number < 1 or number > rank for number in mode_numbers):
        parser.error(f"requested mode numbers must be between 1 and {rank}.")


def _write_pod_output(
    result: PODResult,
    output_directory: Path,
    plot_modes: list[int] | None,
    plot_coefficients: list[int] | None,
) -> None:
    from fluidmodes.pod_plots import (
        plot_energy_spectrum,
        plot_spatial_mode,
        plot_temporal_coefficient,
    )

    mode_numbers = np.arange(1, result.rank + 1)
    np.savetxt(
        output_directory / "singular_values.csv",
        np.column_stack((mode_numbers, result.singular_values)),
        delimiter=",",
        header="mode,singular_value",
        comments="",
        fmt=["%d", "%.17g"],
    )
    np.savetxt(
        output_directory / "energy.csv",
        np.column_stack((mode_numbers, result.energy_fractions)),
        delimiter=",",
        header="mode,energy_fraction",
        comments="",
        fmt=["%d", "%.17g"],
    )
    coefficient_table = np.column_stack((result.time, result.coefficients.T))
    np.savetxt(
        output_directory / "coefficients.csv",
        coefficient_table,
        delimiter=",",
        header=",".join(("time", *(f"a{number}" for number in mode_numbers))),
        comments="",
        fmt="%.17g",
    )
    np.save(output_directory / "modes.npy", result.modes)
    if result.mean is not None:
        np.save(output_directory / "mean.npy", result.mean)

    plot_energy_spectrum(result, output_directory / "energy.png")
    for mode_number in plot_modes or ():
        plot_spatial_mode(result, mode_number, output_directory / f"mode_{mode_number}.png")
    for mode_number in plot_coefficients or ():
        plot_temporal_coefficient(
            result, mode_number, output_directory / f"coefficient_{mode_number}.png"
        )


def _write_dmd_output(
    result: DMDResult, output_directory: Path, plot_modes: list[int] | None
) -> None:
    from fluidmodes.dmd_plots import plot_eigenvalues, plot_spatial_mode

    mode_numbers = np.arange(1, result.rank + 1)
    eigenvalue_table = np.column_stack(
        (
            mode_numbers,
            result.eigenvalues.real,
            result.eigenvalues.imag,
            result.growth_rates,
            result.frequencies,
            np.abs(result.amplitudes),
            result.amplitudes.real,
            result.amplitudes.imag,
        )
    )
    np.savetxt(
        output_directory / "eigenvalues.csv",
        eigenvalue_table,
        delimiter=",",
        header=(
            "mode,eigenvalue_real,eigenvalue_imag,growth_rate_per_input_time,"
            "frequency_cycles_per_input_time,"
            "amplitude_magnitude,amplitude_real,amplitude_imag"
        ),
        comments="",
        fmt=["%d", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g"],
    )
    np.save(output_directory / "modes.npy", result.modes)
    np.save(output_directory / "eigenvalues.npy", result.eigenvalues)
    np.save(output_directory / "amplitudes.npy", result.amplitudes)
    np.save(output_directory / "dynamics.npy", result.dynamics)
    np.save(output_directory / "reconstruction.npy", result.reconstruction)
    if result.mean is not None:
        np.save(output_directory / "mean.npy", result.mean)

    plot_eigenvalues(result, output_directory / "eigenvalues.png")
    for mode_number in plot_modes or ():
        plot_spatial_mode(result, mode_number, output_directory / f"mode_{mode_number}.png")


def _write_spod_output(
    result: SPODResult,
    output_directory: Path,
    plot_frequencies: list[float] | None,
    plot_modes: list[int] | None,
) -> None:
    from fluidmodes.spod_plots import plot_eigenvalue_spectrum, plot_spatial_mode

    np.savetxt(
        output_directory / "eigenvalues.csv",
        np.column_stack((result.frequencies, result.eigenvalues)),
        delimiter=",",
        header=",".join(
            (
                "frequency_cycles_per_input_time",
                *(f"mode_{number}" for number in range(1, result.eigenvalues.shape[1] + 1)),
            )
        ),
        comments="",
        fmt="%.17g",
    )
    np.save(output_directory / "eigenvalues.npy", result.eigenvalues)
    np.save(output_directory / "modes.npy", result.modes)
    if result.mean is not None:
        np.save(output_directory / "mean.npy", result.mean)

    plot_eigenvalue_spectrum(result, output_directory / "spectrum.png")
    if plot_frequencies is not None:
        for frequency_index, frequency in enumerate(plot_frequencies, start=1):
            for mode_number in plot_modes or (1,):
                selected_frequency = plot_spatial_mode(
                    result,
                    frequency,
                    mode_number,
                    output_directory / f"mode_{mode_number}_frequency_{frequency_index}.png",
                )
                print(
                    f"Requested plot frequency {frequency:.17g}; selected nearest bin {selected_frequency:.17g}."
                )


if __name__ == "__main__":
    main()
