"""Convenience wrapper for the independently auditable jet-SPOD stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_results import main as compare_results
from make_plots import main as make_plots
from prepare_data import DEFAULT_MAT_PATH, prepare_csv
from reference_spod import write_reference
from run_fluidmodes import main as run_fluidmodes


def main() -> None:
    """Prepare data → independent SPOD → CLI → comparison → figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", nargs="?", type=Path, default=DEFAULT_MAT_PATH)
    args = parser.parse_args()
    prepare_csv(args.mat_path)
    write_reference(args.mat_path)
    run_fluidmodes()
    compare_results()
    make_plots(args.mat_path)


if __name__ == "__main__":
    main()
