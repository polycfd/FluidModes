"""Run the independently auditable experimental-cylinder validation stages."""

from __future__ import annotations

import subprocess
import sys

from prepare_data import EXAMPLE_DIRECTORY


def main() -> None:
    """Prepare -> NumPy references -> CLI -> compare -> curated figures."""
    for name in (
        "prepare_data.py",
        "reference_pod.py",
        "reference_dmd.py",
        "run_fluidmodes.py",
        "compare_results.py",
        "make_plots.py",
    ):
        subprocess.run([sys.executable, str(EXAMPLE_DIRECTORY / name)], check=True, cwd=EXAMPLE_DIRECTORY)


if __name__ == "__main__":
    main()
