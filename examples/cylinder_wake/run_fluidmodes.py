"""Run the released FluidModes CLI on the prepared cylinder-wake inputs."""

from __future__ import annotations

import shutil
import subprocess

from prepare_data import DMD_CSV_PATH, EXAMPLE_DIRECTORY, POD_CSV_PATH
from reference_dmd import DMD_RANK


RESULTS_DIRECTORY = EXAMPLE_DIRECTORY / "results"
POD_RESULTS = RESULTS_DIRECTORY / "pod"
DMD_RESULTS = RESULTS_DIRECTORY / "dmd"


def main() -> None:
    """Invoke POD with default mean removal and rank-21 DMD on raw snapshots."""
    if shutil.which("fluidmodes") is None:
        raise FileNotFoundError("The fluidmodes CLI executable is not on PATH.")
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for directory in (POD_RESULTS, DMD_RESULTS):
        if directory.exists():
            shutil.rmtree(directory)
    subprocess.run(["fluidmodes", "pod", str(POD_CSV_PATH.resolve()), "--output", str(POD_RESULTS.resolve())], check=True, cwd=EXAMPLE_DIRECTORY)
    subprocess.run(["fluidmodes", "dmd", str(DMD_CSV_PATH.resolve()), "--rank", str(DMD_RANK), "--output", str(DMD_RESULTS.resolve())], check=True, cwd=EXAMPLE_DIRECTORY)


if __name__ == "__main__":
    main()
