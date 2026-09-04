"""Run released FluidModes CLI POD and DMD on the prepared PIV CSV."""

from __future__ import annotations

import shutil
import subprocess
import time

from prepare_data import CSV_PATH, EXAMPLE_DIRECTORY
from reference_dmd import DMD_RANK


RESULTS_DIRECTORY = EXAMPLE_DIRECTORY / "results"
POD_RESULTS = RESULTS_DIRECTORY / "pod"
DMD_RESULTS = RESULTS_DIRECTORY / "dmd"
RUNTIME_PATH = RESULTS_DIRECTORY / "runtimes.txt"


def main() -> None:
    """Invoke POD (default mean removal) and raw-state rank-12 DMD through the CLI."""
    if shutil.which("fluidmodes") is None:
        raise FileNotFoundError("The fluidmodes CLI executable is not on PATH.")
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing prepared input {CSV_PATH}; run prepare_data.py first.")
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for directory in (POD_RESULTS, DMD_RESULTS):
        if directory.exists():
            shutil.rmtree(directory)

    pod_started = time.perf_counter()
    subprocess.run(
        [
            "fluidmodes",
            "pod",
            str(CSV_PATH),
            "--output",
            str(POD_RESULTS),
        ],
        check=True,
    )
    dmd_started = time.perf_counter()
    subprocess.run(
        [
            "fluidmodes",
            "dmd",
            str(CSV_PATH),
            "--rank",
            str(DMD_RANK),
            "--output",
            str(DMD_RESULTS),
        ],
        check=True,
    )
    RUNTIME_PATH.write_text(
        f"pod_seconds={dmd_started - pod_started:.6f}\n"
        f"dmd_seconds={time.perf_counter() - dmd_started:.6f}\n"
    )
    print(f"Wrote FluidModes POD output: {POD_RESULTS}")
    print(f"Wrote FluidModes DMD output: {DMD_RESULTS}")


if __name__ == "__main__":
    main()
