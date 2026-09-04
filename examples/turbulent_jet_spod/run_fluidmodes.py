"""Run the released FluidModes SPOD CLI on the prepared jet-pressure CSV."""

from __future__ import annotations

import shutil
import subprocess

from prepare_data import CSV_PATH, EXAMPLE_DIRECTORY


RESULTS_DIRECTORY = EXAMPLE_DIRECTORY / "results"
SPOD_RESULTS = RESULTS_DIRECTORY / "spod"


def main() -> None:
    """Invoke the released CLI with the fixed unweighted batch-SPOD definition."""
    if shutil.which("fluidmodes") is None:
        raise FileNotFoundError("The fluidmodes CLI executable is not on PATH.")
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if SPOD_RESULTS.exists():
        shutil.rmtree(SPOD_RESULTS)
    subprocess.run(
        [
            "fluidmodes",
            "spod",
            str(CSV_PATH.resolve()),
            "--block-size", "256",
            "--overlap", "0.5",
            "--modes", "2",
            "--output", str(SPOD_RESULTS.resolve()),
        ],
        check=True,
        cwd=EXAMPLE_DIRECTORY,
    )


if __name__ == "__main__":
    main()
