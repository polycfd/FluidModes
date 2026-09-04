"""Convenience wrapper for the independently auditable cylinder-wake stages."""

from compare_results import main as compare_results
from make_plots import main as make_plots
from prepare_data import prepare_inputs
from reference_dmd import main as reference_dmd
from reference_pod import main as reference_pod
from run_fluidmodes import main as run_fluidmodes


def main() -> None:
    """Prepare data → independent references → CLI → comparison → figures."""
    prepare_inputs()
    reference_pod()
    reference_dmd()
    run_fluidmodes()
    compare_results()
    make_plots()


if __name__ == "__main__":
    main()
