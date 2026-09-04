import pytest
import numpy as np
from PIL import Image

from fluidmodes import __version__
from fluidmodes.cli import main
from fluidmodes.pod import compute_pod
from fluidmodes.tabular import read_csv, write_csv, write_txt
from fluidmodes.manufactured import make_dmd_case, make_spod_case


def test_package_version_is_available() -> None:
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    "arguments",
    [["--help"], ["pod", "--help"], ["dmd", "--help"], ["spod", "--help"]],
)
def test_help_commands_succeed(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 0
    assert "usage: fluidmodes" in capsys.readouterr().out


def test_no_command_succeeds() -> None:
    assert main([]) == 0


def test_version_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_spod_requires_its_input_and_spectral_settings() -> None:
    with pytest.raises(SystemExit) as error:
        main(["spod"])

    assert error.value.code == 2


def test_pod_csv_writes_results_with_mean_subtraction_by_default(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "snapshots.csv"
    output_path = tmp_path / "pod_output"
    input_path.write_text("time,q1,q2\n0,1,4\n1,2,6\n2,3,8\n")

    assert main(["pod", str(input_path), "--rank", "1", "--output", str(output_path)]) == 0

    assert {path.name for path in output_path.iterdir()} == {
        "singular_values.csv",
        "energy.csv",
        "coefficients.csv",
        "modes.npy",
        "mean.npy",
        "energy.png",
    }
    assert np.load(output_path / "modes.npy").shape == (2, 1)
    assert np.allclose(np.load(output_path / "mean.npy"), [2.0, 6.0])

    expected = compute_pod(read_csv(input_path), rank=1)
    saved_singular_values = np.atleast_2d(
        np.loadtxt(output_path / "singular_values.csv", delimiter=",", skiprows=1)
    )
    saved_energy = np.atleast_2d(np.loadtxt(output_path / "energy.csv", delimiter=",", skiprows=1))
    saved_coefficients = np.loadtxt(output_path / "coefficients.csv", delimiter=",", skiprows=1)
    assert np.allclose(saved_singular_values[:, 1], expected.singular_values)
    assert np.allclose(saved_energy[:, 1], expected.energy_fractions)
    assert np.allclose(saved_coefficients[:, 0], expected.time)
    assert np.allclose(saved_coefficients[:, 1:], expected.coefficients.T)
    assert "Temporal mean subtracted: yes" in capsys.readouterr().out


def test_pod_txt_can_disable_mean_subtraction(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "snapshots.txt"
    output_path = tmp_path / "pod_output"
    input_path.write_text("time q1 q2\n0 1 4\n1 2 6\n2 3 8\n")

    assert main(
        ["pod", str(input_path), "--rank", "1", "--no-subtract-mean", "--output", str(output_path)]
    ) == 0

    assert not (output_path / "mean.npy").exists()
    assert "Temporal mean subtracted: no" in capsys.readouterr().out


def test_pod_png_requires_dt_and_writes_requested_plots(tmp_path) -> None:
    paths = []
    for index, values in enumerate(([[0, 1], [2, 3]], [[1, 3], [2, 5]], [[3, 4], [5, 8]])):
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(np.array(values, dtype=np.uint8), mode="L").save(path)
        paths.append(path)
    output_path = tmp_path / "pod_output"

    with pytest.raises(SystemExit) as error:
        main(["pod", *(str(path) for path in paths), "--output", str(output_path)])
    assert error.value.code == 2
    assert not output_path.exists()

    assert main(
        [
            "pod",
            *(str(path) for path in paths),
            "--dt",
            "0.1",
            "--rank",
            "1",
            "--plot-modes",
            "1",
            "--plot-coefficients",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 0
    assert (output_path / "mode_1.png").is_file()
    assert (output_path / "coefficient_1.png").is_file()


def test_pod_rejects_mixed_input_types(tmp_path) -> None:
    csv_path = tmp_path / "snapshots.csv"
    png_path = tmp_path / "frame.png"
    csv_path.write_text("time,q1\n0,1\n1,2\n")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(png_path)

    with pytest.raises(SystemExit) as error:
        main(["pod", str(csv_path), str(png_path), "--dt", "0.1"])

    assert error.value.code == 2


def test_dmd_csv_writes_results_and_reports_raw_data_by_default(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    case = make_dmd_case()
    input_path = tmp_path / "snapshots.csv"
    output_path = tmp_path / "dmd_output"
    write_csv(case.data, input_path)

    assert main(["dmd", str(input_path), "--output", str(output_path)]) == 0

    assert {path.name for path in output_path.iterdir()} == {
        "eigenvalues.csv",
        "eigenvalues.npy",
        "modes.npy",
        "amplitudes.npy",
        "dynamics.npy",
        "reconstruction.npy",
        "eigenvalues.png",
    }
    assert (output_path / "eigenvalues.csv").read_text().splitlines()[0] == (
        "mode,eigenvalue_real,eigenvalue_imag,growth_rate_per_input_time,"
        "frequency_cycles_per_input_time,amplitude_magnitude,amplitude_real,amplitude_imag"
    )
    table = np.atleast_2d(np.loadtxt(output_path / "eigenvalues.csv", delimiter=",", skiprows=1))
    saved_eigenvalues = table[:, 1] + 1j * table[:, 2]
    for expected in case.eigenvalues:
        assert np.min(np.abs(saved_eigenvalues - expected)) < 1e-10
    assert np.allclose(np.load(output_path / "eigenvalues.npy"), saved_eigenvalues)
    assert np.load(output_path / "modes.npy").shape == (4, 4)
    assert np.load(output_path / "amplitudes.npy").shape == (4,)
    assert np.load(output_path / "dynamics.npy").shape == (4, 80)
    assert "Temporal mean subtracted: no" in capsys.readouterr().out


def test_dmd_txt_supports_rank_and_mean_subtraction(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "snapshots.txt"
    output_path = tmp_path / "dmd_output"
    write_txt(make_dmd_case().data, input_path)

    assert main(
        ["dmd", str(input_path), "--rank", "2", "--subtract-mean", "--output", str(output_path)]
    ) == 0

    assert np.load(output_path / "modes.npy").shape == (4, 2)
    assert (output_path / "mean.npy").is_file()
    assert "Temporal mean subtracted: yes" in capsys.readouterr().out


def test_dmd_png_requires_dt_and_writes_requested_plot(tmp_path) -> None:
    paths = []
    for index, values in enumerate(([[0, 1], [2, 3]], [[1, 3], [2, 5]], [[3, 4], [5, 8]])):
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(np.array(values, dtype=np.uint8), mode="L").save(path)
        paths.append(path)
    output_path = tmp_path / "dmd_output"

    with pytest.raises(SystemExit) as error:
        main(["dmd", *(str(path) for path in paths), "--output", str(output_path)])
    assert error.value.code == 2

    assert main(
        [
            "dmd",
            *(str(path) for path in paths),
            "--dt",
            "0.1",
            "--rank",
            "1",
            "--plot-modes",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 0
    assert (output_path / "mode_1.png").is_file()


def test_dmd_rejects_nonuniform_time_and_existing_output_directory(tmp_path) -> None:
    input_path = tmp_path / "snapshots.csv"
    input_path.write_text("time,q1\n0,1\n0.1,2\n0.3,3\n")
    output_path = tmp_path / "dmd_output"

    with pytest.raises(SystemExit) as error:
        main(["dmd", str(input_path), "--output", str(output_path)])
    assert error.value.code == 2
    assert not output_path.exists()

    write_csv(make_dmd_case().data, input_path)
    output_path.mkdir()
    with pytest.raises(SystemExit) as error:
        main(["dmd", str(input_path), "--output", str(output_path)])
    assert error.value.code == 2


def test_dmd_cli_reports_zero_discrete_eigenvalue_cleanly(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "zero_eigenvalue.csv"
    output_path = tmp_path / "dmd_output"
    input_path.write_text("time,q1,q2\n0,1,0\n1,0,1\n2,0,0.5\n")

    with pytest.raises(SystemExit) as error:
        main(["dmd", str(input_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "DMD produced a zero discrete eigenvalue" in captured.err
    assert "log(0)" not in captured.err
    assert not output_path.exists()


def test_spod_csv_and_txt_write_expected_results(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    case = make_spod_case()
    csv_path = tmp_path / "snapshots.csv"
    txt_path = tmp_path / "snapshots.txt"
    csv_output = tmp_path / "spod_csv_output"
    txt_output = tmp_path / "spod_txt_output"
    write_csv(case.data, csv_path)
    write_txt(case.data, txt_path)

    assert main(
        [
            "spod",
            str(csv_path),
            "--block-size",
            "100",
            "--plot-frequency",
            "5.1",
            "--output",
            str(csv_output),
        ]
    ) == 0
    assert main(
        [
            "spod",
            str(txt_path),
            "--block-size",
            "100",
            "--overlap",
            "0",
            "--modes",
            "1",
            "--no-subtract-mean",
            "--output",
            str(txt_output),
        ]
    ) == 0

    assert {path.name for path in csv_output.iterdir()} == {
        "eigenvalues.csv",
        "eigenvalues.npy",
        "modes.npy",
        "mean.npy",
        "spectrum.png",
        "mode_1_frequency_1.png",
    }
    eigenvalue_table = np.loadtxt(csv_output / "eigenvalues.csv", delimiter=",", skiprows=1)
    assert np.allclose(eigenvalue_table[:, 0], np.arange(51, dtype=float))
    assert np.allclose(np.load(csv_output / "eigenvalues.npy"), eigenvalue_table[:, 1:])
    assert np.load(csv_output / "modes.npy").shape == (51, 16, 1)
    assert not (txt_output / "mean.npy").exists()
    assert "Overlap fraction: 0.5" in capsys.readouterr().out


def test_spod_png_requires_dt_and_runs_on_manufactured_frames(tmp_path) -> None:
    case = make_spod_case()
    frame_paths = []
    for index, snapshot in enumerate(case.data.X.T):
        path = tmp_path / f"frame_{index}.png"
        values = np.rint(128.0 + 80.0 * snapshot).reshape(4, 4).astype(np.uint8)
        Image.fromarray(values, mode="L").save(path)
        frame_paths.append(path)
    output_path = tmp_path / "spod_png_output"

    with pytest.raises(SystemExit) as error:
        main(["spod", *(str(path) for path in frame_paths), "--block-size", "100"])
    assert error.value.code == 2

    assert main(
        [
            "spod",
            *(str(path) for path in frame_paths),
            "--dt",
            "0.01",
            "--block-size",
            "100",
            "--plot-frequency",
            "12",
            "--plot-modes",
            "1",
            "--output",
            str(output_path),
        ]
    ) == 0
    assert np.load(output_path / "modes.npy").shape == (51, 16, 1)
    assert (output_path / "mode_1_frequency_1.png").is_file()


def test_spod_rejects_invalid_settings_and_existing_output(tmp_path) -> None:
    input_path = tmp_path / "snapshots.csv"
    output_path = tmp_path / "spod_output"
    write_csv(make_spod_case().data, input_path)

    for arguments in (
        ["--block-size", "3"],
        ["--block-size", "100", "--overlap", "1"],
        ["--block-size", "150", "--overlap", "0"],
        ["--block-size", "100", "--modes", "4"],
    ):
        with pytest.raises(SystemExit) as error:
            main(["spod", str(input_path), *arguments, "--output", str(output_path)])
        assert error.value.code == 2
        assert not output_path.exists()

    output_path.mkdir()
    with pytest.raises(SystemExit) as error:
        main(["spod", str(input_path), "--block-size", "100", "--output", str(output_path)])
    assert error.value.code == 2
