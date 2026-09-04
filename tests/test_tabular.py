import numpy as np
import pytest

from fluidmodes.manufactured import make_pod_case
from fluidmodes.tabular import read_csv, read_txt, write_csv, write_txt


def test_read_csv_converts_rows_to_snapshot_columns(tmp_path) -> None:
    path = tmp_path / "snapshots.csv"
    path.write_text("time,q1,q2\n0.0,1.0,10.0\n0.5,2.0,20.0\n")

    data = read_csv(path)

    assert np.array_equal(data.t, [0.0, 0.5])
    assert np.array_equal(data.X, [[1.0, 2.0], [10.0, 20.0]])
    assert data.labels == ("q1", "q2")
    assert data.time_label == "time"


def test_read_txt_converts_rows_to_snapshot_columns(tmp_path) -> None:
    path = tmp_path / "snapshots.txt"
    path.write_text("time q1 q2\n0.0 1.0 10.0\n0.5 2.0 20.0\n")

    data = read_txt(path)

    assert np.array_equal(data.t, [0.0, 0.5])
    assert np.array_equal(data.X, [[1.0, 2.0], [10.0, 20.0]])
    assert data.labels == ("q1", "q2")


@pytest.mark.parametrize(
    ("reader", "filename", "contents"),
    [
        (read_csv, "invalid.csv", ""),
        (read_csv, "invalid.csv", "time\n"),
        (read_csv, "invalid.csv", "time,q1\n"),
        (read_csv, "invalid.csv", "time,q1\n0.0,not-a-number\n"),
        (read_csv, "invalid.csv", "time,q1\n0.0,1.0,extra\n"),
        (read_csv, "invalid.csv", "time,q1\n0.1,1\n0.0,2\n"),
        (read_txt, "invalid.txt", "time q1\n0.0 not-a-number\n"),
    ],
)
def test_readers_reject_malformed_input(tmp_path, reader, filename: str, contents: str) -> None:
    path = tmp_path / filename
    path.write_text(contents)

    with pytest.raises(ValueError):
        reader(path)


@pytest.mark.parametrize(
    ("writer", "reader", "suffix"),
    [(write_csv, read_csv, ".csv"), (write_txt, read_txt, ".txt")],
)
def test_manufactured_data_round_trips_through_tabular_files(tmp_path, writer, reader, suffix: str) -> None:
    original = make_pod_case().data
    path = tmp_path / f"pod{suffix}"

    writer(original, path)
    restored = reader(path)

    assert np.allclose(restored.X, original.X)
    assert np.allclose(restored.t, original.t)
    assert restored.labels == original.labels
    assert restored.time_label == original.time_label
