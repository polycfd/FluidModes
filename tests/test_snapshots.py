import numpy as np
import pytest

from fluidmodes.snapshots import SnapshotData


def test_snapshot_data_accepts_valid_matrix() -> None:
    data = SnapshotData(
        X=[[1, 2, 3], [4, 5, 6]],
        t=[0.0, 0.1, 0.2],
        labels=("a", "b"),
        spatial_shape=(1, 2),
    )

    assert data.X.shape == (2, 3)
    assert np.array_equal(data.t, [0.0, 0.1, 0.2])
    assert data.labels == ("a", "b")
    assert data.spatial_shape == (1, 2)


@pytest.mark.parametrize(
    ("X", "t", "message"),
    [
        ([1, 2, 3], [0.0, 0.1, 0.2], "two-dimensional"),
        ([[1, 2], [3, 4]], [0.0], "Number of time coordinates"),
        ([[1, 2], [3, 4]], [0.0, 0.0], "strictly increasing"),
        ([[1, np.nan], [3, 4]], [0.0, 0.1], "finite"),
    ],
)
def test_snapshot_data_rejects_invalid_data(X: object, t: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SnapshotData(X=X, t=t)


def test_snapshot_data_rejects_incompatible_spatial_shape() -> None:
    with pytest.raises(ValueError, match="Product of spatial shape"):
        SnapshotData(X=[[1, 2], [3, 4]], t=[0.0, 0.1], spatial_shape=(3, 1))
