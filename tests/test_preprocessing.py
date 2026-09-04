import numpy as np

from fluidmodes.preprocessing import subtract_temporal_mean
from fluidmodes.snapshots import SnapshotData


def test_temporal_mean_subtraction_preserves_metadata_and_input() -> None:
    original = SnapshotData(
        X=np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]]),
        t=np.array([0.0, 0.2, 0.4]),
        labels=("q1", "q2"),
        time_label="time",
        spatial_shape=(1, 2),
    )
    original_values = original.X.copy()

    result = subtract_temporal_mean(original)

    assert np.allclose(result.mean, [3.0, 14.0 / 3.0])
    assert np.allclose(np.mean(result.data.X, axis=1), 0.0)
    assert np.array_equal(result.data.t, original.t)
    assert result.data.labels == original.labels
    assert result.data.time_label == original.time_label
    assert result.data.spatial_shape == original.spatial_shape
    assert np.array_equal(original.X, original_values)
