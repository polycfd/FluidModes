import numpy as np
import pytest
from PIL import Image

from fluidmodes.images import read_png_sequence
from fluidmodes.manufactured import make_pod_case


def _write_grayscale_png(path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def test_png_reader_uses_natural_frame_order_and_preserves_shape(tmp_path) -> None:
    frame_10 = tmp_path / "frame_10.png"
    frame_2 = tmp_path / "frame_2.png"
    _write_grayscale_png(frame_10, np.array([[10, 11], [12, 13]]))
    _write_grayscale_png(frame_2, np.array([[2, 3], [4, 5]]))

    data = read_png_sequence([frame_10, frame_2], dt=0.25)

    assert data.X.shape == (4, 2)
    assert data.spatial_shape == (2, 2)
    assert np.array_equal(data.t, [0.0, 0.25])
    assert np.array_equal(data.X[:, 0].reshape(data.spatial_shape), [[2, 3], [4, 5]])
    assert np.array_equal(data.X[:, 1].reshape(data.spatial_shape), [[10, 11], [12, 13]])


def test_png_reader_rejects_invalid_sequences(tmp_path) -> None:
    valid = tmp_path / "valid.png"
    mismatched = tmp_path / "mismatched.png"
    rgb = tmp_path / "rgb.png"
    text = tmp_path / "not-an-image.txt"
    _write_grayscale_png(valid, np.zeros((2, 2)))
    _write_grayscale_png(mismatched, np.zeros((3, 2)))
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(rgb)
    text.write_text("not a PNG")

    with pytest.raises(ValueError, match="at least one frame"):
        read_png_sequence([], dt=0.1)
    with pytest.raises(ValueError, match="does not match"):
        read_png_sequence([valid, mismatched], dt=0.1)
    with pytest.raises(ValueError, match="grayscale"):
        read_png_sequence([rgb], dt=0.1)
    with pytest.raises(ValueError, match="expected a PNG"):
        read_png_sequence([text], dt=0.1)


def test_manufactured_pod_data_round_trips_as_png_frames(tmp_path) -> None:
    original = make_pod_case().data
    assert original.spatial_shape == (2, 2)
    frames = original.X.T.reshape(-1, *original.spatial_shape)
    lower, upper = frames.min(), frames.max()
    quantized = np.rint((frames - lower) / (upper - lower) * 255.0).astype(np.uint8)
    paths = []
    for index, frame in enumerate(quantized):
        path = tmp_path / f"pod_{index}.png"
        _write_grayscale_png(path, frame)
        paths.append(path)

    restored = read_png_sequence(reversed(paths), dt=0.1)

    assert restored.X.shape == original.X.shape
    assert restored.spatial_shape == original.spatial_shape
    assert np.allclose(restored.X.T.reshape(quantized.shape), quantized, atol=0.5)
