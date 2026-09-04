"""Reader for grayscale PNG image sequences."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from fluidmodes.snapshots import SnapshotData


def read_png_sequence(paths: Iterable[str | Path], *, dt: float) -> SnapshotData:
    """Read 8-bit grayscale PNG frames into a time-ordered snapshot matrix.

    Grayscale input gives one unambiguous scalar degree of freedom per pixel.
    Each naturally filename-sorted frame becomes one column of ``X`` after
    NumPy's default C-order flattening. ``spatial_shape=(n_y, n_x)`` retains
    the inverse reshape for plots. Pixel intensities remain unnormalized
    ``float64`` values in ``[0, 255]``; ``dt`` defines the physical or
    analysis sampling interval through ``t[k] = k * dt``.
    """
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    frame_paths = sorted((Path(path) for path in paths), key=_natural_path_key)
    if not frame_paths:
        raise ValueError("PNG sequence must contain at least one frame.")

    try:
        time_step = float(dt)
    except (TypeError, ValueError) as error:
        raise ValueError("Image time step dt must be a positive finite number.") from error
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("Image time step dt must be a positive finite number.")

    frames: list[np.ndarray] = []
    spatial_shape: tuple[int, int] | None = None
    for path in frame_paths:
        if path.suffix.lower() != ".png":
            raise ValueError(f"{path}: expected a PNG file.")
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"{path}: expected PNG image data.")
                if image.mode != "L":
                    raise ValueError(f"{path}: expected an 8-bit grayscale PNG (mode 'L').")
                frame = np.asarray(image, dtype=float)
        except UnidentifiedImageError as error:
            raise ValueError(f"{path}: expected a readable PNG image.") from error

        if spatial_shape is None:
            spatial_shape = frame.shape
        elif frame.shape != spatial_shape:
            raise ValueError(
                f"{path}: image shape {frame.shape} does not match expected shape {spatial_shape}."
            )
        frames.append(frame)

    # ``ravel`` and later mode reshaping both use NumPy's established C order.
    return SnapshotData(
        X=np.column_stack([frame.ravel() for frame in frames]),
        t=np.arange(len(frames), dtype=float) * time_step,
        spatial_shape=spatial_shape,
    )


def _natural_path_key(path: Path) -> tuple[tuple[tuple[int, int | str], ...], str]:
    parts = re.split(r"(\d+)", path.name.casefold())
    key = tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
    return key, str(path)
