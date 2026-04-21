"""Data loading for the face detection dataset.

Each raw face image is an ASCII grid of 70 rows by 60 columns. A space
means background (feature 0); a `+` or `#` means ink (feature 1).
Labels are binary: 1 means face, 0 means not a face.

Call `images, labels = load_faces(dtype=...)` from classifier files.
Note the training split file is `facedatatrain`, not `facedatatraining`.
"""

import os
import numpy as np

FACE_DATADIR = "facedata"
FACE_ROWS = 70
FACE_COLS = 60


def load_faces(dtype: str = "train", data_dir: str = FACE_DATADIR):
    """Return (images, labels) for the requested split.

    `dtype` is one of "train", "validation", "test". `images` is a float
    ndarray of shape (N, 70, 60); `labels` is an int ndarray of shape
    (N,).
    """
    images_path = os.path.join(data_dir, "facedata" + dtype)
    labels_path = os.path.join(data_dir, "facedata" + dtype + "labels")
    images = _parse_image_file(images_path, FACE_ROWS, FACE_COLS)
    labels = _parse_label_file(labels_path)
    assert len(images) == len(labels), (
        f"Mismatch: {len(images)} images vs {len(labels)} labels"
    )
    return images, labels


def _parse_image_file(path: str, rows: int, cols: int) -> np.ndarray:
    data = []
    with open(path, "r") as fp:
        lines = fp.readlines()
    for i in range(0, len(lines), rows):
        block = lines[i : i + rows]
        grid = np.zeros((rows, cols), dtype=float)
        for r, line in enumerate(block):
            line = line.rstrip("\n")
            for c, ch in enumerate(line[:cols]):
                if ch == "+" or ch == "#":
                    grid[r][c] = 1.0
        data.append(grid)
    return np.array(data)


def _parse_label_file(path: str) -> np.ndarray:
    labels = []
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if line:
                labels.append(int(line))
    return np.array(labels, dtype=int)


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten (N, 70, 60) into (N, 4200)."""
    return images.reshape(images.shape[0], -1)
