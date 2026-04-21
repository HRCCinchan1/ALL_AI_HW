"""Data loading for the handwritten digit dataset.

Each raw digit image is an ASCII grid of 28 rows by 28 columns. A space
means background (feature 0); a `+` or `#` means ink (feature 1).
Labels are integers in {0..9}.

Call `images, labels = load_digits(dtype=...)` from classifier files.
Do not parse the raw files by hand.
"""

import os
import numpy as np

DIGIT_DATADIR = "digitdata"
DIGIT_ROWS = 28
DIGIT_COLS = 28


def load_digits(dtype: str = "training", data_dir: str = DIGIT_DATADIR):
    """Return (images, labels) for the requested split.

    `dtype` is one of "training", "validation", "test". `images` is a
    float ndarray of shape (N, 28, 28); `labels` is an int ndarray of
    shape (N,).
    """
    images_path = os.path.join(data_dir, dtype + "images")
    labels_path = os.path.join(data_dir, dtype + "labels")
    images = _parse_image_file(images_path, DIGIT_ROWS, DIGIT_COLS)
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
    """Flatten (N, 28, 28) into (N, 784)."""
    return images.reshape(images.shape[0], -1)
