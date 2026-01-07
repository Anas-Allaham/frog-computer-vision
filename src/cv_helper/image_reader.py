__all__ = [
    'image_reader'
]

import cv2 as cv
from pathlib import Path


def image_reader(path, flags=cv.IMREAD_COLOR):
    if isinstance(path, Path):
        path = str(path)

    img = cv.imread(path, flags)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return img
