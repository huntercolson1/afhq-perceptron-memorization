"""Shared constants for the AFHQ perceptron experiment."""

from __future__ import annotations

IMAGE_SIZE = 64
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE
BIAS_DIM = INPUT_DIM + 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
KAGGLE_SLUG = "andrewmvd/animal-faces"
OFFICIAL_AFHQ_URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
