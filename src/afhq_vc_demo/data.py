"""Data download and loading utilities for AFHQ cat/dog face images."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import IMAGE_EXTENSIONS, IMAGE_SIZE, INPUT_DIM, KAGGLE_SLUG, OFFICIAL_AFHQ_URL


def download_with_progress(url: str, destination: Path) -> None:
    """Download a large file with a compact terminal progress indicator."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def hook(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        pct = downloaded / total_size * 100
        sys.stderr.write(f"\rDownloading {destination.name}: {pct:5.1f}%")
        sys.stderr.flush()

    urlretrieve(url, destination, hook)
    sys.stderr.write("\n")


def unzip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def download_from_kagglehub(data_dir: Path) -> Path:
    """Download the exact Kaggle dataset slug using KaggleHub if it is available."""
    import kagglehub

    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    target = data_dir / "raw" / "kaggle_animal_faces"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(downloaded_path, target)
    return target


def download_from_kaggle_cli(data_dir: Path) -> Path:
    """Download the exact Kaggle dataset slug using the Kaggle CLI."""
    if shutil.which("kaggle") is None:
        raise RuntimeError("The kaggle CLI is not installed.")
    target = data_dir / "raw" / "kaggle_animal_faces"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(target), "--unzip"],
        check=True,
    )
    return target


def download_from_official_mirror(data_dir: Path) -> Path:
    """Download AFHQ from the official StarGAN v2 Dropbox mirror."""
    raw_dir = data_dir / "raw"
    zip_path = raw_dir / "afhq.zip"
    download_with_progress(OFFICIAL_AFHQ_URL, zip_path)
    unzip(zip_path, raw_dir)
    zip_path.unlink()
    return raw_dir / "afhq"


def download_dataset(data_dir: Path, source: str) -> Path:
    """Download AFHQ, preferring the exact Kaggle slug when requested."""
    if source == "kagglehub":
        return download_from_kagglehub(data_dir)
    if source == "kaggle":
        return download_from_kaggle_cli(data_dir)
    if source == "official":
        return download_from_official_mirror(data_dir)
    if source == "auto":
        errors: list[str] = []
        for candidate in (download_from_kagglehub, download_from_kaggle_cli, download_from_official_mirror):
            try:
                return candidate(data_dir)
            except Exception as exc:  # noqa: BLE001 - keep trying known download routes.
                errors.append(f"{candidate.__name__}: {exc}")
        raise RuntimeError("All download methods failed:\n" + "\n".join(errors))
    raise ValueError(f"Unknown source: {source}")


def find_class_dirs(data_root: Path) -> tuple[Path, Path]:
    """Find cat and dog directories across the common AFHQ/Kaggle layouts."""
    candidates = [
        (data_root / "train" / "cat", data_root / "train" / "dog"),
        (data_root / "val" / "cat", data_root / "val" / "dog"),
        (data_root / "cat", data_root / "dog"),
        (data_root / "afhq" / "train" / "cat", data_root / "afhq" / "train" / "dog"),
        (data_root / "kaggle_animal_faces" / "afhq" / "train" / "cat", data_root / "kaggle_animal_faces" / "afhq" / "train" / "dog"),
        (data_root / "Animal-Faces-HQ" / "data" / "cat", data_root / "Animal-Faces-HQ" / "data" / "dog"),
    ]
    for cat_dir, dog_dir in candidates:
        if cat_dir.exists() and dog_dir.exists():
            return cat_dir, dog_dir

    for cat_dir in data_root.rglob("cat"):
        dog_dir = cat_dir.parent / "dog"
        if dog_dir.exists():
            return cat_dir, dog_dir

    raise FileNotFoundError(
        f"Could not find AFHQ cat/dog folders under {data_root}. "
        "Expected a layout like data/raw/afhq/train/cat and data/raw/afhq/train/dog."
    )


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(arr, dtype=np.float32).reshape(-1) / 255.0


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x -= x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    x /= scale
    return x.astype(np.float32)


def load_afhq_cat_dog(data_root: Path, max_images: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load a balanced-ish shuffled cat/dog sample as 64 x 64 grayscale vectors."""
    cat_dir, dog_dir = find_class_dirs(data_root)
    items = [(p, -1) for p in list_images(cat_dir)] + [(p, 1) for p in list_images(dog_dir)]
    if len(items) < max_images:
        raise ValueError(f"Need {max_images} cat/dog images, but only found {len(items)}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    selected = items[:max_images]

    x = np.empty((len(selected), INPUT_DIM), dtype=np.float32)
    y = np.empty(len(selected), dtype=np.int8)
    paths: list[Path] = []
    for i, (path, label) in enumerate(tqdm(selected, desc="Loading AFHQ images")):
        x[i] = load_image(path)
        y[i] = label
        paths.append(path)
    return standardize(x), y, paths


def make_synthetic_vectors(max_images: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(max_images, INPUT_DIM)).astype(np.float32)
    y = rng.choice(np.array([-1, 1], dtype=np.int8), size=max_images)
    return standardize(x), y
