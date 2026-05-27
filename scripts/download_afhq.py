#!/usr/bin/env python3
"""Download AFHQ data for the experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afhq_vc_demo.data import download_dataset  # noqa: E402
from afhq_vc_demo.config import KAGGLE_SLUG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("auto", "kagglehub", "kaggle", "official"),
        default="auto",
        help="auto tries KaggleHub, Kaggle CLI, then the official StarGAN v2 mirror.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_dataset(args.data_dir, args.source)
    print(f"Dataset handle: {KAGGLE_SLUG}")
    print(f"Dataset path: {path.resolve()}")


if __name__ == "__main__":
    main()
