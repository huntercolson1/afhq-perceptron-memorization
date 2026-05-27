#!/usr/bin/env python3
"""Create tutorial figures that make perceptron memorization visible."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afhq_vc_demo.config import IMAGE_SIZE  # noqa: E402
from afhq_vc_demo.data import load_afhq_cat_dog  # noqa: E402
from afhq_vc_demo.models import run_perceptron_with_history  # noqa: E402


def unstandardized_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(arr, dtype=np.float32) / 255.0


def normalize_weight_image(w: np.ndarray) -> np.ndarray:
    image = w.reshape(IMAGE_SIZE, IMAGE_SIZE)
    vmax = np.percentile(np.abs(image), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(image, -vmax, vmax)


def signed_feature_image(x_row: np.ndarray, label: int) -> np.ndarray:
    return label * x_row.reshape(IMAGE_SIZE, IMAGE_SIZE)


def make_training_journey(data_root: Path, output_dir: Path, seed: int = 7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_size = 500
    max_experiment_size = 5000
    x_all, true_labels_all, paths_all = load_afhq_cat_dog(data_root, max_experiment_size, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(x_all.shape[0])
    random_labels_all = rng.choice(np.array([-1, 1], dtype=np.int8), size=x_all.shape[0])
    subset = order[:sample_size]
    x = x_all[subset]
    true_labels = true_labels_all[subset]
    paths = [paths_all[i] for i in subset]
    random_labels = random_labels_all[subset]

    snapshot_epochs = {0, 1, 2, 5, 10, 25, 48}
    w, b, history, snapshots = run_perceptron_with_history(
        x,
        random_labels,
        max_epochs=60,
        seed=seed + sample_size,
        snapshot_epochs=snapshot_epochs,
    )

    history_path = output_dir.parent / "tables" / "perceptron_history_n500.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        f.write("epoch,mistakes,updates,train_error\n")
        for row in history:
            f.write(f"{row['epoch']},{row['mistakes']},{row['updates']},{row['train_error']}\n")

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.25)

    ax = fig.add_subplot(grid[0, :])
    ax.plot([r["epoch"] for r in history], [r["train_error"] for r in history], marker="o", linewidth=1.6)
    ax.set_title("Perceptron learning rule on 500 randomly labeled AFHQ images")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training error")
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.25)

    ordered_snapshots = sorted(snapshots.items())[:8]
    for idx, (epoch, (weights, _bias)) in enumerate(ordered_snapshots):
        ax_w = fig.add_subplot(grid[1 + idx // 4, idx % 4])
        im = ax_w.imshow(normalize_weight_image(weights), cmap="viridis")
        ax_w.set_title(f"weights after epoch {epoch}", fontsize=10)
        ax_w.axis("off")
    colorbar = fig.colorbar(im, ax=fig.axes[1:], fraction=0.02, pad=0.02)
    colorbar.set_label("learned weight value")
    fig.savefig(output_dir / "perceptron_training_journey_n500.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    scores = x @ w + b
    predictions = np.where(scores >= 0, 1, -1)
    chosen = np.linspace(0, sample_size - 1, 8, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax_img, idx in zip(axes.flat, chosen):
        ax_img.imshow(unstandardized_image(paths[idx]), cmap="gray")
        true_name = "dog" if true_labels[idx] == 1 else "cat"
        assigned = "+1" if random_labels[idx] == 1 else "-1"
        pred = "+1" if predictions[idx] == 1 else "-1"
        ax_img.set_title(f"true: {true_name}\nrandom label: {assigned}, pred: {pred}", fontsize=9)
        ax_img.axis("off")
    fig.suptitle("The model fits random labels, not the visual cat/dog concept", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "random_label_examples_n500.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    selected = np.linspace(0, sample_size - 1, 6, dtype=int)
    contribution_stack = np.array([signed_feature_image(x[i], int(random_labels[i])) for i in selected])
    contribution_scale = np.percentile(np.abs(contribution_stack), 99)
    if contribution_scale <= 0:
        contribution_scale = 1.0

    fig = plt.figure(figsize=(13, 5.8))
    grid = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1.15], hspace=0.25, wspace=0.14)
    for col, idx in enumerate(selected):
        ax_raw = fig.add_subplot(grid[0, col])
        ax_raw.imshow(unstandardized_image(paths[idx]), cmap="gray")
        true_name = "dog" if true_labels[idx] == 1 else "cat"
        assigned = "+1" if random_labels[idx] == 1 else "-1"
        ax_raw.set_title(f"true: {true_name}\nrandom label {assigned}", fontsize=9)
        ax_raw.axis("off")

        ax_contrib = fig.add_subplot(grid[1, col])
        ax_contrib.imshow(
            signed_feature_image(x[idx], int(random_labels[idx])),
            cmap="viridis",
            vmin=-contribution_scale,
            vmax=contribution_scale,
        )
        ax_contrib.set_title("signed update\n$y_i x_i$", fontsize=9)
        ax_contrib.axis("off")

    ax_final = fig.add_subplot(grid[:, 6])
    im = ax_final.imshow(normalize_weight_image(w), cmap="viridis")
    ax_final.set_title("final weights\nas a 64 x 64 image", fontsize=10)
    ax_final.axis("off")
    colorbar = fig.colorbar(im, ax=ax_final, fraction=0.05, pad=0.04)
    colorbar.set_label("weight value")
    fig.suptitle(
        "A perceptron weight vector is an accumulation of signed image-like updates",
        fontsize=13,
    )
    fig.savefig(output_dir / "weight_update_story_n500.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    make_training_journey(
        data_root=project_root / "data" / "raw" / "kaggle_animal_faces",
        output_dir=project_root / "outputs" / "figures",
    )


if __name__ == "__main__":
    main()
