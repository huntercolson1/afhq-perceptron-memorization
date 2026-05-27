#!/usr/bin/env python3
"""Create tutorial figures that make perceptron memorization visible."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afhq_vc_demo.config import IMAGE_SIZE  # noqa: E402
from afhq_vc_demo.data import load_afhq_cat_dog  # noqa: E402
from afhq_vc_demo.models import run_perceptron_with_history  # noqa: E402

INK = "#18212f"
MUTED = "#657184"
BLUE = "#2674b8"
GREEN = "#277a55"
GOLD = "#c58b2d"
PLUM = "#7a4eab"
PAPER = "#fbfaf7"
PANEL = "#f2efe8"
GRID = "#d8d2c7"


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.facecolor": PAPER,
        }
    )


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


def add_round_box(ax, xy, width, height, text, fc=PANEL, ec=INK, fontsize=11, lw=1.0, radius=0.018):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        transform=ax.transAxes,
    )
    return box


def add_arrow(ax, start, end, color=BLUE, lw=1.8, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def make_perceptron_anatomy(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")

    pixel_positions = [(0.08, 0.80), (0.08, 0.62), (0.08, 0.44), (0.08, 0.26)]
    labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_{4096}$"]
    weights = [r"$w_1$", r"$w_2$", r"$w_3$", r"$w_{4096}$"]
    for (x, y), label, weight in zip(pixel_positions, labels, weights):
        ax.add_patch(Circle((x, y), 0.045, facecolor="#ffffff", edgecolor=INK, lw=1.4, transform=ax.transAxes))
        ax.text(x, y, label, ha="center", va="center", fontsize=13, color=INK, transform=ax.transAxes)
        ax.text(0.25, y + 0.03, weight, ha="center", va="center", fontsize=12, color=MUTED, transform=ax.transAxes)
        add_arrow(ax, (x + 0.05, y), (0.44, 0.54), color=BLUE, lw=1.4, rad=0.04 if y > 0.55 else -0.04)

    ax.text(0.08, 0.08, "4096 pixel values", ha="center", color=MUTED, fontsize=10, transform=ax.transAxes)

    ax.add_patch(Circle((0.50, 0.54), 0.12, facecolor="#ffffff", edgecolor=INK, lw=1.6, transform=ax.transAxes))
    ax.text(0.50, 0.56, r"$\sum w_jx_j + b$", ha="center", va="center", fontsize=15, color=INK, transform=ax.transAxes)
    ax.text(0.50, 0.48, "score", ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    add_round_box(ax, (0.67, 0.45), 0.17, 0.18, "sign check\n$s \\geq 0$?", fc="#ffffff", ec=INK, fontsize=12)
    add_arrow(ax, (0.62, 0.54), (0.67, 0.54), color=BLUE)
    add_arrow(ax, (0.84, 0.59), (0.94, 0.75), color=GREEN, rad=0.18)
    add_arrow(ax, (0.84, 0.49), (0.94, 0.33), color=PLUM, rad=-0.18)
    ax.text(0.95, 0.77, r"predict $+1$", ha="left", va="center", fontsize=12, color=GREEN, transform=ax.transAxes)
    ax.text(0.95, 0.31, r"predict $-1$", ha="left", va="center", fontsize=12, color=PLUM, transform=ax.transAxes)

    fig.savefig(output_dir / "perceptron_anatomy.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_linear_system_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.25))
    ax.axis("off")

    matrix_x, matrix_y = 0.08, 0.31
    cell_w, cell_h = 0.035, 0.08
    rows, cols = 5, 9
    for r in range(rows):
        for c in range(cols):
            fc = "#ffffff" if c < cols - 1 else "#fff1d4"
            ax.add_patch(
                Rectangle(
                    (matrix_x + c * cell_w, matrix_y + (rows - 1 - r) * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=fc,
                    edgecolor=GRID,
                    lw=0.8,
                    transform=ax.transAxes,
                )
            )
    ax.text(matrix_x + cols * cell_w / 2, 0.81, r"$X_{\mathrm{aug}}$", ha="center", fontsize=18, color=INK, transform=ax.transAxes)
    ax.text(matrix_x + cols * cell_w / 2, 0.25, r"$N$ rows by $4097$ columns", ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)
    ax.text(matrix_x + (cols - 0.5) * cell_w, 0.21, "bias\ncolumn", ha="center", fontsize=9, color=GOLD, transform=ax.transAxes)

    ax.text(0.45, 0.52, r"$\times$", ha="center", va="center", fontsize=24, color=INK, transform=ax.transAxes)

    vx, vy = 0.51, 0.31
    for r in range(rows):
        ax.add_patch(Rectangle((vx, vy + (rows - 1 - r) * cell_h), 0.055, cell_h, facecolor="#ffffff", edgecolor=GRID, lw=0.8, transform=ax.transAxes))
    ax.text(vx + 0.027, 0.81, r"$\tilde{w}$", ha="center", fontsize=18, color=INK, transform=ax.transAxes)
    ax.text(vx + 0.027, 0.25, "weights\n+ bias", ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    ax.text(0.63, 0.52, r"$=$", ha="center", va="center", fontsize=24, color=INK, transform=ax.transAxes)

    yx, yy = 0.70, 0.31
    yvals = ["+1", "-1", "+1", "...", "-1"]
    for r, val in enumerate(yvals):
        ax.add_patch(Rectangle((yx, yy + (rows - 1 - r) * cell_h), 0.065, cell_h, facecolor="#ffffff", edgecolor=GRID, lw=0.8, transform=ax.transAxes))
        ax.text(yx + 0.032, yy + (rows - 1 - r) * cell_h + cell_h / 2, val, ha="center", va="center", fontsize=10, color=INK, transform=ax.transAxes)
    ax.text(yx + 0.032, 0.81, r"$y$", ha="center", fontsize=18, color=INK, transform=ax.transAxes)
    ax.text(yx + 0.032, 0.25, "random\nlabels", ha="center", fontsize=10, color=MUTED, transform=ax.transAxes)

    fig.savefig(output_dir / "linear_system_construction.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_capacity_boundary_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sample_sizes = np.array([500, 1000, 2000, 4096, 4097, 5000])
    ranks = np.array([500, 1000, 2000, 4096, 4097, 4097])
    ax.plot(sample_sizes, ranks, color=BLUE, linewidth=2.4, marker="o", markersize=7)
    ax.axhline(4097, color=GOLD, linestyle="--", linewidth=1.8)
    ax.axvline(4097, color=GOLD, linestyle="--", linewidth=1.8)
    ax.fill_between([0, 4097], [0, 4097], [4097, 4097], color=GREEN, alpha=0.08)
    ax.fill_between([4097, 5200], [0, 0], [4097, 4097], color=PLUM, alpha=0.06)
    ax.annotate(
        "VC boundary:\n4096 pixels + 1 bias",
        xy=(4097, 4097),
        xytext=(2800, 4620),
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.6},
        fontsize=11,
        color=INK,
    )
    ax.text(900, 3800, "full row rank\nobserved here", color=GREEN, fontsize=12)
    ax.text(4300, 1450, "rank cannot exceed\n4097 columns", color=PLUM, fontsize=12)
    ax.set_xlim(0, 5200)
    ax.set_ylim(0, 5000)
    ax.set_xlabel("number of training examples, N")
    ax.set_ylabel(r"rank of $X_{\mathrm{aug}}$")
    ax.grid(True, alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_boundary.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_title_plate(data_root: Path, output_dir: Path, seed: int = 7) -> None:
    x_all, true_labels_all, paths_all = load_afhq_cat_dog(data_root, 5000, seed)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(len(paths_all)), size=6, replace=False)
    random_labels = rng.choice(np.array([-1, 1], dtype=np.int8), size=6)

    fig = plt.figure(figsize=(12, 4.7))
    grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.18, 1.4, 0.18, 1.65], wspace=0.08, hspace=0.1)

    for slot, idx in enumerate(chosen[:4]):
        ax = fig.add_subplot(grid[slot // 2, slot % 2])
        ax.imshow(unstandardized_image(paths_all[idx]), cmap="gray")
        true_name = "dog" if true_labels_all[idx] == 1 else "cat"
        assigned = "+1" if random_labels[slot] == 1 else "-1"
        ax.set_title(f"{true_name}, random {assigned}", fontsize=8.5, pad=3)
        ax.axis("off")

    ax_arrow_1 = fig.add_subplot(grid[:, 2])
    ax_arrow_1.axis("off")
    ax_arrow_1.annotate(
        "",
        xy=(0.9, 0.5),
        xytext=(0.1, 0.5),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": BLUE},
        xycoords=ax_arrow_1.transAxes,
    )

    ax_matrix = fig.add_subplot(grid[:, 3])
    ax_matrix.axis("off")
    rows, cols = 7, 13
    for r in range(rows):
        for c in range(cols):
            fc = "#ffffff" if c < cols - 1 else "#fff1d4"
            ax_matrix.add_patch(
                Rectangle(
                    (0.08 + c * 0.062, 0.2 + (rows - r - 1) * 0.075),
                    0.062,
                    0.075,
                    facecolor=fc,
                    edgecolor=GRID,
                    lw=0.75,
                    transform=ax_matrix.transAxes,
                )
            )
    ax_matrix.text(0.49, 0.84, r"$X_{\mathrm{aug}}$", ha="center", fontsize=18, color=INK, transform=ax_matrix.transAxes)
    ax_matrix.text(0.49, 0.13, "4096 pixels plus one bias column", ha="center", fontsize=8.5, color=MUTED, transform=ax_matrix.transAxes)

    ax_arrow_2 = fig.add_subplot(grid[:, 4])
    ax_arrow_2.axis("off")
    ax_arrow_2.annotate(
        "",
        xy=(0.9, 0.5),
        xytext=(0.1, 0.5),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": BLUE},
        xycoords=ax_arrow_2.transAxes,
    )

    ax_sep = fig.add_subplot(grid[:, 5])
    rng_plot = np.random.default_rng(seed + 100)
    positive = rng_plot.normal(loc=[0.6, 1.0], scale=[0.23, 0.22], size=(24, 2))
    negative = rng_plot.normal(loc=[1.05, 0.45], scale=[0.24, 0.22], size=(24, 2))
    ax_sep.scatter(positive[:, 0], positive[:, 1], s=28, color=GREEN, edgecolor=INK, linewidth=0.4)
    ax_sep.scatter(negative[:, 0], negative[:, 1], s=28, color=PLUM, edgecolor=INK, linewidth=0.4)
    xs = np.linspace(0.15, 1.5, 100)
    ax_sep.plot(xs, -0.75 * xs + 1.42, color=INK, linewidth=1.6)
    ax_sep.fill_between(xs, -0.75 * xs + 1.42, 1.7, color=GREEN, alpha=0.08)
    ax_sep.fill_between(xs, 0.0, -0.75 * xs + 1.42, color=PLUM, alpha=0.08)
    ax_sep.set_xlim(0.12, 1.55)
    ax_sep.set_ylim(0.02, 1.55)
    ax_sep.set_xticks([])
    ax_sep.set_yticks([])
    ax_sep.set_title("separation geometry", fontsize=8.5, pad=5)
    for spine in ["top", "right"]:
        ax_sep.spines[spine].set_visible(False)

    fig.savefig(output_dir / "title_plate.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_concept_figures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_figure_style()
    make_perceptron_anatomy(output_dir)
    make_linear_system_figure(output_dir)
    make_capacity_boundary_figure(output_dir)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    labels = np.array([-1, 1, 1])
    colors = np.where(labels == 1, GREEN, PLUM)
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=140, edgecolor="black", linewidth=1.2, zorder=3)
    for name, (x1, x2), label in zip(["A", "B", "C"], points, labels):
        ax.annotate(f"{name}: {label:+d}", (x1, x2), xytext=(8, 8), textcoords="offset points", fontsize=11)
    xs = np.linspace(-0.2, 1.2, 200)
    ys = 0.5 - xs
    ax.plot(xs, ys, color=INK, linewidth=2.0, label=r"$2x_1 + 2x_2 - 1 = 0$")
    ax.fill_between(xs, ys, 1.25, color=GREEN, alpha=0.08)
    ax.fill_between(xs, -0.25, ys, color=PLUM, alpha=0.08)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_dir / "toy_separator_2d.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    xor_points = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    xor_labels = np.array([-1, -1, 1, 1])
    colors = np.where(xor_labels == 1, GREEN, PLUM)
    markers = np.where(xor_labels == 1, "o", "s")
    for point, label, color, marker in zip(xor_points, xor_labels, colors, markers):
        ax.scatter(point[0], point[1], c=color, marker=marker, s=150, edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(f"{label:+d}", point, xytext=(8, 8), textcoords="offset points", fontsize=11)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_dir / "xor_nonseparable.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.axis("off")
    boxes = [
        ("AFHQ cat/dog\nimages", 0.08),
        ("64 x 64\ngrayscale", 0.25),
        ("4096-pixel\nvectors", 0.42),
        ("add bias:\n4097 columns", 0.59),
        ("random labels\n-1 or +1", 0.76),
        ("separator proof\nand perceptron", 0.92),
    ]
    for idx, (text, x) in enumerate(boxes):
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": INK},
            transform=ax.transAxes,
        )
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[idx + 1][1] - 0.07, 0.55),
                xytext=(x + 0.07, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": BLUE},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    fig.savefig(output_dir / "experiment_pipeline.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_training_journey(data_root: Path, output_dir: Path, seed: int = 7) -> None:
    set_figure_style()
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
    ax.plot([r["epoch"] for r in history], [r["train_error"] for r in history], marker="o", color=BLUE, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training error")
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

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
    fig.savefig(output_dir / "weight_update_story_n500.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    make_concept_figures(project_root / "outputs" / "figures")
    make_title_plate(
        data_root=project_root / "data" / "raw" / "kaggle_animal_faces",
        output_dir=project_root / "outputs" / "figures",
    )
    make_training_journey(
        data_root=project_root / "data" / "raw" / "kaggle_animal_faces",
        output_dir=project_root / "outputs" / "figures",
    )


if __name__ == "__main__":
    main()
