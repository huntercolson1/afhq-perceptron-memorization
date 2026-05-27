"""Write tables, summaries, and figures for the experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .config import BIAS_DIM, IMAGE_SIZE, INPUT_DIM
from .experiment import ExperimentResult

INK = "#18212f"
MUTED = "#657184"
BLUE = "#2674b8"
GREEN = "#277a55"
PLUM = "#7a4eab"
GOLD = "#c58b2d"
PAPER = "#fbfaf7"
GRID = "#d8d2c7"


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": GRID,
            "savefig.facecolor": PAPER,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def write_results_csv(results: list[ExperimentResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExperimentResult.__dataclass_fields__))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)
    return csv_path


def write_summary(results: list[ExperimentResult], output_dir: Path, source_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# AFHQ Perceptron Memorization Results\n\n")
        f.write(f"Data source: {source_name}\n\n")
        f.write(f"Image shape: {IMAGE_SIZE} x {IMAGE_SIZE} grayscale\n\n")
        f.write(f"Input dimension: {INPUT_DIM}; bias-augmented dimension: {BIAS_DIM}\n\n")
        f.write("| N | rank(X with bias) | Exact separator proof error | Perceptron train error | Perceptron converged? |\n")
        f.write("|---:|---:|---:|---:|:---|\n")
        for r in results:
            f.write(
                f"| {r.sample_size} | {r.rank_with_bias} | "
                f"{r.vc_solution_train_error:.4f} | {r.perceptron_train_error:.4f} | "
                f"{'yes' if r.perceptron_converged else 'no'} |\n"
            )
        f.write("\nLabels were randomized. Zero training error means memorization of an arbitrary split.\n")
        f.write(
            "\nFor `N <= 4097`, `rank(X with bias) = N`, so the bias-augmented data matrix "
            "has full row rank. That proves an exact linear separator exists for any labels "
            "on those examples, including the randomized labels used here.\n"
        )
        f.write(
            "\nThe exact-separator proof row reports a direct solve of `X_aug @ w = y`, not a perceptron training trace. "
            "The perceptron row reports finite training with the configured epoch limit; non-convergence "
            "within that limit is evidence of practical difficulty, not a formal proof of non-separability. "
            "The perceptron convergence theorem is the formal link from separability to eventual convergence "
            "of the perceptron learning rule.\n"
        )
    return md_path


def write_training_error_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    set_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    vc_errors = [r.vc_solution_train_error for r in results]
    p_errors = [r.perceptron_train_error for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, vc_errors, marker="o", linewidth=2.2, color=GREEN, label="exact separator proof")
    ax.plot(ns, p_errors, marker="o", linewidth=2.2, color=BLUE, label="perceptron after 50 epochs")
    ax.axvline(BIAS_DIM, color=GOLD, linestyle="--", linewidth=1.6, label="d + 1 = 4097")
    ax.set_xlabel("Number of randomly labeled examples")
    ax.set_ylabel("Training error")
    ax.set_ylim(-0.02, 0.34)
    ax.legend(loc="upper left")
    finish_axes(ax)
    fig.tight_layout()

    figure_path = output_dir / "training_error.png"
    fig.savefig(figure_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_rank_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    set_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    ranks = [r.rank_with_bias for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, ranks, marker="o", linewidth=2.2, color=BLUE, label="observed rank")
    ax.plot(ns, ns, linestyle=":", linewidth=1.6, color=MUTED, label="rank = N")
    ax.axhline(BIAS_DIM, color=GOLD, linestyle="--", linewidth=1.6, label="maximum rank = 4097")
    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Rank of bias-augmented data matrix")
    ax.set_ylim(0, BIAS_DIM + 350)
    ax.legend(loc="lower right")
    finish_axes(ax)
    fig.tight_layout()

    figure_path = output_dir / "rank_vs_sample_size.png"
    fig.savefig(figure_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_perceptron_updates_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    set_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    updates = [r.perceptron_updates for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ns, updates, width=220, color=PLUM, alpha=0.95)
    ax.axvline(BIAS_DIM, color=GOLD, linestyle="--", linewidth=1.6, label="d + 1 = 4097")
    ax.set_xlabel("Number of randomly labeled examples")
    ax.set_ylabel("Perceptron updates over 50 epochs")
    ax.legend(loc="upper left")
    finish_axes(ax)
    fig.tight_layout()

    figure_path = output_dir / "perceptron_updates.png"
    fig.savefig(figure_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_long_run_plot(csv_path: Path, output_dir: Path) -> Path | None:
    set_plot_style()
    if not csv_path.exists():
        return None

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "sample_size": int(row["sample_size"]),
                    "perceptron_epochs": int(row["perceptron_epochs"]),
                    "perceptron_updates": int(row["perceptron_updates"]),
                }
            )
    if not rows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [row["sample_size"] for row in rows]
    epochs = [row["perceptron_epochs"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ns, epochs, width=220, color=GREEN, alpha=0.95)
    ax.axvline(BIAS_DIM, color=GOLD, linestyle="--", linewidth=1.6, label="d + 1 = 4097")
    ax.set_xlabel("Number of randomly labeled AFHQ examples")
    ax.set_ylabel("Epochs until zero training error")
    ax.legend(loc="upper left")
    finish_axes(ax)
    fig.tight_layout()

    figure_path = output_dir / "perceptron_long_run_epochs.png"
    fig.savefig(figure_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def write_outputs(results: list[ExperimentResult], output_root: Path, source_name: str) -> None:
    write_results_csv(results, output_root / "tables")
    write_summary(results, output_root, source_name)
    figure_dir = output_root / "figures"
    write_training_error_plot(results, figure_dir)
    write_rank_plot(results, figure_dir)
    write_perceptron_updates_plot(results, figure_dir)
    write_long_run_plot(output_root / "tables" / "perceptron_long_run.csv", figure_dir)
