"""Write tables, summaries, and figures for the experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .config import BIAS_DIM, IMAGE_SIZE, INPUT_DIM
from .experiment import ExperimentResult


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
        f.write("# AFHQ Perceptron VC-Dimension Demo Results\n\n")
        f.write(f"Data source: {source_name}\n\n")
        f.write(f"Image shape: {IMAGE_SIZE} x {IMAGE_SIZE} grayscale\n\n")
        f.write(f"Input dimension: {INPUT_DIM}; bias-augmented dimension: {BIAS_DIM}\n\n")
        f.write("| N | rank(X with bias) | VC construction train error | Perceptron train error | Perceptron converged? |\n")
        f.write("|---:|---:|---:|---:|:---|\n")
        for r in results:
            f.write(
                f"| {r.sample_size} | {r.rank_with_bias} | "
                f"{r.vc_solution_train_error:.4f} | {r.perceptron_train_error:.4f} | "
                f"{'yes' if r.perceptron_converged else 'no'} |\n"
            )
        f.write("\nLabels were randomized. Zero training error means memorization of an arbitrary split.\n")
        f.write(
            "\nThe linear-system row reports an exact-value construction, not a perceptron training trace. "
            "The perceptron row reports finite training with the configured epoch limit; non-convergence "
            "within that limit is evidence of practical difficulty, not a formal proof of non-separability.\n"
        )
    return md_path


def write_training_error_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    vc_errors = [r.vc_solution_train_error for r in results]
    p_errors = [r.perceptron_train_error for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, vc_errors, marker="o", label="linear system construction")
    plt.plot(ns, p_errors, marker="o", label="perceptron learning rule")
    plt.axvline(BIAS_DIM, color="black", linestyle="--", linewidth=1, label="d + 1 = 4097")
    plt.xlabel("Number of randomly labeled examples")
    plt.ylabel("Training error")
    plt.ylim(-0.02, 1.0)
    plt.legend()
    plt.tight_layout()

    figure_path = output_dir / "training_error.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def write_rank_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    ranks = [r.rank_with_bias for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ns, ranks, marker="o", color="#2f6f8f", label="observed rank")
    plt.plot(ns, ns, linestyle=":", color="#6b7280", label="rank = N")
    plt.axhline(BIAS_DIM, color="black", linestyle="--", linewidth=1, label="maximum rank = 4097")
    plt.xlabel("Number of examples")
    plt.ylabel("Rank of bias-augmented data matrix")
    plt.ylim(0, BIAS_DIM + 350)
    plt.legend()
    plt.tight_layout()

    figure_path = output_dir / "rank_vs_sample_size.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def write_perceptron_updates_plot(results: list[ExperimentResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ns = [r.sample_size for r in results]
    updates = [r.perceptron_updates for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(ns, updates, width=220, color="#7a4eab")
    plt.axvline(BIAS_DIM, color="black", linestyle="--", linewidth=1, label="d + 1 = 4097")
    plt.xlabel("Number of randomly labeled examples")
    plt.ylabel("Perceptron updates over 50 epochs")
    plt.legend()
    plt.tight_layout()

    figure_path = output_dir / "perceptron_updates.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def write_outputs(results: list[ExperimentResult], output_root: Path, source_name: str) -> None:
    write_results_csv(results, output_root / "tables")
    write_summary(results, output_root, source_name)
    figure_dir = output_root / "figures"
    write_training_error_plot(results, figure_dir)
    write_rank_plot(results, figure_dir)
    write_perceptron_updates_plot(results, figure_dir)
