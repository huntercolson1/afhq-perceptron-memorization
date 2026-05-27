"""Experiment orchestration for the AFHQ VC-dimension demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import run_perceptron, vc_linear_system_solution


@dataclass
class ExperimentResult:
    sample_size: int
    rank_with_bias: int
    vc_solution_train_error: float
    perceptron_train_error: float
    perceptron_epochs: int
    perceptron_updates: int
    perceptron_converged: bool


def run_experiments(
    x: np.ndarray,
    sample_sizes: list[int],
    max_epochs: int,
    seed: int,
) -> list[ExperimentResult]:
    rng = np.random.default_rng(seed)
    if x.shape[0] < max(sample_sizes):
        raise ValueError(f"Need {max(sample_sizes)} rows, but received {x.shape[0]}.")

    order = rng.permutation(x.shape[0])
    random_labels = rng.choice(np.array([-1, 1], dtype=np.int8), size=x.shape[0])
    results: list[ExperimentResult] = []

    for n in sample_sizes:
        subset = order[:n]
        x_n = x[subset]
        y_n = random_labels[subset]

        rank, vc_error = vc_linear_system_solution(x_n, y_n)
        p_error, epochs, updates, converged = run_perceptron(
            x_n,
            y_n,
            max_epochs=max_epochs,
            seed=seed + n,
        )
        results.append(
            ExperimentResult(
                sample_size=n,
                rank_with_bias=rank,
                vc_solution_train_error=vc_error,
                perceptron_train_error=p_error,
                perceptron_epochs=epochs,
                perceptron_updates=updates,
                perceptron_converged=converged,
            )
        )
    return results
