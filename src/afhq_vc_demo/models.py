"""Linear separator and perceptron routines for randomized-label experiments."""

from __future__ import annotations

import numpy as np


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.c_[x.astype(np.float64), np.ones(x.shape[0], dtype=np.float64)]


def vc_linear_system_solution(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Construct the minimum-norm linear solution to X_aug w ~= y."""
    x_aug = add_bias(x)
    rank = int(np.linalg.matrix_rank(x_aug, tol=1e-8))
    w, *_ = np.linalg.lstsq(x_aug, y.astype(np.float64), rcond=None)
    margins = y * (x_aug @ w)
    train_error = float(np.mean(margins <= 0))
    return rank, train_error


def run_perceptron(
    x: np.ndarray,
    y: np.ndarray,
    max_epochs: int,
    seed: int,
) -> tuple[float, int, int, bool]:
    """Run Rosenblatt's perceptron learning rule on labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    updates = 0

    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for i in rng.permutation(x.shape[0]):
            if y[i] * float(x[i] @ w + b) <= 0:
                w += y[i] * x[i]
                b += float(y[i])
                updates += 1
                mistakes += 1
        if mistakes == 0:
            return 0.0, epoch, updates, True

    scores = x @ w + b
    return float(np.mean(y * scores <= 0)), max_epochs, updates, False


def run_perceptron_with_history(
    x: np.ndarray,
    y: np.ndarray,
    max_epochs: int,
    seed: int,
    snapshot_epochs: set[int] | None = None,
) -> tuple[np.ndarray, float, list[dict[str, float | int]], dict[int, tuple[np.ndarray, float]]]:
    """Run the perceptron while keeping epoch-level error and selected weights."""
    rng = np.random.default_rng(seed)
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    updates = 0
    history: list[dict[str, float | int]] = []
    snapshots: dict[int, tuple[np.ndarray, float]] = {}
    snapshot_epochs = snapshot_epochs or set()

    if 0 in snapshot_epochs:
        snapshots[0] = (w.copy(), b)

    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for i in rng.permutation(x.shape[0]):
            if y[i] * float(x[i] @ w + b) <= 0:
                w += y[i] * x[i]
                b += float(y[i])
                updates += 1
                mistakes += 1

        scores = x @ w + b
        train_error = float(np.mean(y * scores <= 0))
        history.append(
            {
                "epoch": epoch,
                "mistakes": mistakes,
                "updates": updates,
                "train_error": train_error,
            }
        )
        if epoch in snapshot_epochs:
            snapshots[epoch] = (w.copy(), b)
        if mistakes == 0:
            snapshots.setdefault(epoch, (w.copy(), b))
            break

    return w, b, history, snapshots
