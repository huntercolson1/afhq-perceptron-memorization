from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afhq_vc_demo.config import INPUT_DIM  # noqa: E402
from afhq_vc_demo.data import make_synthetic_vectors  # noqa: E402
from afhq_vc_demo.experiment import run_experiments  # noqa: E402
from afhq_vc_demo.models import vc_linear_system_solution  # noqa: E402


def test_synthetic_vectors_have_expected_shape() -> None:
    x, y = make_synthetic_vectors(8, seed=1)

    assert x.shape == (8, INPUT_DIM)
    assert y.shape == (8,)
    assert set(np.unique(y)) <= {-1, 1}


def test_vc_construction_separates_small_random_labels() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 20)).astype(np.float32)
    y = rng.choice(np.array([-1, 1], dtype=np.int8), size=10)

    rank, error = vc_linear_system_solution(x, y)

    assert rank == 10
    assert error == 0.0


def test_experiment_returns_one_row_per_sample_size() -> None:
    x, _ = make_synthetic_vectors(20, seed=3)

    results = run_experiments(x, sample_sizes=[5, 10], max_epochs=2, seed=3)

    assert [r.sample_size for r in results] == [5, 10]
