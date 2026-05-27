#!/usr/bin/env python3
"""Run the AFHQ randomized-label perceptron experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afhq_vc_demo.data import load_afhq_cat_dog, make_synthetic_vectors  # noqa: E402
from afhq_vc_demo.experiment import run_experiments  # noqa: E402
from afhq_vc_demo.reporting import write_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 1000, 2000, 4096, 4097, 5000])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_n = max(args.sample_sizes)

    if args.synthetic:
        x, _ = make_synthetic_vectors(max_n, args.seed)
        source_name = "synthetic Gaussian vectors"
    else:
        x, _, _ = load_afhq_cat_dog(args.data_root, max_n, args.seed)
        source_name = str(args.data_root)

    results = run_experiments(x, args.sample_sizes, args.max_epochs, args.seed)
    write_outputs(results, args.output_dir, source_name)

    for result in results:
        print(
            f"N={result.sample_size:5d} "
            f"rank={result.rank_with_bias:5d} "
            f"vc_error={result.vc_solution_train_error:.4f} "
            f"perceptron_error={result.perceptron_train_error:.4f} "
            f"converged={result.perceptron_converged}"
        )


if __name__ == "__main__":
    main()
