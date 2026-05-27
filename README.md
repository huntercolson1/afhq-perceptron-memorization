# High-Dimensional Perceptron Memorization on AFHQ

This is a notebook-first, reproducible experiment about a subtle point in
machine learning: high-dimensional linear models can sometimes perfectly fit
arbitrary labels without learning the concept we care about.

The concrete question:

> Can a 64 x 64 grayscale perceptron memorize random labels on cat/dog face
> images just because the input dimension is large?

The experiment uses Kaggle's `andrewmvd/animal-faces` dataset, also known as
Animal Faces-HQ (AFHQ). It contains cat, dog, and wildlife face images at
512 x 512 resolution. This experiment resizes cat/dog images to 64 x 64
grayscale, so each image is a 4096-dimensional vector.

## Folder structure

```text
afhq_vc_perceptron_experiment/
  README.md
  requirements.txt
  data/
    raw/                  # downloaded AFHQ data
    processed/            # reserved for cached arrays, if needed
  notebooks/
    01_afhq_vc_perceptron_demo.ipynb
  scripts/
    download_afhq.py
    run_experiment.py
  src/afhq_vc_demo/
    config.py
    data.py
    experiment.py
    models.py
    reporting.py
  outputs/
    figures/
    tables/
  docs/
    math_background.md
    interpreting_results.md
    RUN_LOG.md
```

## What the demo tests

The experiment deliberately randomizes labels after loading real cat/dog images.
If a model fits those random labels, it is memorizing an arbitrary split rather
than learning cats versus dogs.

It runs two checks:

1. **VC construction check:** for `N <= 4097`, solve a linear system to find a
   separating hyperplane for arbitrary random labels, when the image vectors are
   full row rank after adding a bias column.
2. **Perceptron learning check:** run Rosenblatt's perceptron learning rule on
   the same randomized labels and report whether it reaches zero training error.

For `N = 5000`, the VC guarantee no longer applies. The demo reports what
actually happens on the resized AFHQ data.

For a more beginner-friendly explanation, see:

- `docs/math_background.md`
- `docs/interpreting_results.md`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Note: this project pins `kagglehub==0.3.13` because the latest `1.0.1` release
hit a local `kagglesdk` import mismatch during verification on this Mac.

## Download AFHQ

Try the exact Kaggle dataset first:

```bash
python scripts/download_afhq.py --source kagglehub
```

If KaggleHub asks for credentials, try the Kaggle CLI path after configuring
`~/.kaggle/kaggle.json`:

```bash
python scripts/download_afhq.py --source kaggle
```

If Kaggle is blocked locally, use the official AFHQ mirror from the StarGAN v2
authors:

```bash
python scripts/download_afhq.py --source official
```

The Kaggle dataset slug is:

```text
andrewmvd/animal-faces
```

## Run the experiment

```bash
python scripts/run_experiment.py --data-root data/raw --sample-sizes 500 1000 2000 4096 4097 5000
```

Outputs are written to `outputs/`:

- `outputs/tables/results.csv`
- `outputs/summary.md`
- `outputs/figures/training_error.png`
- `outputs/figures/rank_vs_sample_size.png`
- `outputs/figures/perceptron_updates.png`

## Verified result on this Mac

The exact Kaggle dataset downloaded successfully on 2026-05-27, and the notebook
executed top-to-bottom. The durable result table is:

| N | rank(X with bias) | VC construction train error | Perceptron train error | Perceptron converged? |
|---:|---:|---:|---:|:---|
| 500 | 500 | 0.0000 | 0.0000 | yes |
| 1000 | 1000 | 0.0000 | 0.0340 | no |
| 2000 | 2000 | 0.0000 | 0.1585 | no |
| 4096 | 4096 | 0.0000 | 0.2925 | no |
| 4097 | 4097 | 0.0000 | 0.2563 | no |
| 5000 | 4097 | 0.0262 | 0.2252 | no |

See `docs/RUN_LOG.md` for commands and verification evidence.

## Quick synthetic smoke test

If the AFHQ download is not available yet, this verifies the code path using
random 4096-dimensional vectors:

```bash
python scripts/run_experiment.py --synthetic --sample-sizes 500 1000 2000 4096 4097 5000
```

## Interpretation guardrail

The important comparison is not simply whether `N = 5000` is larger than 4096.
The VC-dimension result gives a guarantee up to the bias-augmented dimension
under appropriate rank/general-position assumptions. Some random splits above
that boundary can still be separable in high dimensions. The experiment reports
what happens on this actual resized AFHQ sample instead of assuming the answer.
