# High-Dimensional Perceptron Memorization

This repository is a runnable experiment and textbook-style tutorial about a
small but important machine-learning lesson: a model can fit its training set
perfectly without learning the concept we care about.

The experiment uses real cat and dog images from Kaggle's
`andrewmvd/animal-faces` dataset. Each image is converted to 64 x 64 grayscale,
so every image becomes a vector with 4096 pixel values. A perceptron then gets
4096 pixel weights plus one bias term, giving 4097 adjustable parameters.

The core question is:

> Can a single linear perceptron fit random labels on real cat and dog images
> up to the 4097-parameter VC-dimension boundary?

In this sampled AFHQ experiment, yes. The exact linear-system construction fits
random labels with zero training error through `N = 4097`, and longer
perceptron-learning-rule runs converge to zero training error for every tested
sample size up to that boundary.

## Start here

The polished learning module is available in two formats:

- [HTML textbook chapter](docs/perceptron_memorization_textbook.html)
- [PDF textbook chapter](docs/perceptron_memorization_textbook.pdf)

The rendered chapter is the public teaching artifact. The repository keeps the
experiment code, notebook, figures, tables, and final HTML/PDF outputs together
so the result can be read first and reproduced second.

## Inspiration

This project was inspired by the perceptron and VC-dimension discussion in
[*The Welch Labs Illustrated Guide to AI*](https://www.welchlabs.com/resources/ai-book-ezrzm-msrmc).
The repository is otherwise written as a standalone experiment and learning
resource.

## What the experiment shows

The labels are randomized after loading real cat and dog images. That detail is
the point. If the perceptron fits those labels, the fit cannot be explained by
the visual concept of cat or dog. The target no longer contains that concept.

The experiment runs two checks:

1. **Exact separator construction.** Add a bias column to the image matrix,
   check the matrix rank, and solve `X_aug @ w = y`. When the augmented matrix
   has full row rank, this proves that a linear separator exists for the
   randomized labels.
2. **Perceptron learning rule.** Train the perceptron by mistake-driven updates
   and record how many epochs it takes to reach zero training error.

The exact construction answers "does a separator exist?" The perceptron run
answers "did this training algorithm find one, and how long did it take?"

## Verified result

The experiment was run on the Kaggle AFHQ animal-faces data on 2026-05-27.

| N | rank(X with bias) | Exact separator proof error | Perceptron train error after 50 epochs | Perceptron converged? |
|---:|---:|---:|---:|:---|
| 500 | 500 | 0.0000 | 0.0000 | yes |
| 1000 | 1000 | 0.0000 | 0.0340 | no |
| 2000 | 2000 | 0.0000 | 0.1585 | no |
| 4096 | 4096 | 0.0000 | 0.2925 | no |
| 4097 | 4097 | 0.0000 | 0.2563 | no |
| 5000 | 4097 | 0.0262 | 0.2252 | no |

The 50-epoch sweep is intentionally finite. To test the "given enough
iterations" part directly, longer perceptron runs were also performed:

| N | Max epochs allowed | Epochs to zero error | Updates | Converged? |
|---:|---:|---:|---:|:---|
| 500 | 50 | 48 | 2638 | yes |
| 1000 | 1000 | 80 | 9046 | yes |
| 2000 | 2000 | 214 | 44162 | yes |
| 4096 | 5000 | 945 | 365288 | yes |
| 4097 | 5000 | 1228 | 381320 | yes |

The exact epoch counts are not universal. They depend on the random seed, data
order, preprocessing, feature scaling, and margin. VC dimension explains the
capacity boundary; it does not predict a specific epoch count.

## Repository layout

```text
afhq_vc_perceptron_experiment/
  README.md
  requirements.txt
  docs/
    perceptron_memorization_textbook.html
    perceptron_memorization_textbook.pdf
  notebooks/
    01_afhq_vc_perceptron_demo.ipynb
  scripts/
    download_afhq.py
    run_experiment.py
    make_tutorial_figures.py
  src/afhq_vc_demo/
    config.py
    data.py
    experiment.py
    models.py
    reporting.py
  outputs/
    figures/
    tables/
    summary.md
  tests/
    test_core.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This project pins `kagglehub==0.3.13` because a newer local KaggleHub install
hit a `kagglesdk` import mismatch during verification.

## Download the data

Try the KaggleHub path first:

```bash
python scripts/download_afhq.py --source kagglehub
```

If KaggleHub asks for credentials, use the Kaggle CLI after configuring
`~/.kaggle/kaggle.json`:

```bash
python scripts/download_afhq.py --source kaggle
```

If Kaggle is unavailable, the downloader can also use the official AFHQ mirror
from the StarGAN v2 authors:

```bash
python scripts/download_afhq.py --source official
```

## Reproduce the experiment

```bash
python scripts/run_experiment.py --data-root data/raw/kaggle_animal_faces --sample-sizes 500 1000 2000 4096 4097 5000
python scripts/make_tutorial_figures.py
```

The main reproducible outputs are written to:

- `outputs/tables/results.csv`
- `outputs/tables/perceptron_long_run.csv`
- `outputs/summary.md`
- `outputs/figures/`

## Synthetic smoke test

If AFHQ is not downloaded yet, this verifies the code path with random
4096-dimensional vectors:

```bash
python scripts/run_experiment.py --synthetic --sample-sizes 500 1000 2000 4096 4097 5000
```

The synthetic test is only a smoke test. The reported result in the chapter uses
the AFHQ image data.
