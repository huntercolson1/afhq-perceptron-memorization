# High-Dimensional Perceptron Memorization on AFHQ

This is a personal-tutor style learning module about a subtle point in machine
learning: high-dimensional linear models can sometimes perfectly fit arbitrary
labels without learning the concept we care about.

The concrete question:

> Can a 64 x 64 grayscale perceptron memorize random labels on cat/dog face
> images just because the input dimension is large?

The experiment uses Kaggle's `andrewmvd/animal-faces` dataset, also known as
Animal Faces-HQ (AFHQ). It contains cat, dog, and wildlife face images at
512 x 512 resolution. This experiment resizes cat/dog images to 64 x 64
grayscale, so each image is a 4096-dimensional vector.

## Inspiration

This project was inspired by the perceptron and VC-dimension discussion in
[*The Welch Labs Illustrated Guide to AI*](https://www.welchlabs.com/resources/ai-book-ezrzm-msrmc).
The goal here is to turn that idea into a self-contained, runnable tutorial.

## How to use this repo

If you are new to this topic, read it in this order:

1. Start with `docs/learning_module.md`.
2. Work through `docs/worked_toy_example.md`.
3. Try `docs/worksheet.md`.
4. Download or print `docs/perceptron_vc_dimension_worksheet.pdf`.
5. Read `docs/exact_separator_proof.md`.
6. Read `docs/why_epochs_vary.md`.
7. Read `docs/how_weights_store_information.md`.
8. Open `notebooks/01_afhq_vc_perceptron_demo.ipynb`.
9. Use `docs/glossary.md` whenever a term feels fuzzy.

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
    make_tutorial_figures.py
    build_worksheet_pdf.py
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
    learning_module.md
    worked_toy_example.md
    worksheet.md
    perceptron_vc_dimension_worksheet.pdf
    math_background.md
    exact_separator_proof.md
    why_epochs_vary.md
    how_weights_store_information.md
    interpreting_results.md
    proof_chain.md
    glossary.md
    RUN_LOG.md
```

## What the demo tests

The experiment deliberately randomizes labels after loading real cat/dog images.
If a model fits those random labels, it is memorizing an arbitrary split rather
than learning cats versus dogs.

It runs two checks:

1. **Separability proof:** for `N <= 4097`, verify full row rank after adding a
   bias column, then directly solve `X_aug @ w = y` to prove a separating
   hyperplane exists for arbitrary random labels.
2. **Perceptron learning check:** run Rosenblatt's perceptron learning rule on
   the same randomized labels and report whether it reaches zero training error.

The first check proves a separator exists. The perceptron convergence theorem
then says the perceptron learning rule eventually converges on linearly
separable data, although the number of updates can be much larger than the
finite training budget shown here.

For `N = 5000`, the VC guarantee no longer applies. The demo reports what
actually happens on the resized AFHQ data.

For a more beginner-friendly explanation, see:

- `docs/learning_module.md`
- `docs/worked_toy_example.md`
- `docs/worksheet.md`
- `docs/perceptron_vc_dimension_worksheet.pdf`
- `docs/math_background.md`
- `docs/exact_separator_proof.md`
- `docs/why_epochs_vary.md`
- `docs/how_weights_store_information.md`
- `docs/interpreting_results.md`
- `docs/proof_chain.md`
- `docs/glossary.md`

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
- `outputs/figures/perceptron_long_run_epochs.png`
- `outputs/figures/perceptron_training_journey_n500.png`
- `outputs/figures/random_label_examples_n500.png`
- `outputs/figures/weight_update_story_n500.png`

To rebuild the tutorial figures:

```bash
python scripts/make_tutorial_figures.py
```

To rebuild the downloadable worksheet PDF:

```bash
python -m playwright install chromium
python scripts/build_worksheet_pdf.py
```

That PDF build also expects `pandoc` to be available on your system path.

## Verified result on this Mac

The exact Kaggle dataset downloaded successfully on 2026-05-27, and the notebook
executed top-to-bottom. The durable result table is:

| N | rank(X with bias) | Exact separator proof error | Perceptron train error | Perceptron converged? |
|---:|---:|---:|---:|:---|
| 500 | 500 | 0.0000 | 0.0000 | yes |
| 1000 | 1000 | 0.0000 | 0.0340 | no |
| 2000 | 2000 | 0.0000 | 0.1585 | no |
| 4096 | 4096 | 0.0000 | 0.2925 | no |
| 4097 | 4097 | 0.0000 | 0.2563 | no |
| 5000 | 4097 | 0.0262 | 0.2252 | no |

See `docs/RUN_LOG.md` for commands and verification evidence.

## Perceptron learning-rule convergence demo

The 50-epoch sweep intentionally shows a practical training budget. To
demonstrate the "given enough iterations" part more directly, longer perceptron
runs were also performed on randomized labels:

| N | Max epochs allowed | Epochs to zero error | Updates | Converged? |
|---:|---:|---:|---:|:---|
| 500 | 50 | 48 | 2638 | yes |
| 1000 | 1000 | 80 | 9046 | yes |
| 2000 | 2000 | 214 | 44162 | yes |
| 4096 | 5000 | 945 | 365288 | yes |
| 4097 | 5000 | 1228 | 381320 | yes |

These runs use the actual perceptron learning rule. The rank proof explains why
convergence is possible; the longer runs show it happening concretely.

The exact epoch count is not predicted by the VC-dimension rule. It depends on
the margin, the data order, the random labels, and the feature scaling. See
`docs/why_epochs_vary.md` for the perceptron mistake-bound explanation.

The learned weight vector can also be reshaped into a 64 x 64 image because each
weight corresponds to one pixel. See `docs/how_weights_store_information.md` and
`outputs/figures/weight_update_story_n500.png` for the visual explanation.

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
