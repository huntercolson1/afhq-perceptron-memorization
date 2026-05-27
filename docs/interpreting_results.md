# Interpreting The Results

The labels in this experiment are randomized. That is the main trick.

If a classifier gets zero training error on these labels, it cannot be because it
learned the difference between cats and dogs. The labels no longer mean cat or
dog. Zero error means the classifier found a way to memorize an arbitrary split.

## Results Summary

| N | rank(X with bias) | exact separator proof error | perceptron error after 50 epochs |
|---:|---:|---:|---:|
| 500 | 500 | 0.0000 | 0.0000 |
| 1000 | 1000 | 0.0000 | 0.0340 |
| 2000 | 2000 | 0.0000 | 0.1585 |
| 4096 | 4096 | 0.0000 | 0.2925 |
| 4097 | 4097 | 0.0000 | 0.2563 |
| 5000 | 4097 | 0.0262 | 0.2252 |

## What The Columns Mean

`N` is the number of randomly labeled training examples.

`rank(X with bias)` measures how many independent directions the bias-augmented
image matrix provides. Once `N` exceeds 4097, the rank cannot keep increasing
because there are only 4097 columns.

`exact separator proof error` comes from solving `X_aug @ w = y`. It asks:
does a linear separator exist that matches these randomized labels?

`perceptron error after 50 epochs` comes from actually running the perceptron
learning rule for a fixed number of passes through the data.

## Main Takeaway

The exact separator proof fits the randomized labels up through `N = 4097`.
That proves a separating hyperplane exists for those examples and labels. Since
the labels are random, the separator is memorizing an arbitrary assignment, not
learning cats versus dogs.

At `N = 5000`, the rank is still only 4097, and the exact separator proof no longer
fits every randomized label.

The perceptron training rule is more practical and more limited: it only reached
zero training error for `N = 500` within the short 50-epoch sweep. When we gave
the same learning rule more epochs, it reached zero training error all the way
through `N = 4097`:

| N | Epochs to zero error | Updates |
|---:|---:|---:|
| 500 | 48 | 2638 |
| 1000 | 80 | 9046 |
| 2000 | 214 | 44162 |
| 4096 | 945 | 365288 |
| 4097 | 1228 | 381320 |

That is the concrete "given enough iterations" demonstration. The perceptron
convergence theorem explains why this should happen once separability is proven.
The VC-dimension statement itself does not predict the exact epoch count; the
epoch count depends on the margin, feature scaling, random labels, and training
order. See `docs/why_epochs_vary.md`.

So the clean proof is:

```text
full row rank up to 4097
=> exact separator exists for arbitrary labels
=> random labels can be perfectly separated
=> the true cat/dog labels, which are one possible labeling, can also be separated
=> perfect training accuracy at this size does not prove semantic learning
=> perceptron convergence theorem implies eventual perceptron convergence
   on the separable cases, given enough updates
```

## Why This Matters

High-dimensional models can look impressive on training data for the wrong
reason. This is one of the simplest examples of why held-out validation data,
regularization, and careful experimental design matter.

## What To Look At Visually

The most useful figures for a tutorial or blog post are:

- `outputs/figures/rank_vs_sample_size.png`: why rank stops at 4097.
- `outputs/figures/training_error.png`: exact separator proof versus a finite
  perceptron run.
- `outputs/figures/perceptron_long_run_epochs.png`: how many epochs the
  perceptron needed once we gave it enough time.
- `outputs/figures/perceptron_training_journey_n500.png`: error over time plus
  weight heatmaps.
- `outputs/figures/weight_update_story_n500.png`: raw images, signed updates,
  and the final learned weights as an image.
