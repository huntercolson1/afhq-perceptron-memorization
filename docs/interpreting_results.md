# Interpreting The Results

The labels in this experiment are randomized. That is the main trick.

If a classifier gets zero training error on these labels, it cannot be because it
learned the difference between cats and dogs. The labels no longer mean cat or
dog. Zero error means the classifier found a way to memorize an arbitrary split.

## Results Summary

| N | rank(X with bias) | exact linear construction error | perceptron error after 50 epochs |
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

`exact linear construction error` comes from solving a linear system. It asks:
does a linear separator exist that matches these randomized labels?

`perceptron error after 50 epochs` comes from actually running the perceptron
learning rule for a fixed number of passes through the data.

## Main Takeaway

The exact linear construction fits the randomized labels up through `N = 4097`.
That is the VC-dimension idea showing up in real image data after resizing to
64 x 64 grayscale.

At `N = 5000`, the rank is still only 4097, and the exact construction no longer
fits every randomized label.

The perceptron training rule is more practical and more limited: it only reached
zero training error for `N = 500` within 50 epochs. That does not contradict the
math. It shows that a separator can exist without being easy for this particular
training procedure to find quickly.

## Why This Matters

High-dimensional models can look impressive on training data for the wrong
reason. This is one of the simplest examples of why held-out validation data,
regularization, and careful experimental design matter.
