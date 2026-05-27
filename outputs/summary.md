# AFHQ Perceptron VC-Dimension Demo Results

Data source: Kaggle AFHQ dataset (andrewmvd/animal-faces), local data/raw

Image shape: 64 x 64 grayscale

Input dimension: 4096; bias-augmented dimension: 4097

| N | rank(X with bias) | VC construction train error | Perceptron train error | Perceptron converged? |
|---:|---:|---:|---:|:---|
| 500 | 500 | 0.0000 | 0.0000 | yes |
| 1000 | 1000 | 0.0000 | 0.0340 | no |
| 2000 | 2000 | 0.0000 | 0.1585 | no |
| 4096 | 4096 | 0.0000 | 0.2925 | no |
| 4097 | 4097 | 0.0000 | 0.2563 | no |
| 5000 | 4097 | 0.0262 | 0.2252 | no |

Labels were randomized. Zero training error means memorization of an arbitrary split.

The linear-system row reports an exact-value construction, not a perceptron training trace. The perceptron row reports finite training with the configured epoch limit; non-convergence within that limit is evidence of practical difficulty, not a formal proof of non-separability.
