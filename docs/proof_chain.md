# Proof Chain

This project is designed to support one specific claim:

```text
A perceptron can perfectly separate a small-enough set of 64 x 64 cat/dog
images, even when the labels are arbitrary, so perfect training accuracy alone
does not prove that the model learned the concept "cat versus dog."
```

## Step 1: Turn Images Into A Matrix

Each image is resized to 64 x 64 grayscale and flattened:

```text
64 x 64 = 4096 pixel features
```

Then a bias column of ones is added:

```text
4096 + 1 = 4097 columns
```

Call this bias-augmented matrix `X_aug`.

## Step 2: Check Rank

For `N <= 4097`, the experiment found:

| N | rank(X_aug) |
|---:|---:|
| 500 | 500 |
| 1000 | 1000 |
| 2000 | 2000 |
| 4096 | 4096 |
| 4097 | 4097 |

That means the rows are independent for these sample sizes.

## Step 3: Why Full Row Rank Proves Separability

If `X_aug` has rank `N`, then for any vector of labels `y`, there is a weight
vector `w` such that:

```text
X_aug @ w = y
```

In this project, `y` is a vector of random `-1` and `+1` labels.

If `X_aug @ w = y`, then every example has the correct sign:

```text
y_i * (x_i @ w) = 1 > 0
```

So the hyperplane perfectly separates the training examples.

The real cat/dog labels are also just one possible `-1`/`+1` label vector. If
the matrix can fit any label vector at these sample sizes, it can fit the real
cat/dog labels too.

## Step 4: Why Random Labels Matter

The labels are deliberately randomized.

So when the exact separator fits them perfectly, it cannot be because the model
learned cats versus dogs. The labels no longer mean cat or dog.

This is stronger than showing that the model fits the real labels. If it can
fit random labels, then fitting the real labels at the same sample size is not,
by itself, proof of semantic learning.

## Step 5: Where The Perceptron Learning Rule Enters

The exact separator proof shows that a separating hyperplane exists.

The perceptron convergence theorem says that if a dataset is linearly separable,
then the standard perceptron learning rule will eventually find a separating
hyperplane after a finite number of updates.

The number of updates can be extremely large when the separating margin is
small. That is why this project separates:

- `exact separator proof`: proves separability;
- `50-epoch perceptron run`: shows what happened under a practical finite
  training budget;
- `longer perceptron run`: demonstrates that the learning rule reaches zero
  training error through `N = 4097` when given enough epochs.

In this run, the boundary cases converged:

| N | Epochs to zero error | Updates |
|---:|---:|---:|
| 4096 | 945 | 365288 |
| 4097 | 1228 | 381320 |

## Step 6: What Happens Above The Boundary

At `N = 5000`, the matrix rank is still only `4097`, because there are only
4097 columns. The exact separator proof no longer fits every randomized label:

```text
exact separator proof error = 0.0262
```

That is the expected change once the number of examples exceeds the
bias-augmented input dimension.

## Step 7: What The Weight Vector Is Doing

The perceptron update is:

```text
w <- w + y_i*x_i
```

whenever example `i` is misclassified.

So after training, the weight vector is a sum of signed training images:

```text
w = alpha_1*y_1*x_1 + ... + alpha_N*y_N*x_N
```

This helps explain why the weights can be reshaped into a face-like heatmap.
The heatmap is built out of image updates. In the random-label setting, that
does not mean the model learned cats or dogs; it means the model accumulated a
high-dimensional memorization rule.
