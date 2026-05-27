# Math Background

This project is about a small but important distinction:

```text
fitting the training labels != learning the real pattern
```

## Images As Vectors

A grayscale 64 x 64 image can be flattened into one long list of pixel values:

```text
64 x 64 = 4096 numbers
```

So each image becomes a point in 4096-dimensional space.

## What A Perceptron Does

A perceptron is a linear classifier. It computes:

```text
score = w1*x1 + w2*x2 + ... + w4096*x4096 + b
```

Then it predicts one class if the score is positive and the other class if the
score is negative. Geometrically, it is trying to draw one hyperplane that
separates the examples.

## Why The Bias Term Matters

The weights contribute 4096 adjustable numbers. The bias term adds one more.
That gives:

```text
4096 + 1 = 4097
```

This is why the experiment highlights the `4097` boundary.

## VC Dimension In Plain Language

The VC dimension of linear separators in `d` dimensions is `d + 1`.

Plain-language meaning:

If you have at most `d + 1` points in general position, a linear classifier can
usually arrange itself to match any labeling of those points, even a completely
random labeling.

For this experiment:

```text
d = 4096
d + 1 = 4097
```

So below or around 4097 examples, a zero-error classifier may simply be
memorizing the training set.

## The Proof Idea Used Here

The experiment does not merely train a perceptron and hope for a good result.
It checks a stronger mathematical condition.

After adding the bias column, the data matrix has 4097 columns. For every sample
size up to 4097 in this run, the matrix had full row rank:

```text
rank(X_aug) = N
```

Full row rank means there is a weight vector that can match any chosen labels on
those examples. In symbols, for labels `y` in `{-1, +1}`:

```text
X_aug @ w = y
```

That immediately gives a perfect separator, because every score has the same
sign as its label.

The actual cat/dog labels are one possible choice of `y`. Random labels are
another possible choice of `y`. Showing that random labels can be separated is
the stronger demonstration, because random labels cannot encode a real visual
cat/dog rule.

The perceptron convergence theorem then supplies the learning-rule link: if the
training set is linearly separable, the perceptron update rule converges after a
finite number of updates. The finite run in this repository is an illustration;
the rank/exact-separator check is the separability proof.

## Why The Number Of Epochs Is Hard To Predict

The VC-dimension result does not say how many epochs training will take.

The classic perceptron mistake bound says that if the data are separable with
margin `gamma`, and every input has norm at most `R`, then the perceptron makes
at most:

```text
(R / gamma)^2
```

mistakes before converging.

This tells us why small-margin separators can be slow to find. It does not
usually give a clean exact epoch prediction for one image dataset, one random
label assignment, and one shuffled training order.

## What This Does Not Mean

It does not mean the model understands cats and dogs.

It does not mean every perceptron training run will quickly find the separator.

It does not mean that every random labeling above 4097 examples is impossible to
separate.

It means the dimensionality is large enough that perfect training accuracy can
be misleading.
