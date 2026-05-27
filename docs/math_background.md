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

## What This Does Not Mean

It does not mean the model understands cats and dogs.

It does not mean every perceptron training run will quickly find the separator.

It does not mean that every random labeling above 4097 examples is impossible to
separate.

It means the dimensionality is large enough that perfect training accuracy can
be misleading.
