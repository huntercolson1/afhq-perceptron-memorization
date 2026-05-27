# Worked Toy Example

This tiny example is here because 4096-dimensional geometry is hard to picture.
The same idea is much easier in two dimensions.

## The Setup

Imagine three points in 2D:

| point | x1 | x2 |
|---:|---:|---:|
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 1 |

A 2D linear classifier has two feature weights plus a bias:

```text
score = w1*x1 + w2*x2 + b
```

That is three adjustable numbers:

```text
w1, w2, b
```

So the bias-augmented dimension is:

```text
2 + 1 = 3
```

## Add The Bias Column

The bias-augmented data matrix is:

```text
[0 0 1]
[1 0 1]
[0 1 1]
```

This matrix has rank 3. That means its rows are independent.

## Choose Arbitrary Labels

Now give the points arbitrary labels:

| point | label |
|---:|---:|
| A | -1 |
| B | +1 |
| C | +1 |

Because the matrix has full row rank, there is a vector `[w1, w2, b]` that makes:

```text
X_aug @ w = y
```

For these labels, one solution is:

```text
w1 = 2
w2 = 2
b  = -1
```

Check it:

```text
A: 2*0 + 2*0 - 1 = -1
B: 2*1 + 2*0 - 1 = +1
C: 2*0 + 2*1 - 1 = +1
```

The classifier got all three labels exactly right.

## Why This Matters

In two dimensions, `d + 1 = 3`. So three well-positioned points can be labeled
in any binary pattern and still be separated by a line.

In the image experiment, `d + 1 = 4097`. The same idea applies, just in a space
we cannot draw.
