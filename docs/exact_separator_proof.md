# What Does "Solve Xw = y" Mean?

This phrase sounds like a different model, but it is really just a direct way to
ask:

```text
Do weights exist that classify every training example correctly?
```

That is the existence question. The perceptron learning rule is the training
question.

## Start With One Image

For one 64 x 64 grayscale image, the perceptron score is:

```text
score = w1*x1 + w2*x2 + ... + w4096*x4096 + b
```

If the label is `+1`, we want the score to be positive.

If the label is `-1`, we want the score to be negative.

That is what it means to separate the example correctly.

## Stack All Images Together

Now put every image into one big table. That table is called `X`.

Each row is one image. Each column is one pixel feature.

Then add one column of `1`s for the bias term. That gives `X_aug`.

For this project:

```text
X_aug has 4097 columns
```

because:

```text
4096 pixels + 1 bias = 4097
```

## The Direct Solve

Now ask whether we can find weights `w` so that:

```text
X_aug @ w = y
```

In plain English:

```text
Can we choose weights so every training score equals its label?
```

If the labels are `-1` and `+1`, this is stronger than ordinary correct
classification. Ordinary classification only needs the right sign. This asks for
the exact label value.

If a row has label `+1`, the score becomes `+1`.

If a row has label `-1`, the score becomes `-1`.

So every point is on the correct side of the hyperplane.

## Why Rank Is The Shortcut

Rank tells us how many independent constraints the matrix has.

If there are `N` examples and:

```text
rank(X_aug) = N
```

then the rows are independent. In that case, the direct solve can match any
`N` labels.

That is the key proof in this repo.

For the AFHQ experiment:

| N | rank(X_aug) |
|---:|---:|
| 500 | 500 |
| 1000 | 1000 |
| 2000 | 2000 |
| 4096 | 4096 |
| 4097 | 4097 |

So for all of those sample sizes, a perfect separator exists for arbitrary
labels, including random labels.

## How This Connects Back To The Perceptron

The direct solve proves:

```text
a perfect separator exists
```

The perceptron convergence theorem says:

```text
if a perfect separator exists, the perceptron learning rule eventually finds
one after finitely many updates
```

So the full chain is:

```text
rank(X_aug) = N
=> X_aug @ w = y has a solution for arbitrary labels
=> a perfect linear separator exists
=> random labels can be separated
=> this is memorization, not visual understanding
=> the perceptron learning rule can eventually find a separator
```

## Why We Still Run Epochs

The proof is clean, but it can feel abstract. So the repo also runs the actual
perceptron learning rule.

The boundary result is the satisfying part:

| N | Epochs to zero training error | Updates |
|---:|---:|---:|
| 4096 | 945 | 365288 |
| 4097 | 1228 | 381320 |

That is the perceptron learning rule actually doing it.

## Why The Epoch Count Is Not The Proof

It is tempting to treat the epoch count as the main mathematical object:

```text
How many epochs should it take?
```

But the proof does not work that way.

The proof has two parts:

1. Full row rank proves a perfect separator exists.
2. The perceptron convergence theorem proves the perceptron learning rule finds
   a separator after finitely many updates.

The exact number of updates depends on the margin and the training order. A
small margin can make convergence much slower even when separability is already
guaranteed. That is why this repo reports epoch counts as empirical evidence,
not as the formal proof itself.

See `docs/why_epochs_vary.md` for the mistake-bound version of that story.
