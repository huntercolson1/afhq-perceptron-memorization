# Why Do The Epoch Counts Vary?

The short answer is:

```text
VC dimension tells us whether a separator can exist.
It does not tell us how quickly the perceptron will find one.
```

Those are different questions.

## The Two Questions

The rank and VC-dimension argument asks:

```text
Is there some hyperplane that can classify these training points perfectly?
```

The perceptron training run asks:

```text
How long does this particular update rule take to find such a hyperplane?
```

For this experiment, the answer to the first question is clean through
`N = 4097`: yes, a separator exists for the randomized labels.

The answer to the second question depends on geometry.

## The Perceptron Mistake Bound

There is a classic bound for the perceptron. Suppose the data are linearly
separable and every input has norm at most `R`. Suppose there is a unit-length
separator with margin `gamma`.

Then the perceptron makes at most:

```text
(R / gamma)^2
```

mistakes before it converges.

This is useful, but it is not a convenient stopwatch. It says convergence is
finite when the data are separable, but it usually does not give a tight
prediction for the exact epoch count.

## What Is The Margin?

The margin is the breathing room around the separating hyperplane.

If the closest examples sit far from the boundary, the margin is large. The
perceptron tends to find a separator quickly.

If some examples sit barely on the correct side, the margin is small. The
perceptron may need many updates.

Random labels in high dimensions often create a separator with very little
breathing room. That is why the existence proof can be simple while training
still takes a while.

## What Else Changes The Epoch Count?

Epoch counts can change when you change:

- the random labels;
- the order of examples during training;
- the feature scaling;
- the learning rate convention;
- the particular subset of images;
- the stopping rule;
- numerical precision.

So the exact number of epochs is an empirical result for this run, not a
universal mathematical constant.

## What Happened In This Run?

The longer perceptron runs reached zero training error through the VC boundary:

| N | Epochs to zero error | Updates |
|---:|---:|---:|
| 500 | 48 | 2638 |
| 1000 | 80 | 9046 |
| 2000 | 214 | 44162 |
| 4096 | 945 | 365288 |
| 4097 | 1228 | 381320 |

The pattern is the point. As the number of arbitrary labels gets closer to the
capacity boundary, the separator can still exist, but it can be harder for this
simple update rule to find.

## The Teaching Takeaway

Do not read the VC-dimension statement as:

```text
the perceptron quickly learns any 4097 examples
```

Read it as:

```text
the model class is expressive enough to fit any 4097 well-positioned examples
```

The perceptron convergence theorem then says the update rule can eventually find
a separator when the data are separable. The mistake bound explains why
"eventually" can mean very different numbers of updates for different datasets.
