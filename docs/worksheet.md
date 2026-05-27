# Worksheet

This worksheet is meant to be solved by hand before or after running the
notebook. The goal is to make the VC-dimension idea feel less mysterious.

## Part 1: Images As Vectors

1. A grayscale image is 64 pixels wide and 64 pixels tall. How many pixel
   features does it have?

2. A linear classifier also has a bias term. After adding the bias, how many
   adjustable numbers does the classifier have?

3. Why does this project keep comparing sample sizes to `4097`?

## Part 2: A Tiny Perceptron

For a 2D input, the perceptron score is:

```text
score = w1*x1 + w2*x2 + b
```

Use:

```text
w1 = 2
w2 = 2
b = -1
```

Classify these points by the sign of the score:

| point | x1 | x2 | score | predicted label |
|---:|---:|---:|---:|---:|
| A | 0 | 0 | | |
| B | 1 | 0 | | |
| C | 0 | 1 | | |

## Part 3: Random Labels

Suppose a model perfectly fits labels that were assigned by coin flip.

1. Did it learn a real pattern?
2. What did it probably do instead?
3. Why is this a warning about training accuracy?

## Part 4: Perceptron Convergence

The perceptron convergence theorem says:

```text
If the data are linearly separable, the perceptron learning rule eventually
finds a separating hyperplane.
```

Answer in plain language:

1. What does "linearly separable" mean?
2. Why can "eventually" still take many epochs?
3. Why is a rank/exact-separator proof useful before running many epochs?

## Part 5: Weights As Stored Information

The perceptron update can be written as:

```text
w <- w + y_i*x_i
```

1. If `y_i = +1`, what happens to the image vector?
2. If `y_i = -1`, what happens to the image vector?
3. Why can the final weight vector be reshaped into a 64 x 64 heatmap?
4. Why does an animal-looking heatmap not prove the model learned the concept
   "cat versus dog"?

## Answer Key

### Part 1

1. `64 x 64 = 4096` pixel features.
2. `4096 + 1 = 4097` including the bias.
3. Because a linear separator in 4096 dimensions has VC dimension `4097`.

### Part 2

| point | x1 | x2 | score | predicted label |
|---:|---:|---:|---:|---:|
| A | 0 | 0 | -1 | -1 |
| B | 1 | 0 | +1 | +1 |
| C | 0 | 1 | +1 | +1 |

### Part 3

1. No, not in any meaningful semantic sense.
2. It memorized the training assignment.
3. Because training accuracy can be high even when the model has not learned a
   rule that generalizes.

### Part 4

1. One hyperplane can perfectly separate the two classes.
2. The number of updates depends on the margin. A tiny margin can make training
   slow.
3. The rank proof tells us a separator exists before we spend time asking the
   perceptron update rule to find one.

### Part 5

1. The image vector is added to the weights.
2. The image vector is subtracted from the weights.
3. Because the main weight vector has one number per pixel.
4. Because in the random-label experiment, the target labels do not encode the
   real cat/dog concept. The heatmap can look image-like simply because it was
   built from image-like updates.
