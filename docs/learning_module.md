# Learning Module: Perceptrons, VC Dimension, And Memorization

## Learning Goals

By the end of this module, you should be able to explain:

- what a perceptron is;
- why a 64 x 64 grayscale image lives in 4096-dimensional space;
- what it means for a hyperplane to separate examples;
- why the number `4097` matters here;
- how random labels demonstrate memorization rather than understanding;
- how the perceptron convergence theorem relates to "given enough iterations";
- why perfect training accuracy can be misleading.

## 1. The Starting Problem

Suppose we have images of cats and dogs. A natural question is:

```text
Can a simple perceptron learn to separate cats from dogs?
```

But there is a hidden trap. Images have many pixel features. A 64 x 64 grayscale
image has 4096 numbers. In such a high-dimensional space, a simple linear model
can sometimes fit surprisingly complicated label assignments.

That means a model can get perfect training accuracy for a boring reason:

```text
It memorized the training examples.
```

This project demonstrates that effect, step by step.

## 2. A Perceptron From First Principles

A perceptron computes a weighted sum:

```text
score = w1*x1 + w2*x2 + ... + wd*xd + b
```

Then it predicts based on the sign:

```text
score > 0  => class +1
score < 0  => class -1
```

The weights `w` decide the orientation of the separating hyperplane. The bias
`b` shifts it.

For a 2D toy example, the hyperplane is a line. For a 3D example, it is a plane.
For a 4096D image vector, it is still called a hyperplane, even though we cannot
visualize it directly.

## 3. Why 4097 Matters

The image contributes:

```text
64 x 64 = 4096 features
```

The bias adds one more adjustable term:

```text
4096 + 1 = 4097
```

Linear separators in `d` dimensions have VC dimension `d + 1`. Here, that number
is `4097`.

Plain-language version:

```text
With up to 4097 well-positioned examples, a linear classifier can fit any
assignment of binary labels.
```

"Any assignment" includes the true cat/dog labels, but it also includes
completely random labels.

## 4. Why Random Labels Are The Key

If we train on true labels and get perfect training accuracy, we might be
tempted to say:

```text
The perceptron learned cats versus dogs.
```

But if the same kind of model can fit random labels, then perfect training
accuracy alone is not enough evidence.

Random labels deliberately destroy the real pattern. If a model fits them, it is
memorizing.

## 5. What This Project Proves

The project uses AFHQ cat/dog images resized to 64 x 64 grayscale.

For each sample size, it checks the rank of the bias-augmented data matrix.

The first key result:

| N | rank(X with bias) | exact separator proof error |
|---:|---:|---:|
| 500 | 500 | 0.0000 |
| 1000 | 1000 | 0.0000 |
| 2000 | 2000 | 0.0000 |
| 4096 | 4096 | 0.0000 |
| 4097 | 4097 | 0.0000 |
| 5000 | 4097 | 0.0262 |

For `N <= 4097`, the rows are independent. That means an exact linear separator
exists for arbitrary labels.

At `N = 5000`, the rank is capped at 4097, and the exact separator proof no longer
fits every randomized label.

## 6. Where The Perceptron Learning Rule Fits

The exact separator proof proves existence:

```text
There is a separating hyperplane.
```

The perceptron convergence theorem connects that existence proof to the learning
rule:

```text
If the data are linearly separable, the perceptron learning rule eventually
finds a separating hyperplane after finitely many updates.
```

In practice, "finitely many" can still be a lot. The bound depends on the margin:
how confidently the separating hyperplane separates the closest points.

This project therefore includes two training views:

| N | Short run max epochs | Short run converged? | Longer run epochs to zero error |
|---:|---:|:---|---:|
| 500 | 50 | yes | 48 |
| 1000 | 50 | no | 80 |
| 2000 | 50 | no | 214 |
| 4096 | 50 | no | 945 |
| 4097 | 50 | no | 1228 |

The longer runs demonstrate the actual perceptron learning rule reaching zero
training error on randomized labels. The rank proof explains why that can happen.

## 7. Can We Predict The Number Of Epochs?

Not exactly from VC dimension alone.

VC dimension tells us about capacity:

```text
Can a separator exist?
```

Epoch count tells us about optimization:

```text
How long did this update rule take to find one?
```

Those are related, but not the same.

The classic perceptron mistake bound says that if every input has norm at most
`R` and the data are separable with margin `gamma`, then the perceptron makes no
more than:

```text
(R / gamma)^2
```

mistakes before converging.

That explains the direction of the result: as we ask the model to fit more
arbitrary labels, the separating boundary usually has less breathing room, so
the perceptron can need many more updates. But the bound is usually not tight
enough to predict the exact epoch count.

For the full explanation, see `docs/why_epochs_vary.md`.

## 8. What "Solving Xw = y" Is Doing

This phrase sounds more intimidating than it is.

`X` is the table of examples. Each row is one flattened image. `w` is the list
of weights we want to find. `y` is the list of labels.

So:

```text
X_aug @ w = y
```

means:

```text
find weights that make every training score equal its label
```

If the label is `+1`, the score becomes `+1`. If the label is `-1`, the score
becomes `-1`. Either way, every score has the right sign.

This is not replacing the perceptron learning rule. It is proving that a perfect
separator exists. Once a separator exists, the perceptron convergence theorem
says the perceptron learning rule can eventually find one.

## 9. How The Weights Store Information

Here is the perceptron update rule again:

```text
w <- w + y_i*x_i
```

when the model makes a mistake.

That means the weight vector becomes a sum of signed training images:

```text
w = alpha_1*y_1*x_1 + alpha_2*y_2*x_2 + ... + alpha_N*y_N*x_N
```

where each `alpha_i` counts how many times example `i` caused an update.

So the weights are not a magical idea of "catness" or "dogness." They are an
accumulation of pixel patterns pushed positive or negative by the labels.

Because the weights correspond to pixels, we can reshape them back into a
64 x 64 image. The resulting heatmap can look animal-like because it was built
from animal images. But in the random-label experiment, it is still memorizing
an arbitrary assignment.

See `docs/how_weights_store_information.md` and:

```text
outputs/figures/weight_update_story_n500.png
outputs/figures/perceptron_training_journey_n500.png
```

## 10. Why This Matters Beyond This Demo

This is a small example of a large lesson:

```text
Training performance is not the same thing as generalization.
```

Modern models have far more capacity than a perceptron. If evaluation is only
done on the training set, a model can look impressive while merely memorizing.

That is why machine learning workflows use held-out validation sets, test sets,
regularization, and careful experimental design.

## Suggested Reading Order

1. `README.md`
2. `docs/learning_module.md`
3. `docs/worked_toy_example.md`
4. `docs/worksheet.md`
5. `docs/perceptron_vc_dimension_worksheet.pdf`
6. `docs/math_background.md`
7. `docs/exact_separator_proof.md`
8. `docs/why_epochs_vary.md`
9. `docs/how_weights_store_information.md`
10. `docs/proof_chain.md`
11. `docs/interpreting_results.md`
12. `notebooks/01_afhq_vc_perceptron_demo.ipynb`
