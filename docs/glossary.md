# Glossary

## Bias Term

An extra constant term added to a linear model. It lets the decision boundary
shift away from the origin.

## Binary Classification

A classification problem with two possible labels, such as `-1` and `+1`.

## Epoch

One full pass through the training dataset.

## Feature

An input variable used by a model. In this project, each grayscale pixel is one
feature.

## Generalization

How well a model performs on new examples it did not train on.

## Grayscale Image

An image represented by one brightness value per pixel instead of separate red,
green, and blue color channels.

## Hyperplane

The decision boundary of a linear classifier. In two dimensions it is a line; in
three dimensions it is a plane; in higher dimensions it is called a hyperplane.

## Linear Separability

A dataset is linearly separable if one hyperplane can perfectly separate the two
classes.

## Margin

The distance between the separating hyperplane and the closest training example.
The perceptron can take many more updates when the margin is small.

## Mistake Bound

A theorem-level upper bound on how many mistakes the perceptron can make before
converging on separable data. A common form is `(R / gamma)^2`, where `R` bounds
the input norms and `gamma` is the margin of a unit separator.

## Memorization

Fitting the training examples without learning a rule that generalizes.

## Perceptron

A simple linear classifier that predicts based on the sign of a weighted sum.

## Perceptron Convergence Theorem

If the training data are linearly separable, the perceptron learning rule will
find a separating hyperplane after a finite number of updates.

## Perceptron Learning Rule

The update rule that changes the weights whenever the perceptron misclassifies a
training example.

For labels in `{-1, +1}`, one common form is:

```text
if y_i * (w dot x_i + b) <= 0:
    w <- w + y_i * x_i
    b <- b + y_i
```

## Rank

The number of independent directions in a matrix. In this project, full row rank
means the data matrix has enough independent information to fit arbitrary labels
on those examples.

## Training Accuracy

Accuracy measured on the examples the model trained on. High training accuracy
does not necessarily imply generalization.

## VC Dimension

A measure of model capacity. For linear separators in `d` dimensions, the VC
dimension is `d + 1`.

## Weight Vector

The list of learned numbers used by a linear model. In this project, the main
part of the weight vector has one number per image pixel, so it can be reshaped
back into a 64 x 64 heatmap.
