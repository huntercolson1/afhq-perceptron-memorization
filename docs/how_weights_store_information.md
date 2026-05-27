# How Do The Weights Store Information?

A perceptron does not store examples in a database. It stores information in one
long weight vector.

For a 64 x 64 grayscale image, that vector has:

```text
4096 pixel weights + 1 bias weight = 4097 adjustable numbers
```

Each pixel gets its own knob.

## The Update Rule Is The Key

For labels in `{-1, +1}`, the perceptron update is:

```text
if y_i * (w dot x_i + b) <= 0:
    w <- w + y_i * x_i
    b <- b + y_i
```

Plain English:

```text
When the model is wrong, add or subtract the whole image vector.
```

If the label is `+1`, the image vector gets added to the weights.

If the label is `-1`, the image vector gets subtracted from the weights.

After many mistakes, the weight vector has this form:

```text
w = alpha_1*y_1*x_1 + alpha_2*y_2*x_2 + ... + alpha_N*y_N*x_N
```

The `alpha_i` values count how often each example caused an update.

That means the weights are literally an accumulation of signed training images.

## Why The Weight Image Looks Like An Animal Face

Because each feature is a pixel, we can reshape the 4096 learned pixel weights
back into a 64 x 64 image.

This does not mean the perceptron has learned a clean cat detector or dog
detector. In the randomized-label experiment, the labels are meaningless.

The image-like pattern appears because the updates are made from images.

So the right interpretation is:

```text
The weights are a compressed signed mixture of training examples.
```

They can look face-like while still representing memorization.

## Why Random Labels Make This So Useful

With true cat/dog labels, a face-like weight image can tempt us into saying:

```text
the model found cat features
```

But when labels are random, that interpretation breaks.

The same update rule still creates image-like weights, but the target it is
trying to hit is arbitrary. That is the lesson:

```text
visual-looking weights are not automatically semantic understanding
```

## Where To See This In The Repo

The figure below is generated from the real AFHQ images used in the experiment:

```text
outputs/figures/weight_update_story_n500.png
```

It shows raw images, the signed image-like update each example would contribute
if it caused a mistake, and the final learned weight vector reshaped as a
64 x 64 image.

The companion figure:

```text
outputs/figures/perceptron_training_journey_n500.png
```

shows the training error over epochs alongside weight snapshots.
