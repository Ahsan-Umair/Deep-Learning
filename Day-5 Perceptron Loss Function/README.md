# Day 5: Perceptron Loss Function

This folder contains the implementation for **Day 5** of the "100 Days of Deep Learning" journey. The notebook explores a loss-based view of the Perceptron by checking the signed margin of each training example and updating the model when that margin is negative.

## Overview

The notebook builds a two-feature binary classifier from scratch. It generates linearly separable data, trains a Perceptron using explicit weights and bias, and plots the learned decision boundary.

The model parameters are initialized as:

- `w1 = 1`
- `w2 = 1`
- `b = 1`

The learning rate is set to `0.1`.

## Workflow

1. **Generate Data**: Create 100 samples with two features and two classes using `sklearn.datasets.make_classification`.
2. **Inspect the Shapes**: Check the dimensions of the feature matrix `X` and label array `y`.
3. **Visualize the Samples**: Plot the two classes with `matplotlib`.
4. **Calculate the Score**: For each sample, compute the linear score:

   ```text
   z = w1 * x1 + w2 * x2 + b
   ```

5. **Check the Margin**: Use the condition `z * y < 0` to identify samples that violate the desired classification margin.
6. **Update the Parameters**: Adjust both feature weights and the bias using the learning rate and the sample label.
7. **Plot the Boundary**: Convert the learned parameters into a line and draw it over the data points.

## Perceptron Loss Idea

For a sample with label `y`, the signed margin is represented by:

```text
margin = y * (w1 * x1 + w2 * x2 + b)
```

A negative margin indicates an incorrectly classified sample under the signed-label formulation. The notebook updates the parameters using:

```text
w1 = w1 + learning_rate * y * x1
w2 = w2 + learning_rate * y * x2
b  = b  + learning_rate * y
```

## Decision Boundary

After training, the decision boundary is derived from:

```text
w1 * x1 + w2 * x2 + b = 0
```

and plotted in slope-intercept form:

```text
y = -(w1 / w2) * x - (b / w2)
```

## Libraries Used

- `numpy`
- `matplotlib`
- `scikit-learn`

## Usage

Open and run `hinge-loss-perceptron.ipynb` cell by cell in Jupyter Notebook or VS Code. The notebook first displays the generated data and then overlays the decision boundary produced by the custom training loop.

## Label Encoding Note

The margin condition and update rule used in the notebook are written for signed labels, conventionally `-1` and `+1`. The dataset generator currently returns labels in the `0/1` format by default. For a standard hinge-loss Perceptron implementation, convert the labels before training, for example with:

```python
y = np.where(y == 0, -1, 1)
```
