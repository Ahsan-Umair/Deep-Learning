# Day 4: Perceptron Trick

This folder contains the implementation for **Day 4** of the "100 Days of Deep Learning" journey. The notebook builds a binary Perceptron classifier from scratch and visualizes the linear decision boundary it learns.

## Overview

The notebook demonstrates how a Perceptron can classify two linearly separable classes using:

- A step activation function
- A learning rate of `0.1`
- Random single-sample weight updates
- A bias/intercept term added to the input data

The model learns a weight vector and intercept, which are then converted into the equation of a line:

```text
y = mx + b
```

## Workflow

1. **Generate Data**: Create a two-feature, two-class dataset with `sklearn.datasets.make_classification`.
2. **Visualize the Data**: Plot the generated samples with colors representing their class labels.
3. **Add the Bias Term**: Insert a column of ones into the feature matrix so the first weight represents the intercept.
4. **Train the Perceptron**: Repeatedly select a random training example, calculate its prediction with the step function, and update the weights when the prediction is incorrect.
5. **Extract Parameters**: Separate the learned intercept from the feature coefficients.
6. **Plot the Decision Boundary**: Calculate and draw the line represented by the learned parameters over the original data.

## Perceptron Update Rule

For a selected sample, the prediction is calculated as:

```text
prediction = step(X[j] dot weights)
```

The weights are updated using:

```text
weights = weights + learning_rate * (actual - prediction) * sample
```

The step function returns `1` for positive values and `0` otherwise.

## Libraries Used

- `numpy`
- `matplotlib`
- `scikit-learn`

## Usage

Open and run `perceptron-trick.ipynb` cell by cell in Jupyter Notebook or VS Code. The notebook displays the generated data first and then overlays the decision boundary learned by the custom Perceptron implementation.

Because the training loop selects samples randomly and does not set a NumPy seed, the learned parameters and plotted boundary may vary slightly between runs.
