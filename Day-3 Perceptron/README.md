# Day 3: Perceptron

This folder contains the implementation for **Day 3** of the "100 Days of Deep Learning" journey. The focus of this day is understanding and implementing a basic **Perceptron**, which is the fundamental building block of neural networks.

## Overview

The goal of this project is to build a linear binary classifier using the Perceptron algorithm to predict whether a student will be placed based on their academic and resume scores.

## Dataset
- **File**: `placement.csv`
- **Features**: 
  - `cgpa`: The student's Cumulative Grade Point Average.
  - `resume_score`: A numerical score given to the student's resume.
- **Target**: 
  - `placed`: A binary label indicating whether the student was successfully placed (`1`) or not (`0`).

## Workflow
1. **Data Loading**: Loaded the dataset using `pandas`.
2. **Exploratory Data Analysis (EDA)**: Visualized the dataset using `seaborn` scatterplots to see the relationship between CGPA, Resume Score, and Placement status.
3. **Data Preprocessing**: Separated the input features (`X`) from the target labels (`y`).
4. **Model Training**: Initialized and trained a `Perceptron` model from `scikit-learn` on the dataset.
5. **Model Evaluation**: Examined the model's learned weights (`coef_`) and bias (`intercept_`), and calculated its accuracy using `accuracy_score`.
6. **Decision Boundary Visualization**: Used `mlxtend.plotting` to visualize the linear decision regions created by the Perceptron.

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `mlxtend`

## Usage
Run the `Perceptron.ipynb` Jupyter Notebook step-by-step to see the data visualizations and the training process of the Perceptron model.
