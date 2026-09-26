# Customer Churn Predictor ANN

This project uses a feed-forward artificial neural network to predict customer churn from the `Churn_Modelling.csv` dataset.

## Workflow

1. Load and inspect the customer data with `pandas`.
2. Remove identifier columns that do not contribute to the prediction.
3. One-hot encode the `Geography` and `Gender` columns.
4. Split the data into training and test sets.
5. Build and train an 11-11-1 dense network with TensorFlow/Keras.
6. Evaluate predictions with classification accuracy and plot training history.

## Files

- `Cusstomer Churn ANN.ipynb`: Notebook containing the data preparation, model training, and evaluation workflow.
- `Churn_Modelling.csv`: Customer churn dataset.

## Usage

Open the notebook in Jupyter Notebook or VS Code and run the cells in order. The notebook installs or uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `tensorflow`.