# Graduate Admission Predictor ANN

This project uses a feed-forward artificial neural network to predict a graduate applicant's chance of admission from the `admission_data.csv` dataset.

## Workflow

1. Load and inspect the admission dataset with `pandas`.
2. Separate the predictor features from the `Chance of Admit ` target.
3. Split the data into training and test sets.
4. Scale the features with `MinMaxScaler`.
5. Train a 7-7-1 TensorFlow/Keras regression network for 100 epochs.
6. Evaluate the predictions with the R² score and plot training and validation loss.

## Files

- `Graduate-Admission-Predictor.ipynb`: Notebook containing data exploration, model training, and evaluation.
- `admission_data.csv`: Graduate admission dataset.

## Usage

Open the notebook in Jupyter Notebook or VS Code and run the cells in order. The project uses `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `tensorflow`.