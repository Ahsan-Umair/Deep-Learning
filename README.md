# 100 Days of Deep Learning

Hands-on notebooks from a deep-learning study series. This repository currently contains three perceptron exercises and two applied artificial-neural-network projects; the day numbers describe the included lessons, not a claim that all 100 days are present.

## Notebook index

| Folder | What the notebook demonstrates |
| --- | --- |
| [Day 3: Perceptron trick](Day-3%20Perceptron%20Trick/README.md) | Scikit-learn perceptron on student placement data; inspect accuracy and the learned decision boundary. |
| [Day 4: Perceptron from scratch](Day-4%20Perceptron/README.md) | Implement sample-by-sample weight updates and draw a line separating synthetic classes. |
| [Day 5: Perceptron loss](Day-5%20Perceptron%20Loss%20Function/README.md) | Explore signed margins, parameter updates, and the resulting decision boundary. |
| [Customer Churn Predictor ANN](Projects/Customer%20Churn%20Predictor%20ANN/README.md) | Prepare bank-customer data and train an 11-11-1 dense classifier in Keras. |
| [Graduate Admission Predictor ANN](Projects/Gradute%20Admission%20Predictor%20ANN/README.md) | Scale admission features and train a 7-7-1 dense regression network. |

The [Projects index](Projects/README.md) groups the applied work. Each linked folder has its own README with the dataset, notebook steps, and relevant dependencies.

## Learning progression

1. **Day 3** fits scikit-learn's perceptron to placement records with `cgpa` and `resume_score` predictors.
2. **Day 4** replaces the library model with an explicit step function, bias term, and iterative weight updates on synthetic two-class data.
3. **Day 5** examines the margin-based update rule. The notebook uses `0/1` labels from the data generator, while the signed-margin rule is normally formulated for `-1/+1`; see that folder's README before interpreting its training behavior.
4. **Applied projects** move to TensorFlow/Keras dense networks for binary customer churn and continuous admission chance prediction.

## Run the notebooks

Open this repository in JupyterLab or VS Code and run a notebook from its own folder so relative dataset paths resolve. A base environment for the perceptron exercises is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn mlxtend
jupyter lab
```

The two ANN projects also require `tensorflow`. Their folder READMEs describe the training and evaluation steps. Notebook results can vary between runs where a random seed is not fixed.
