# 🎮 League of Legends Match Predictor

A PyTorch-based logistic regression model to predict the winner of a League of Legends match using only the first 10 minutes of in-game data.

***

## ✨ Key Features

* **Data Processing**: Utilizes `pandas` for data handling and `scikit-learn` for feature scaling and train/test splitting.
* **PyTorch Model**: A custom logistic regression model built using `torch.nn.Module`.
* **Regularization**: Implements L2 regularization (weight decay) during training to prevent overfitting.
* **Performance Evaluation**: Assesses model performance with a confusion matrix, ROC/AUC curve, and a detailed classification report.
* **Hyperparameter Tuning**: Includes a search for the optimal learning rate to maximize accuracy.
* **Feature Importance**: Analyzes and visualizes the model's learned weights to determine the most impactful in-game features.
* **Model Persistence**: Demonstrates how to save and load the trained model using `torch.save` and `torch.load`.

***

## 📊 Results at a Glance

* **Test Accuracy**: **73.58%** (with L2 Regularization)
* **AUC Score**: **0.80**
* **Optimal Learning Rate**: **0.01**

***

## 🛠️ Tech Stack

* Python 3
* PyTorch
* Pandas
* Scikit-learn
* Matplotlib

***

## 📦 Core Libraries

The project relies on the following key libraries and modules:

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
import matplotlib.pyplot as plt
