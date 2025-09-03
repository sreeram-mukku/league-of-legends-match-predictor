# League of Legends Match Predictor using PyTorch

This project builds a logistic regression model to predict the outcomes of League of Legends matches based on in-game statistics from the first 10 minutes. The model is implemented using PyTorch, and the project covers a complete machine learning workflow from data preprocessing to feature importance analysis.

## 📊 Dataset

The model is trained on the "High Diamond Ranked Games (10 min)" dataset. Although the initial instructions mentioned `league_of_legends_data_large.csv`, the actual implementation uses `high_diamond_ranked_10min.csv`.

- **Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
- **Target Variable**: `blueWins` (1 if the blue team wins, 0 otherwise)

## ⚙️ Project Workflow

The project is structured into eight key steps, covering the end-to-end process of building and evaluating a machine learning model.

### 1. Data Loading and Preprocessing
- The dataset is loaded using `pandas`.
- The data is split into features (X) and the target variable (y).
- The dataset is divided into training (80%) and testing (20%) sets using `scikit-learn`'s `train_test_split`.
- Features are standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1.
- The preprocessed data is converted into PyTorch tensors for model training.

### 2. Logistic Regression Model Implementation
- A logistic regression model is defined as a class `LogisticRegressionModel` that inherits from `torch.nn.Module`.
- The model consists of a single linear layer (`nn.Linear`) that maps the input features to a single output.
- A sigmoid activation function (`torch.sigmoid`) is applied in the `forward` method to produce a probability score between 0 and 1.

### 3. Model Training
- The model is trained for 100 epochs.
- The **Binary Cross-Entropy Loss** (`nn.BCELoss`) is used as the loss function, suitable for binary classification tasks.
- **Stochastic Gradient Descent** (`optim.SGD`) with a learning rate of 0.01 is used as the optimizer.
- The training loop involves a forward pass, loss calculation, backpropagation (`loss.backward()`), and parameter updates (`optimizer.step()`).

### 4. Model Optimization and Evaluation
- To prevent overfitting, **L2 regularization** (weight decay) is introduced by setting the `weight_decay` parameter in the SGD optimizer.
- The model is retrained for 1000 epochs with the regularized optimizer.
- The optimized model achieves a training accuracy of **73.06%** and a test accuracy of **73.58%**.

### 5. Visualization and Interpretation
- The model's performance on the test set is visualized and interpreted using:
  - **Confusion Matrix**: To understand the counts of true positives, true negatives, false positives, and false negatives.
  - **ROC Curve**: To illustrate the trade-off between the true positive rate and false positive rate. The model achieves an **AUC of 0.80**.
  - **Classification Report**: To evaluate precision, recall, and F1-score for each class.

![Confusion Matrix](https'//i.imgur.com/gK9tY9g.png')
![ROC Curve](https'//i.imgur.com/O6P1pA5.png')


### 6. Model Saving and Loading
- The trained model's state dictionary (weights and biases) is saved to a file (`logistic_regression_model.pth`) using `torch.save`.
- A new model instance is created, and the saved parameters are loaded using `model.load_state_dict()`.
- The loaded model is evaluated on the test data to confirm that its performance is preserved, achieving the same accuracy of **73.58%**.

### 7. Hyperparameter Tuning
- A simple hyperparameter search is performed to find the optimal learning rate.
- The model is trained and evaluated with three different learning rates: `[0.01, 0.05, 0.1]`.
- The best performance is achieved with a learning rate of **0.01**, resulting in a test accuracy of **73.43%**.

### 8. Feature Importance Analysis
- The importance of each feature is determined by extracting the learned weights from the model's linear layer.
- The weights are visualized using a bar plot, showing which in-game statistics have the most significant positive or negative impact on the match outcome.

![Feature Importance](https'//i.imgur.com/7vjI28V.png')

## 🚀 Technologies Used
- Python 3
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib

## 📋 How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
