import pandas as pd
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib

class MLTrainer:
    """
    A class to train, evaluate, and save machine learning models for supervised learning tasks.
    Supports classification and regression models using scikit-learn API.
    """

    def __init__(self, model: BaseEstimator, task_type: str = 'classification'):
        """
        Initializes the trainer with a model and task type.

        Args:
            model (BaseEstimator): An sklearn-compatible model instance (e.g., LogisticRegression).
            task_type (str): Either 'classification' or 'regression'.
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.model = model
        self.task_type = task_type
        self.is_trained = False
        self.metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on the given training data.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the model on test data and computes appropriate metrics.

        Args:
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Test target vector.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)

        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            self.metrics = {'accuracy': accuracy, 'f1_score': f1}
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.metrics = {'mean_squared_error': mse, 'r2_score': r2}

        return self.metrics

    def predict(self, X: pd.DataFrame) -> Any:
        """
        Predict target values for given features.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray or list: Predicted target values.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained model to a file using joblib.

        Args:
            filepath (str): Path to save the model.
        """
        if not self.is_trained:
            raise RuntimeError("Train the model before saving.")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Loads a model from a file.

        Args:
            filepath (str): Path from which to load the model.
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
