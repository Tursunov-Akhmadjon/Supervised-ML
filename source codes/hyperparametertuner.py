from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Optional, Dict, Any
import pandas as pd

class HyperparameterTuner:
    """
    A class to perform hyperparameter tuning for sklearn models using Grid Search or Randomized Search.

    Attributes:
        model: The sklearn estimator (model) to tune.
        param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        search_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
        cv: Number of cross-validation folds.
        n_iter: Number of parameter settings sampled in RandomizedSearchCV (only for 'random' method).
        refit_metric: Metric used for refitting the best model.
        best_estimator_: Best model found after tuning.
        best_params_: Best hyperparameters found.
    """

    def __init__(self, model, param_grid: Dict[str, list], search_method: str = 'grid', 
                 cv: int = 5, n_iter: Optional[int] = 10, refit_metric: str = 'accuracy'):
        """
        Initializes the HyperparameterTuner.

        Args:
            model: sklearn estimator object.
            param_grid (Dict[str, list]): Parameter grid or distributions to sample from.
            search_method (str): 'grid' or 'random'. Default is 'grid'.
            cv (int): Number of cross-validation folds. Default is 5.
            n_iter (Optional[int]): Number of iterations for RandomizedSearchCV. Ignored if search_method='grid'.
            refit_metric (str): Metric used to refit the best model. Default is 'accuracy'.
        """
        if search_method not in ['grid', 'random']:
            raise ValueError("search_method must be 'grid' or 'random'.")

        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.cv = cv
        self.n_iter = n_iter if n_iter is not None else 10
        self.refit_metric = refit_metric
        self.best_estimator_ = None
        self.best_params_ = None
        self.search_result_ = None

    def tune(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Runs hyperparameter tuning on the data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            Best estimator after tuning.
        """
        if self.search_method == 'grid':
            search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.refit_metric, refit=True)
        else:
            search = RandomizedSearchCV(self.model, self.param_grid, n_iter=self.n_iter, cv=self.cv,
                                        scoring=self.refit_metric, refit=True, random_state=42)

        search.fit(X, y)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.search_result_ = search
        return self.best_estimator_

    def get_best_params(self) -> Dict[str, Any]:
        """
        Returns the best hyperparameters found by the search.

        Returns:
            dict: Best hyperparameters.
        """
        if self.best_params_ is None:
            raise ValueError("No tuning has been done yet. Call tune() first.")
        return self.best_params_

    def get_cv_results(self) -> pd.DataFrame:
        """
        Returns the cross-validation results as a DataFrame.

        Returns:
            pd.DataFrame: CV results.
        """
        if self.search_result_ is None:
            raise ValueError("No tuning has been done yet. Call tune() first.")
        return pd.DataFrame(self.search_result_.cv_results_)
