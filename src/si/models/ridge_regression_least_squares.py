import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using Least Squares (closed-form solution).
    
    Ridge Regression is a linear regression model with L2 regularization.
    This implementation uses the normal equations to directly compute the coefficients
    without iterative optimization.

    Parameters
    ----------
    l2_penalty : float
        L2 regularization parameter (lambda).
    scale : bool
        Whether to scale the data or not.

    Attributes
    ----------
    theta : np.ndarray
        The coefficients of the model for every feature.
    theta_zero : float
        The zero coefficient (y intercept).
    mean : np.ndarray
        Mean of the dataset for every feature.
    std : np.ndarray
        Standard deviation of the dataset for every feature.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Initialize the Ridge Regression Least Squares model.

        Parameters
        ----------
        l2_penalty : float
            L2 regularization parameter (lambda).
        scale : bool
            Whether to scale the data or not.
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """
        Fits the model to the given dataset using the closed-form solution.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns
        -------
        self : RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X
        y = dataset.y

        # Scale the data if required
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        # Add intercept term to X
        X_aug = np.c_[np.ones(X.shape[0]), X]

        # Compute the penalty term (l2_penalty  * identity matrix)
        penalty = self.l2_penalty * np.eye(X_aug.shape[1])

        # Change the first position of the penalty matrix to 0
        penalty[0, 0] = 0

        # Compute the model parameters using the normal equation
        # theta = (X^T X + penalty)^-1 X^T y
        A = X_aug.T @ X_aug + penalty
        b = X_aug.T @ y
        theta_full = np.linalg.inv(A) @ b

        self.theta_zero = theta_full[0]
        self.theta = theta_full[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable y using the estimated theta coefficients.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        predictions : np.ndarray
            The predicted values.
        """
        X = dataset.X

        # Scale the data if required
        if self.scale:
            X = (X - self.mean) / self.std

        # Add intercept term to X
        X_aug = np.c_[np.ones(X.shape[0]), X]

        # Compute the predicted Y
        theta_full = np.concatenate(([self.theta_zero], self.theta))
        predictions = X_aug @ theta_full

        
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the error between the real and predicted y values.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.
        predictions : np.ndarray
            An array with the predictions.

        Returns
        -------
        error : float
            The MSE between actual values and predictions.
        """
        return mse(dataset.y, predictions)
