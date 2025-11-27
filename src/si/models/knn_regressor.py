from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor is a machine learning model that predicts the
    target value of new samples based on a similarity measure (e.g., distance functions).
    This algorithm predicts the value of new samples by averaging the target values
    of the k-nearest samples in the training data

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: Dataset
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_prediction(self, sample: np.ndarray) -> float:
        """
        Returns the predicted target value for a single sample.

        Parameters
        ----------
        sample: np.ndarray
            The sample to predict.

        Returns
        -------
        value: float
            The predicted target value.
        """
        # 1. Calculate the distance between the sample and the training samples
        distances = self.distance(sample, self.dataset.X)

        # 2. Obtain the indexes of the k most similar examples (shortest distance)
        k_nearest_indx = np.argsort(distances)[:self.k]

        # 3. Retrieve the corresponding values in y
        k_nearest_values = self.dataset.y[k_nearest_indx]

        # 4. Calculate the average of these values
        return float(np.mean(k_nearest_values))

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the target values for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        # 5. Apply _get_prediction to all samples in the testing dataset
        predictions = np.apply_along_axis(self._get_prediction, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the RMSE of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.
        predictions: np.ndarray
            An array with the predictions.

        Returns
        -------
        error: float
            The RMSE between actual values and predictions.
        """
        return rmse(dataset.y, predictions)