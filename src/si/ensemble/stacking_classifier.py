import numpy as np
from typing import List
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    StackingClassifier

    Ensemble classifier that combines multiple base models whose predictions
    are used as input features to train a final model.

    Parameters
    ----------
    models : list of Model
        Initial set of base models to be stacked.
    final_model : Model
        Model that learns from the predictions of the base models.

    Attributes
    ----------
    models : list of Model
        Fitted base models.
    final_model : Model
        Fitted final model.
    """

    def __init__(self, models: List[Model], final_model: Model) -> None:
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Train the base models and the final model.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.

        Returns
        -------
        self : StackingClassifier
            Fitted stacking classifier.
        """
        # Fit base models on original dataset
        for model in self.models:
            model.fit(dataset)

        # Get base models' predictions on training data
        base_predictions = []
        for model in self.models:
            preds = model.predict(dataset)
            base_predictions.append(preds.reshape(-1, 1))

        # Stack predictions column-wise -> (n_samples, n_models)
        X_meta = np.hstack(base_predictions)

        # Build meta-dataset for the final model
        meta_dataset = Dataset(X_meta, dataset.y)

        # Fit final model on meta-features
        self.final_model.fit(meta_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict labels using the stacked ensemble.

        Parameters
        ----------
        dataset : Dataset
            Dataset to predict.

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels.
        """
        base_predictions = []
        for model in self.models:
            preds = model.predict(dataset)
            base_predictions.append(preds.reshape(-1, 1))

        X_meta = np.hstack(base_predictions)
        meta_dataset = Dataset(X_meta)

        y_pred = self.final_model.predict(meta_dataset)
        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the accuracy between predicted and real labels.

        Parameters
        ----------
        dataset : Dataset
            Test dataset.

        Returns
        -------
        score : float
            Accuracy score of the stacking classifier.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
