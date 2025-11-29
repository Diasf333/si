import numpy as np
from typing import Callable
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    SelectPercentile

    Selects a given percentage of features based on their F-values,
    ensuring the number of selected features adheres to the specified percentile.

    Parameters
    ----------
    score_function : callable, default=f_classification
        Variance analysis function that receives a Dataset and returns
        the F scores and p-values for each feature.
    percentile : int, default=40
        Percentile for selecting features (between 0 and 100).

    Attributes
    ----------
    F : np.ndarray of shape (n_features,)
        F scores of features estimated by the score_function.
    p : np.ndarray of shape (n_features,)
        p-values of the F scores estimated by the score_function.
    """

    def __init__(
        self,
        score_function: Callable = f_classification,
        percentile: int = 40,
        **kwargs
    ):
        """
        Initialize the SelectPercentile transformer.

        Parameters
        ----------
        score_function : callable, default=f_classification
            Variance analysis function that receives a Dataset and returns
            the F scores and p-values for each feature.
        percentile : int, default=40
            Percentile for selecting features (between 0 and 100).
        """
        super().__init__(**kwargs)
        self.score_function = score_function
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Fit the SelectPercentile transformer by computing the F scores and p-values.

        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        self : SelectPercentile
            Returns self.
        """
        self.F, self.p = self.score_function(dataset)
        return self

    def _transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transform the dataset by selecting a given percentage of features
        based on their F-values. The method handles ties at the threshold
        to ensure that the number of selected features matches the specified
        percentile.

        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_selected_features)
            The array with the selected features.
        """
        # number of features
        n_features = dataset.X.shape[1]

        # number of features to select (round to at least 1)
        k = int(np.ceil(self.percentile / 100 * n_features))
        k = max(1, min(k, n_features))

        # compute threshold F-value at (100 - percentile) percentile
        # e.g., percentile=40 -> we want top 40% -> threshold at 60th percentile
        cutoff = np.percentile(self.F, 100 - self.percentile)

        # initial mask: features strictly greater than cutoff
        mask = self.F > cutoff
        selected_idx = np.where(mask)[0]

        # if we selected too few, we need to include features equal to cutoff until we reach k
        if selected_idx.size < k:
            # indices where F == cutoff
            ties_idx = np.where(self.F == cutoff)[0]
            # how many more features we need
            remaining = k - selected_idx.size
            # take the first `remaining` tied features (sorted for determinism)
            ties_idx = np.sort(ties_idx)[:remaining]
            selected_idx = np.concatenate([selected_idx, ties_idx])

        # if we selected more than k (can happen if many F > cutoff due to rounding),
        # keep only the top k by F-value
        if selected_idx.size > k:
            # sort selected indices by F descending and keep first k
            order = np.argsort(self.F[selected_idx])[::-1]
            selected_idx = selected_idx[order[:k]]

        # finally, select columns
        X_new = dataset.X[:, selected_idx]
        return X_new
