import numpy as np
from typing import Optional, List, Tuple
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """
    RandomForestClassifier


    is an ensemble machine learning technique that
    combines multiple decision trees to improve prediction accuracy
    and reduce overfitting.
    """

    def __init__(
        self,
        nestimators: int = 100,
        maxfeatures: Optional[int] = None,
        minsamplesplit: int = 2,
        maxdepth: int = 10,
        mode: str = "gini",
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nestimators = nestimators
        self.maxfeatures = maxfeatures
        self.minsamplesplit = minsamplesplit
        self.maxdepth = maxdepth
        self.mode = mode
        self.seed = seed
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []

    def _fit(self, dataset: Dataset):
        """
        train the decision trees of the random forest
        Parameters
        ----------
        dataset : Dataset
            Dataset object containing features and labels.
        Returns
        -------
        self : RandomForestClassifier
            Fitted estimator.
        """
        # Sets the random seed
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape

        # Defines self.max_features to be int(np.sqrt(n_features)) if None
        if self.maxfeatures is None:
            self.maxfeatures = int(np.sqrt(n_features))
        self.trees = []

        # Repeat steps for all trees in the forest
        for _ in range(self.nestimators):
            # Create a bootstrap dataset
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            feature_indices = np.random.choice(
                n_features, self.maxfeatures, replace=False
            )
            X_sample = dataset.X[sample_indices][:, feature_indices]
            y_sample = dataset.y[sample_indices]
            sampled_dataset = Dataset(X_sample, y_sample)

            # Create and train a decision tree with the bootstrap dataset
            tree = DecisionTreeClassifier(
                minsamplesplit=self.minsamplesplit,
                maxdepth=self.maxdepth,
                mode=self.mode
            )
            tree.fit(sampled_dataset)

            # Append a tuple containing the features used and the trained tree
            self.trees.append((feature_indices, tree))

        # Return itself (self)
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        predicts the labels using the ensemble models
        Parameters
        ----------
        dataset : Dataset
            Dataset with the samples to classify.
        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        predictions = []

        # Get predictions for each tree using the respective set of features
        for feature_indices, tree in self.trees:
            X_subset = dataset.X[:, feature_indices]
            preds = tree.predict(Dataset(X_subset))
            predictions.append(preds)

        # shape: (n_trees, n_samples)
        all_preds = np.vstack(predictions)

        # Majority vote with np.unique (works for string labels too)
        n_samples = all_preds.shape[1]
        y_pred = []

        for j in range(n_samples):
            col = all_preds[:, j]
            values, counts = np.unique(col, return_counts=True)
            majority_label = values[np.argmax(counts)]
            y_pred.append(majority_label)

        return np.array(y_pred)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        computes the accuracy between predicted and real labels
        Parameters
        ----------
        dataset : Dataset
            Dataset containing ground truth labels.
        Returns
        -------
        score : float
            Accuracy score between true and predicted labels.
        """
        # Computes the accuracy between predicted and real values
        return accuracy(dataset.y, predictions)
