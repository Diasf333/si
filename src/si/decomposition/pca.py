import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """
    PCA

    dimensionality reduction technique that projects the data 
    into a lower-dimensional space by finding directions
    (principal components) that maximize the variance.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.

    Attributes
    ----------
    mean : np.ndarray of shape (n_features,)
        Mean of the samples.
    components : np.ndarray of shape (n_components, n_features)
        Principal components (eigenvectors) sorted by explained variance.
    explained_variance : np.ndarray of shape (n_components,)
        Variance explained by each principal component.
    """

    def __init__(self, n_components: int, **kwargs):
        """
        Initialize the PCA transformer.

        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Fit the PCA model estimating the mean, principal components, and explained variance
        Parameters
        ----------
        dataset : Dataset
            A dataset containing the samples to fit the PCA.

        Returns
        -------
        self : PCA
            Fitted PCA transformer.
        """
        X = dataset.X

        # Center the data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Covariance matrix and eigen decomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues/eigenvectors by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # clip n_components
        n_features = X.shape[1]
        n = min(self.n_components, n_features)

        # Select top n_components
        self.components = eigenvectors[:, :self.n_components].T
        total_var = eigenvalues.sum()
        self.explained_variance = (eigenvalues[:self.n_components] / total_var)

        return self

    def _transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transform the dataset into the reduced space using the principal components.

        Parameters
        ----------
        dataset : Dataset
            A dataset containing the samples to transform.

        Returns
        -------
        X_reduced : np.ndarray of shape (n_samples, n_components)
            The dataset projected onto the principal components.
        """
        X = dataset.X

        # Center data using the mean from fit
        X_centered = X - self.mean

        # Project onto principal components
        X_reduced = np.dot(X_centered, self.components.T)
        return X_reduced
