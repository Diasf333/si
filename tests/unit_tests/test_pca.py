from unittest import TestCase
import numpy as np

from si.data.dataset import Dataset
from si.decomposition.pca import PCA
import os
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv


class TestPCA(TestCase):

    def setUp(self):
        # simple 2D dataset
        X = np.array([
            [1.0, 1.1],
            [2.0, 1.9],
            [3.0, 3.1],
            [4.0, 3.9],
            [5.0, 5.2],
        ])
        self.dataset = Dataset(X=X, y=None)

    def test_fit_sets_attributes(self):
        pca = PCA(n_components=1)
        pca.fit(self.dataset)

        # shapes
        self.assertEqual(pca.mean.shape, (self.dataset.X.shape[1],))
        self.assertEqual(pca.components.shape, (1, self.dataset.X.shape[1]))
        self.assertEqual(pca.explained_variance.shape, (1,))

        # explained variance sensible
        self.assertTrue(np.all(pca.explained_variance >= 0))
        self.assertLessEqual(float(pca.explained_variance.sum()), 1.0)

    def test_transform_reduces_dimension(self):
        pca = PCA(n_components=1)
        pca.fit(self.dataset)
        X_reduced = pca.transform(self.dataset)

        # same n_samples, reduced n_features
        self.assertEqual(X_reduced.shape, (self.dataset.X.shape[0], 1))

    def test_transform_uses_fitted_mean(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)

        # manual transform should match transform() output
        X_centered = self.dataset.X - pca.mean
        X_manual = X_centered @ pca.components.T
        X_reduced = pca.transform(self.dataset)

        np.testing.assert_array_almost_equal(X_reduced, X_manual)






class TestPCA(TestCase):
    def setUp(self) -> None:
        iris_path = os.path.join(DATASETS_PATH, "iris", "iris.csv")
        self.dataset = read_csv(
            filename=iris_path,
            sep=",",
            features=True,
            label=True
        )

    def test_pca_reduction_and_explained_variance(self) -> None:
        # 2) Fit PCA with 2 components
        pca = PCA(n_components=2)
        pca.fit(self.dataset)

        # 3) Transform dataset
        X_reduced = pca.transform(self.dataset)

        # shapes
        self.assertEqual(self.dataset.X.shape[0], X_reduced.shape[0])
        self.assertEqual(X_reduced.shape[1], 2)

        # explained_variance has length 2
        self.assertEqual(pca.explained_variance.shape[0], 2)

        # values between 0 and 1 and sum <= 1 due to normalization
        self.assertTrue(np.all(pca.explained_variance >= 0.0))
        self.assertTrue(np.all(pca.explained_variance <= 1.0))
        self.assertLessEqual(float(np.sum(pca.explained_variance)), 1.0000001)

