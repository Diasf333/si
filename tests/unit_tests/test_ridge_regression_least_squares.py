from unittest import TestCase
import numpy as np

from si.data.dataset import Dataset
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.metrics.mse import mse


class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        # simple linear relation with noise: y = 2*x1 - 3*x2 + 1
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = 2 * X[:, 0] - 3 * X[:, 1] + 1 + 0.1 * rng.randn(100)
        self.dataset = Dataset(X=X, y=y)

    def test_fit(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        model.fit(self.dataset)

        # theta and theta_zero should be set with correct shapes
        self.assertIsNotNone(model.theta_zero)
        self.assertIsNotNone(model.theta)
        self.assertEqual(model.theta.shape, (self.dataset.X.shape[1],))

        # mean and std must match number of features
        self.assertEqual(model.mean.shape, (self.dataset.X.shape[1],))
        self.assertEqual(model.std.shape, (self.dataset.X.shape[1],))

    def test_predict(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        model.fit(self.dataset)

        y_pred = model.predict(self.dataset)

        # same number of predictions as samples
        self.assertEqual(y_pred.shape, self.dataset.y.shape)

    def test_score(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        model.fit(self.dataset)

        y_pred = model.predict(self.dataset)
        score = model.score(self.dataset)

        # manual mse should equal model.score
        manual_mse = mse(self.dataset.y, y_pred)
        self.assertAlmostEqual(score, manual_mse)

        # on training data for a well-specified linear model, MSE should be reasonably small
        self.assertLess(score, 0.5)
