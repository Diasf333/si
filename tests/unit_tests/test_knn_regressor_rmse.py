from unittest import TestCase
import numpy as np

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_regressor import KNNRegressor
from si.metrics.rmse import rmse


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.cpu = read_csv(
            "../datasets/cpu/cpu.csv",
            sep=",",
            features=True,
            label=True
        )

    def test_fit_and_predict_shapes(self):
        train_ds, test_ds = train_test_split(self.cpu, test_size=0.2, random_state=42)

        model = KNNRegressor(k=3)
        model.fit(train_ds)

        y_pred = model.predict(test_ds)

        # same number of predictions as samples in test
        self.assertEqual(y_pred.shape[0], test_ds.X.shape[0])

    def test_score_rmse_reasonable(self):
        train_ds, test_ds = train_test_split(self.cpu, test_size=0.2, random_state=42)

        model = KNNRegressor(k=3)
        model.fit(train_ds)

        y_pred = model.predict(test_ds)
        error = rmse(test_ds.y, y_pred)

        # RMSE should be finite and non-negative
        self.assertTrue(np.isfinite(error))
        self.assertGreaterEqual(error, 0.0)
