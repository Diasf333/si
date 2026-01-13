import unittest
import numpy as np

from si.neural_networks.losses import CategoricalCrossEntropy


class TestCategoricalCrossEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_fn = CategoricalCrossEntropy()

        # simple 3‑class, 4-sample toy example
        self.y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float)

        self.y_pred_good = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.1, 0.85],
            [0.7, 0.2, 0.1],
        ], dtype=float)

        self.y_pred_bad = np.array([
            [0.05, 0.9, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.85, 0.05],
            [0.2, 0.7, 0.1],
        ], dtype=float)

    def test_loss_shape_and_type(self) -> None:
        loss_value = self.loss_fn.loss(self.y_true, self.y_pred_good)
        self.assertIsInstance(loss_value, float)

    def test_loss_better_predictions_have_lower_loss(self) -> None:
        loss_good = self.loss_fn.loss(self.y_true, self.y_pred_good)
        loss_bad = self.loss_fn.loss(self.y_true, self.y_pred_bad)
        self.assertLess(loss_good, loss_bad)

    def test_loss_non_negative(self) -> None:
        loss_value = self.loss_fn.loss(self.y_true, self.y_pred_good)
        self.assertGreaterEqual(loss_value, 0.0)

    def test_derivative_shape(self) -> None:
        grad = self.loss_fn.derivative(self.y_true, self.y_pred_good)
        self.assertEqual(grad.shape, self.y_true.shape)

    def test_derivative_finite_values(self) -> None:
        grad = self.loss_fn.derivative(self.y_true, self.y_pred_good)
        self.assertTrue(np.all(np.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()
