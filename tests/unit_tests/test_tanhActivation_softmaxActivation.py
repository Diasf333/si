import unittest
import numpy as np

from si.neural_networks.activation import TanhActivation, SoftmaxActivation


class TestTanhActivation(unittest.TestCase):
    def setUp(self) -> None:
        self.activation = TanhActivation()
        self.X = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    def test_forward(self) -> None:
        out = self.activation.forward_propagation(self.X, training=True)
        self.assertEqual(out.shape, self.X.shape)
        self.assertTrue(np.all(out >= -1.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_backward(self) -> None:
        out = self.activation.forward_propagation(self.X, training=True)
        grad_out = np.ones_like(out)
        grad_in = self.activation.backward_propagation(grad_out)
        self.assertEqual(grad_in.shape, self.X.shape)
        # derivative of tanh should be <= 1 in magnitude
        self.assertTrue(np.all(np.abs(grad_in) <= 1.0))


class TestSoftmaxActivation(unittest.TestCase):
    def setUp(self) -> None:
        self.activation = SoftmaxActivation()
        self.X = np.array([[1.0, 2.0, 3.0],
                           [0.5, 0.1, -1.0]])

    def test_forward(self) -> None:
        out = self.activation.forward_propagation(self.X, training=True)
        self.assertEqual(out.shape, self.X.shape)
        # probabilities between 0 and 1
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))
        # rows sum to 1
        row_sums = np.sum(out, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_backward(self) -> None:
        out = self.activation.forward_propagation(self.X, training=True)
        grad_out = np.ones_like(out)
        grad_in = self.activation.backward_propagation(grad_out)
        self.assertEqual(grad_in.shape, self.X.shape)


if __name__ == "__main__":
    unittest.main()
