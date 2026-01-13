import unittest
import numpy as np

from si.neural_networks.layers import Dropout


class TestDropout(unittest.TestCase):
    def setUp(self) -> None:
        self.probability = 0.5
        self.layer = Dropout(probability=self.probability)
        self.X = np.random.rand(4, 5)  # small random input

    def test_forward_training_mode(self) -> None:
        out = self.layer.forward_propagation(self.X, training=True)

        # shape preserved
        self.assertEqual(out.shape, self.X.shape)

        # some values should be zeroed (with high probability)
        self.assertTrue(np.any(out == 0.0))

        # surviving activations should be scaled by 1/(1-p)
        keep_prob = 1.0 - self.probability
        scale = 1.0 / keep_prob
        # where mask==1, out should equal X * scale
        mask = self.layer.mask
        self.assertIsNotNone(mask)
        self.assertTrue(np.allclose(out[mask == 1], self.X[mask == 1] * scale))

    def test_forward_inference_mode(self) -> None:
        out = self.layer.forward_propagation(self.X, training=False)
        # no change in inference
        self.assertTrue(np.allclose(out, self.X))

    def test_backward(self) -> None:
        # run a training forward pass to set mask
        _ = self.layer.forward_propagation(self.X, training=True)
        output_error = np.ones_like(self.X)
        input_error = self.layer.backward_propagation(output_error)

        # gradient should be zero where mask==0 and 1 where mask==1
        mask = self.layer.mask
        self.assertTrue(np.all((input_error == 0.0) | (input_error == 1.0)))
        self.assertTrue(np.all(input_error[mask == 0] == 0.0))
        self.assertTrue(np.all(input_error[mask == 1] == 1.0))


if __name__ == "__main__":
    unittest.main()
