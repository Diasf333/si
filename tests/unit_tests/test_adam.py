import unittest
import numpy as np

from si.neural_networks.optimizers import Adam


class TestAdam(unittest.TestCase):
    def setUp(self) -> None:
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.optimizer = Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon
        )

        # simple parameter vector and gradient
        self.w_init = np.array([1.0, -2.0, 3.0], dtype=float)
        self.grad = np.array([0.1, -0.2, 0.3], dtype=float)

    def test_update_returns_same_shape(self) -> None:
        w_new = self.optimizer.update(self.w_init.copy(), self.grad)
        self.assertEqual(w_new.shape, self.w_init.shape)

    def test_update_changes_weights(self) -> None:
        w_new = self.optimizer.update(self.w_init.copy(), self.grad)
        # weights should be updated (not equal to original)
        self.assertFalse(np.allclose(w_new, self.w_init))

    def test_state_initialized_on_first_update(self) -> None:
        _ = self.optimizer.update(self.w_init.copy(), self.grad)
        # m, v should be initialized and t should be 1
        self.assertIsNotNone(self.optimizer.m)
        self.assertIsNotNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 1)
        self.assertEqual(self.optimizer.m.shape, self.grad.shape)
        self.assertEqual(self.optimizer.v.shape, self.grad.shape)

    def test_state_changes_over_multiple_updates(self) -> None:
        _ = self.optimizer.update(self.w_init.copy(), self.grad)
        m1 = self.optimizer.m.copy()
        v1 = self.optimizer.v.copy()
        t1 = self.optimizer.t

        _ = self.optimizer.update(self.w_init.copy(), self.grad)
        self.assertFalse(np.allclose(self.optimizer.m, m1))
        self.assertFalse(np.allclose(self.optimizer.v, v1))
        self.assertEqual(self.optimizer.t, t1 + 1)

    def test_updates_move_against_gradient(self) -> None:
        # apply several updates to see direction of movement
        w = self.w_init.copy()
        for _ in range(10):
            w = self.optimizer.update(w, self.grad)

        # grad[0] > 0 => parameter should decrease
        self.assertLess(w[0], self.w_init[0])
        # grad[1] < 0 => parameter should increase
        self.assertGreater(w[1], self.w_init[1])
        # grad[2] > 0 => parameter should decrease
        self.assertLess(w[2], self.w_init[2])

    def test_no_nan_or_inf_in_updates(self) -> None:
        w = self.w_init.copy()
        for _ in range(20):
            w = self.optimizer.update(w, self.grad)
            self.assertTrue(np.all(np.isfinite(w)))


if __name__ == "__main__":
    unittest.main()

