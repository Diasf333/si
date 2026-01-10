from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

    
class Adam(Optimizer):
    """
    Adam optimizer.

    Combines momentum and adaptive learning rates for each parameter
    using moving averages of the first and second moments of the gradients.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size (alpha).
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default=1e-8
        Small constant to avoid division by zero.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m: np.ndarray | None = None  # first moment
        self.v: np.ndarray | None = None  # second moment
        self.t: int = 0                   # time step

    def update(self, weights: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update the weights using the Adam optimization algorithm.

        Parameters
        ----------
        weights : numpy.ndarray
            Current weights.
        grad : numpy.ndarray
            Gradient of the loss with respect to the weights.

        Returns
        -------
        numpy.ndarray
            Updated weights.
        """
        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        # Compute bias corrected first and second moment estimates
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # Update parameters
        weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights
