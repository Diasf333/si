from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)


class TanhActivation(ActivationLayer):
    """
    TanhActivation

    Applies the hyperbolic tangent activation function element-wise.
    Maps inputs to the range [-1, 1].
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute tanh(x) element-wise.

        Parameters
        ----------
        input_data : np.ndarray
            Input array.

        Returns
        -------
        output : np.ndarray
            tanh(input_data).
        """
        return np.tanh(input_data)

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute derivative of tanh(x).

        Parameters
        ----------
        input_data : np.ndarray
            Input array.

        Returns
        -------
        grad : np.ndarray
            1 - tanh(x)^2.
        """
        t = np.tanh(input_data)
        return 1.0 - t**2



class SoftmaxActivation(ActivationLayer):
    """
    SoftmaxActivation

    Applies the softmax activation function along the last axis,
    converting raw scores into probabilities that sum to 1.
    """

    def activation_function(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute stable softmax.

        Parameters
        ----------
        input_data : np.ndarray
            Input array of shape (n_samples, n_classes).

        Returns
        -------
        output : np.ndarray
            Softmax probabilities with same shape as input.
        """
        # subtract max for numerical stability
        x_shifted = input_data - np.max(input_data, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp

    def derivative(self, input_data: np.ndarray) -> np.ndarray:
        """
        Simplified derivative for softmax when combined with suitable loss
        (e.g., cross-entropy). Here, returns the Jacobian-diagonal form:
        softmax(x) * (1 - softmax(x)).

        Parameters
        ----------
        input_data : np.ndarray
            Input array.

        Returns
        -------
        grad : np.ndarray
            Element-wise softmax derivative approximation.
        """
        s = self.activation_function(input_data)
        return s * (1.0 - s)
