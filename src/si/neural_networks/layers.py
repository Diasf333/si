from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    
class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

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
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)

        # computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) 


class Dropout(Layer):
    """
    Dropout

    Dropout layer that randomly sets a fraction of the input units to zero
    during training to prevent overfitting.

    Parameters
    ----------
    probability : float
        Dropout rate between 0 and 1. Fraction of units to drop.

    Attributes
    ----------
    probability : float
        Dropout rate between 0 and 1.
    mask : np.ndarray
        Binomial mask applied to the input during training.
    input : np.ndarray
        Input of the layer.
    output : np.ndarray
        Output of the layer.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("probability must be in [0, 1).")
        self.probability = probability
        self.mask: np.ndarray | None = None
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the dropout layer.

        In training mode, applies a binomial mask and scales the activations
        to keep the expected value unchanged. In inference mode, returns
        the input unchanged.

        Parameters
        ----------
        input_data : np.ndarray
            Input array to the layer.
        training : bool, default=True
            Whether the layer is in training mode or inference mode.

        Returns
        -------
        output : np.ndarray
            Output after applying dropout (or the original input in inference mode).
        """
        self.input = input_data

        if training:
            # Keep probability = 1 - dropout_rate
            keep_prob = 1.0 - self.probability
            if keep_prob <= 0.0:
                raise ValueError("keep probability must be > 0.0.")

            # Scale factor to maintain expected activations
            scale = 1.0 / keep_prob

            # Binomial mask: 1 with prob=keep_prob, 0 with prob=probability
            self.mask = np.random.binomial(1, keep_prob, size=input_data.shape)

            self.output = input_data * self.mask * scale
            return self.output
        else:
            # No dropout in inference
            self.mask = None
            self.output = input_data
            return input_data

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation through the dropout layer.

        Parameters
        ----------
        output_error : np.ndarray
            The gradient of the loss with respect to the output of this layer.

        Returns
        -------
        input_error : np.ndarray
            The gradient of the loss with respect to the input of this layer.
        """
        if self.mask is None:
            # In inference mode, dropout is inactive; gradient passes unchanged.
            return output_error

        input_error = output_error * self.mask
        return input_error

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        shape : tuple
            The shape of the output, same as the input shape.
        """
        if self.input is None:
            raise ValueError("Layer has not been forward propagated yet.")
        return self.input.shape

    def parameters(self) -> int:
        """
        Return the number of learnable parameters of the layer.

        Returns
        -------
        n_parameters : int
            Number of parameters. Dropout has no learnable parameters, so this is 0.
        """
        return 0
