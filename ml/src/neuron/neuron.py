"""Module that implements a sigmoid neuron in it's simplest form, as specified in:

Aldewereld, H., van der Bijl, B., Bunk, J., van Moergestel, L. 2022. Machine Learning (reader).
Utrecht: Hogeschool Utrecht.
"""
# FIXME(m-jeu): Unfortunately, I currently have no time to currently implement a shared base class
# With the perceptron in the perceptron package. Obviously, this should be changed at some
# later point.


from typing import Iterable, Union, List, Callable

import math

from ml.src.neuron import activation
from ml.src.linalg import vectops


class SigmoidNeuron:
    """A neuron.

    Attributes:
        _weights:
            (ordered) array that contains the weights corresponding to each input/neuron in the previous layer.
        _bias: neuron's bias.
        _activation_function:
            function that casts weighted sum of input vector and weight vector together with bias to a value between 0
            and 1, such as the sigmoid function of ReLu."""

    def __init__(self,
                 weights: List[Union[float or int]],
                 bias: float,
                 activation_function: Callable[[float], float] = activation.sigmoid):
        """Initialize instance with _weights, _bias, _activation_function."""
        self._weights: List[Union[float or int]] = weights
        self._bias: float = bias
        self._activation_function: Callable[[float], float] = activation_function

        self._last_input: Iterable[Union[float, int]] = [0 for _ in range(self.expected_number_of_inputs())]
        self._last_output: float = 0  # FIXME(m-jeu): Consider making sentinel value?

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> float:
        """Compute the neuron's output based on an array of inputs, corresponding to the ordering of weights
        as established in the _weights attribute.

        Also caches function input and output in ._last_input and ._last_output.

        Args:
            inputs: the inputs from the previous layer's output, as ordered in _weights.

        Returns:
            The neuron's output."""
        self._last_input = inputs
        self._last_output = self._activation_function(vectops.dot(self._weights, inputs) + self._bias)
        return self._last_output
        # According to equation in figure 2.3 in reader.
        # Neuron activation function proposed in figure 2.4 would provide (almost) no performance boost
        # Because no optimized parallelized linear algebra operations are used, but pure python
        # implementations, and I prefer the readability of the current implementation.
        # If change to equation in 2.4 is desired, first element in weights should be 'bias weight'
        # ith 1 added as the first element of every inputs array.

    def __str__(self) -> str:
        return f"Neuron: b: {self._bias}. w: {self._weights} )"

    def __repr__(self) -> str:
        return self.__str__()

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this neuron expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return len(self._weights)

