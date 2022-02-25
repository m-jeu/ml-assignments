"""Module that implements a sigmoid neuron in it's simplest form, as specified in:

Aldewereld, H., van der Bijl, B., Bunk, J., van Moergestel, L. 2022. Machine Learning (reader).
Utrecht: Hogeschool Utrecht.
"""
# FIXME(m-jeu): Unfortunately, I currently have no time to currently implement a shared base class
# With the perceptron in the perceptron package. Obviously, this should be changed at some
# later point.


from typing import Iterable, Union, List

import math

from ml.src.linalg import vectops


def _sigmoid(z: float) -> float:
    """Sigmoid function as specified in (Aldewereld, H. et al.), that proportionally
    casts a value to a proportional value between 0 and 1.

    Args:
        z: sigmoid input.

    Returns:
        sigmoid output."""
    return 1 / (1 + (math.e ** -z))


class SigmoidNeuron:
    """A neuron.

    Attributes:
        _weights:
            (ordered) array that contains the weights corresponding to each input/neuron in the previous layer.
        _bias: neuron's bias."""

    def __init__(self,
                 weights: List[Union[float or int]],
                 bias: float):
        """Initialize instance with _weights, _bias and _learning_rate."""
        self._weights: List[Union[float or int]] = weights
        self._bias: float = bias

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> float:
        """Compute the neuron's output based on an array of inputs, corresponding to the ordering of weights
        as established in the _weights attribute.

        Args:
            inputs: the inputs from the previous layer's output, as ordered in _weights.

        Returns:
            The neuron's output."""
        return _sigmoid(vectops.dot(self._weights, inputs) + self._bias)
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
