"""Module that implements a perceptron in it's simplest form, as specified in:

Aldewereld, H., van der Bijl, B., Bunk, J., van Moergestel, L. 2022. Machine Learning (reader).
Utrecht: Hogeschool Utrecht."""


from typing import Iterable, Union

from ml.src.linalg import vectops
from ml.src.perceptron import abstract


class Perceptron(abstract.InOutPutNetworkI):
    """A perceptron.

    Attributes:
        _weights:
            (ordered) array that contains the weights corresponding to each input/perceptron in the previous layer.
        _bias: perceptron's bias."""

    def __init__(self,
                 weights: Iterable[Union[float, int]],
                 bias: float):
        """Initialize instance with _weights and _bias."""
        self._weights: Iterable[Union[float, int]] = weights
        self._bias: float = bias

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> int:
        """Compute the perceptron's output based on an array of inputs, corresponding to the ordering of weights
        as established in the _weights attribute.

        Args:
            inputs: the inputs from the previous layer's output, as ordered in _weights.

        Returns:
            The perceptron's output."""
        return int(vectops.dot(self._weights, inputs) + self._bias >= 0)
        # According to equation in figure 2.3 in reader.
        # Neuron activation function proposed in figure 2.4 is computationally expensive because of O(n) copy operation?
        # That is performed when attempting to insert element at index 0 in numpy array.
        #                                                             ^ FIXME(m-jeu): No longer relevant.
        # If change to equation in 2.4 is desired, first element in weights should be 'bias weight'
        # ith 1 added as the first element of every inputs array.

    def __str__(self) -> str:
        return f"Perceptron: b: {self._bias}. w: {self._weights} )"

    def __repr__(self) -> str:
        return self.__str__()

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this perceptron expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return len(self._weights)
