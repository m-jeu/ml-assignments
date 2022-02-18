"""Module that implements a layer of several perceptrons."""


import numpy as np

from ml.src.perceptron import abstract


class PerceptronLayer(abstract.InOutPutNetworkI):
    """A layer of Perceptrons.

    Attributes:
        _perceptrons: all perceptrons within the layer. In the same ordering as output will be in."""

    def __init__(self, perceptrons: np.ndarray) -> None:
        """Initialize instance with _perceptrons."""
        self._perceptrons: np.ndarray = perceptrons

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply an input to all perceptrons in layer, and output their result.

        Args:
            inputs: input to apply to all perceptrons' .feed_forward().

        Returns:
            all perceptrons' output, as ordered as perceptrons were ordered in __init__()."""
        return np.vectorize(lambda p: p.feed_forward(inputs))(self._perceptrons)

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this layer expects to receive in .feed_forward().

        WARNING: Currently assumes that all perceptrons match the fingerprint of the first perceptron. FIXME(m-jeu).

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self._perceptrons[0].expected_number_of_inputs()  # Assumes all perceptrons are equal.
