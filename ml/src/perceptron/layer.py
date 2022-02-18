"""Module that implements a layer of several perceptrons."""


import numpy as np

from ml.src.perceptron import abstract


class PerceptronLayer(abstract.InOutPutNetworkI):
    """A (dense) layer of Perceptrons.

    Attributes:
        _perceptrons: all perceptrons within the layer. In the same ordering as output will be in."""

    def __init__(self, perceptrons: np.ndarray) -> None:
        """Initialize instance with _perceptrons.

        Raises:
            ValueError: If expected number of inputs is not equal for all perceptrons."""
        # Check whether all perceptrons expect the same number of inputs.
        # So that more confusing errors later down the line are avoided.
        all_input_ns = np.vectorize(lambda p: p.expected_number_of_inputs())(perceptrons)
        if not np.all(perceptrons[0].expected_number_of_inputs() == all_input_ns):
            raise ValueError(f"Amount of expected inputs for perceptrons in layer {self} not homogenous.")
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

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self._perceptrons[0].expected_number_of_inputs()  # Assumes all perceptrons are equal.
