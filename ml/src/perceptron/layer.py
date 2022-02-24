"""Module that implements a layer of several perceptrons."""


from typing import Iterable, Union, List

from ml.src.perceptron import abstract, perceptron


class PerceptronLayer(abstract.InOutPutNetworkI):
    """A (dense) layer of Perceptrons.

    Attributes:
        _perceptrons: all perceptrons within the layer. In the same ordering as output will be in."""

    def __init__(self, perceptrons: List[perceptron.Perceptron]) -> None:
        """Initialize instance with _perceptrons.

        Raises:
            ValueError: If expected number of inputs is not equal for all perceptrons."""
        # Check whether all perceptrons expect the same number of inputs.
        # So that more confusing errors later down the line are avoided.
        percept1_n_inputs = perceptrons[0]
        input_equal = map(lambda p: p.expected_number_of_inputs == percept1_n_inputs, perceptrons)
        if not all(input_equal):
            raise ValueError(f"Amount of expected inputs for perceptrons in layer {self} not homogenous.")
        self._perceptrons: List[perceptron.Perceptron] = perceptrons

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> Iterable[Union[float, int]]:
        """Apply an input to all perceptrons in layer, and output their result.

        Args:
            inputs: input to apply to all perceptrons' .feed_forward().

        Returns:
            all perceptrons' output, as ordered as perceptrons were ordered in __init__()."""
        return map(lambda p: p.feed_forward(inputs), self._perceptrons)  # FIXME(m-jeu): Refactor to non-generator?

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this layer expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self._perceptrons[0].expected_number_of_inputs()  # Assumes all perceptrons are equal.
