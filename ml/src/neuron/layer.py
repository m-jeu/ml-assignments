"""Module that implements a layer of several neurons."""


from typing import Iterable, Union, List

from ml.src.neuron import BaseNeuron, OutputNeuron, HiddenNeuron


class NeuronLayer:
    """A (dense) layer of neurons.

    Attributes:
        _neurons: all neurons within the layer. In the same ordering as output will be in."""

    def __init__(self, neurons: List[BaseNeuron]) -> None:
        """Initialize instance with _neurons.

        Raises:
            ValueError: If expected number of inputs is not equal for all neurons."""
        # Check whether all neurons expect the same number of inputs.
        # So that more confusing errors later down the line are avoided.
        neur1_n_inputs = neurons[0].expected_number_of_inputs()
        input_equal = map(lambda p: p.expected_number_of_inputs() == neur1_n_inputs, neurons)
        if not all(input_equal):
            raise ValueError(f"Amount of expected inputs for neurons in layer {self} not homogenous.")
        self._neurons: List[BaseNeuron] = neurons

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> Iterable[float]:
        """Apply an input to all neurons in layer, and output their result.

        Args:
            inputs: input to apply to all neurons' .feed_forward().

        Returns:
            all neurons' output, as ordered as neurons were ordered in __init__()."""
        return map(lambda p: p.feed_forward(inputs), self._neurons)

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this layer expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self._neurons[0].expected_number_of_inputs()  # Assumes all neurons are equal.

    def __getitem__(self, item: int) -> BaseNeuron:
        return self._neurons[item]

    def get_errors(self) -> List[float]:
        """For convenience"""
        return [n.last_error for n in self._neurons]
