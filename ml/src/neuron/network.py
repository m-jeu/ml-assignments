"""Module that implements a network of neurons."""


from typing import Iterable, Union

from ml.src.neuron import layer


class SigmoidNeuronNetwork:
    """A dense network consisting of one or more layers of neurons.

    Attributes:
        _layers: all layers in the network, ordered from first layer to output layer."""

    def __init__(self, layers: Iterable[layer.neuronLayer]) -> None:
        """Initialize instance with _layers."""
        self._layers = layers

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> Iterable[float]:
        """Apply an input to the network.

        Applies the input to the first layer, and then passes that layer's output to successive
        layers until end is reached.
        Returns input if there are no layers in the network.

        Args:
            inputs: input to apply to the first layer.

        Returns:
            output of the last layer."""

        for layer in self._layers:
            inputs = layer.feed_forward(inputs)

        return inputs

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that the network expects.

        Based on the expectation of the first layer.

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self._layers[0].expected_number_of_inputs()
