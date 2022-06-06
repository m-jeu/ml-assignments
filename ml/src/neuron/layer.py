"""Module that implements a layer of several neurons."""
from __future__ import annotations

from typing import Iterable, Union, List

from ml.src.neuron import BaseNeuron, OutputNeuron, HiddenNeuron  # FIXME(m-jeu): refactor
from ml.src.neuron import activation


class NeuronLayer:
    """A (dense) layer of neurons.

    Attributes:
        neurons: all neurons within the layer. In the same ordering as output will be in."""

    def __init__(self, neurons: List[BaseNeuron]) -> None:
        """Initialize instance with neurons, verify that all Neuron's match in their amount of weights,
        and set Neuron.position_in_layer for every neuron in Neurons.

        Raises:
            ValueError: If expected number of inputs is not equal for all neurons."""
        # Check whether all neurons expect the same number of inputs.
        # So that more confusing errors later down the line are avoided.
        neur1_n_inputs = neurons[0].expected_number_of_inputs()
        input_equal = map(lambda p: p.expected_number_of_inputs() == neur1_n_inputs, neurons)
        if not all(input_equal):
            raise ValueError(f"Amount of expected inputs for neurons in layer {self} not homogenous.")
        self.neurons: List[BaseNeuron] = neurons

        # Set position in layer for all Neurons
        for i, neur in enumerate(neurons):
            neur.position_in_layer = i

    @classmethod
    def build_layer(cls,
                    neuron_type: type,
                    neuron_amount: int,
                    inputs_amount: int,
                    learning_rate: float,
                    activation_function: activation.ActivationFunction,
                    xavier_initialization: bool = False):
        """Build a layer with randomly initialized Neurons.

        Args:
            neuron_type: the type of neuron that should be used (hidden or output).
            neuron_amount: amount of neurons that should be in the layer.
            inputs_amount: amount of inputs the neurons should expect.
            learning_rate: learning rate for all neurons.
            activation_function: activation function for all neurons.
            xavier_initialization: Whether Xavier initialization should be used."""
        return cls(
            [neuron_type.random_instance(
                inputs_amount,
                learning_rate,
                activation_function,
                inputs_amount if xavier_initialization else 1
            ) for _ in range(neuron_amount)]
        )

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> List[float]:
        """Apply an input to all neurons in layer, and output their result.

        Args:
            inputs: input to apply to all neurons' .feed_forward().

        Returns:
            all neurons' output, as ordered as neurons were ordered in __init__()."""
        return list(map(lambda p: p.feed_forward(inputs), self.neurons))

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this layer expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return self.neurons[0].expected_number_of_inputs()  # Assumes all neurons are equal.

    def __getitem__(self, item: int) -> BaseNeuron:
        """Convenience function to directly fetch Neurons."""
        return self.neurons[item]

    def stage_deltas(self, target: Union[NeuronLayer, Iterable[Union[float, int]]]) -> None:
        """Stage the deltas (without applying them) for all neurons in layer through stochastic gradient descent.

        Args:
            target:
                the target for the layer. If neuron type in layer is output neuron, then target would be the
                one-hot-encoded values of the inputs last passed through the network. If neuron type if hidden neuron,
                this would be the next layer in the network."""
        for nrn in self.neurons:
            nrn.stage_deltas(target)

    def update(self) -> None:
        """"Apply staged deltas to all Neurons in layer."""
        for i, nrn in enumerate(self.neurons):
            nrn.update()
