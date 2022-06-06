"""Module that implements a network of neurons."""


from typing import Iterable, Union, List

from ml.src.neuron import layer, activation, neuron


class NeuronNetwork:
    """A dense network consisting of one or more layers of neurons.

    Attributes:
        _layers: all layers in the network, ordered from first layer to output layer."""

    def __init__(self, layers: List[layer.NeuronLayer]) -> None:
        """Initialize instance with _layers."""
        self._layers = layers

    @classmethod
    def build_network(cls,
                      layer_layout: List[int],
                      learning_rate: float,
                      activation_function: activation.ActivationFunction,
                      xavier_initialization: bool = False):
        """Convenience method to easily build a network according to a certain architecture (amount of layer and
        neurons in each layer).

        Args:
            layer_layout:
                the layout of the network, where the first int is the number of inputs, and every int following it
                is an amount of neurons in a layer.
            learning_rate: learning rate for all neurons in the network.
            activation_function: activation function for all neurons in the network.
            xavier_initialization: whether xavier initialization should be used."""

        neuron_type = neuron.OutputNeuron
        layers = []
        for i in range(len(layer_layout) - 1, 0, -1):
            layers.insert(0, layer.NeuronLayer.build_layer(
                neuron_type,
                layer_layout[i],
                layer_layout[i - 1],
                learning_rate,
                activation_function,
                xavier_initialization
            ))
            neuron_type = neuron.HiddenNeuron
        return cls(layers)

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

    def _back_propagate(self, target: Iterable[Union[float, int]]) -> None:
        """Apply the backpropagation algorithm to apply deltas to all neurons in the network for the (single) training
        example last passed to _feed_forward(), without actually applying the changes.

        Args:
            target:
                the one-hot encoded target variable corresponding to the inputs last passed through the network with
                feed_forward()."""
        self._layers[-1].stage_deltas(target)
        last_layer = self._layers[-1]
        for lyr in self._layers[-2::-1]:
            lyr.stage_deltas(last_layer)
            last_layer = lyr

    def _update(self) -> None:
        """Apply all staged changes/deltas in the network."""
        for lyr in self._layers:
            lyr.update()

    def _train_once(self, inputs: Iterable[Union[float, int]], targets: Iterable[Union[float, int]]) -> None:
        """Train according to a single training example.

        Args:
            inputs: the inputs for the training example.
            targets:
                the corresponding one-hot-encoded output, with the same number of values as there are output
                neurons in the last layer."""
        self.feed_forward(inputs)
        self._back_propagate(targets)
        self._update()

    def _train_epoch(self,
                     inputs: Iterable[Iterable[Union[float, int]]],
                     targets: Iterable[Iterable[Union[float, int]]]) -> None:
        """Train a single epoch.

        Args:
            inputs: the training examples
            targets:
                the corresponding one-hot-encoded outputs, with the same number of values as there are output
                neurons in the last layer."""
        for inp, target in zip(inputs, targets):
            self._train_once(inp, target)

    def _train_iterations(self,
                          inputs: Iterable[Iterable[Union[float, int]]],
                          targets: Iterable[Iterable[Union[float, int]]],
                          iterations: int) -> None:
        """Train a certain number of iterations.

        Seperate function from train() because I was planning on implementing more ways to train before.

        Args:
            inputs: the training examples
            targets:
                the corresponding one-hot-encoded outputs, with the same number of values as there are output
                neurons in the last layer.
            iterations: amount of times the entire provided dataset should be passed through the network."""
        for _ in range(iterations):
            self._train_epoch(inputs, targets)

    def train(self,
              inputs: Iterable[Iterable[Union[float, int]]],
              targets: Iterable[Iterable[Union[float, int]]],
              *,
              iterations: int = None) -> None:
        """Train the network.

        Refer to _train_iterations() documentation."""
        
        if iterations is not None:
            self._train_iterations(inputs, targets, iterations)
