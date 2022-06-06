"""Module that implements a sigmoid neuron in it's simplest form, as specified in:

Aldewereld, H., van der Bijl, B., Bunk, J., van Moergestel, L. 2022. Machine Learning (reader).
Utrecht: Hogeschool Utrecht.
"""
from __future__ import annotations

from typing import Iterable, Union, List, TYPE_CHECKING, Tuple

import abc
import random
import math

from ml.src.neuron import activation
from ml.src.linalg import vectops

if TYPE_CHECKING:  # To prevent circle imports :(
    from ml.src.neuron import layer


class BaseNeuron(metaclass=abc.ABCMeta):
    """A neuron.

    Attributes:
        weights:
            (ordered) array that contains the weights corresponding to each input/neuron in the previous layer.
        _bias: neuron's bias.
        _activation_function:
            function that casts weighted sum of input vector and weight vector together with bias to a value between 0
            and 1, such as the sigmoid function or ReLu.
        _learning_rate: the neuron's learning rate.
        position_in_layer:
            the Neuron's position within a NeuronLayer. Is given an actual value when the Neuron
            is added to a NeuronLayer.
        _last_input:
            the last input received in feed_forward() by this Neuron, used for backpropagation. This value is equal
            across all Neurons in a NeuronLayer, so it could be saved there, but considering only a pointer to a list
            is saved, this makes the code cleaner and doesn't cost too much memory.
        _last_output: the last output this Neuron provided in feed_forward(), used for backpropagation.
        _last_output_before_activation:
            the last output ((weighted sum between inputs and weights) + bias) passed forward during feed_forward() by
            this neuron, before it was passed to this Neuron's _activation_function. Used for back propagation.
        last_error: the error most recently computed by _output error.
        staged_deltas:
            A tuple containing the deltas for both the bias (at index 0) and the weights (at index 1) that will be
            applied when update() is called."""

    def __init__(self,
                 weights: List[Union[float or int]],
                 bias: float,
                 learning_rate: float,
                 activation_function: activation.ActivationFunction):
        """Initialize instance with actual values for weights, _bias, _activation_function, _learning_rate,
        and placeholder values for position_in_layer, _last_input, _last_output, _last_output_before_activation,
        last_error, staged_deltas."""
        self.weights: List[Union[float or int]] = weights
        self._bias: float = bias
        self._activation_function: activation.ActivationFunction = activation_function
        self._learning_rate: float = learning_rate

        self.position_in_layer: Union[int, None] = None  # Initialized when added to layer:
        #                                                  useful for computing error of Neuron in hidden layer.

        self._last_input: List[Union[float, int]] = [0 for _ in range(self.expected_number_of_inputs())]
        self._last_output: float = 0
        self._last_output_before_activation = 0
        self.last_error: float = 0

        self.staged_deltas: Tuple[float, Iterable[float]] = (0, tuple())

    # Xavier initialization is specifically scaled to sigmoid activation function, which would be more effective with
    # a different formula/constants for (for instance) ReLu. We are not expected to implement other activation function
    # for P4 however, so I won't bother with taking that into account.
    @classmethod
    def random_instance(cls,
                        amount_of_weights: int,
                        learning_rate: float,
                        activation_function: activation.ActivationFunction,
                        xavier_initialization_n: int = 1):
        """Initialize an instance, with a certain amount of random weights and a bias, uniformly
        distributed between 0 and 1 (if xavier initialization is not used).

        If xavier initialization is used, weights (but not bias) will be scaled appropriately to the amount of weights
        this neuron has.

        Args:
            amount_of_weights: amount of (input) weights the Neuron has.
            learning_rate: learning rate.
            activation_function: activation function.
            xavier_initialization_n: amount of inputs that xavier initialization should scale the weights by."""
        return cls(
            weights=[random.uniform(-1, 1) * math.sqrt(1 / xavier_initialization_n)
                     for _ in range(amount_of_weights)],
            bias=random.uniform(-1, 1),
            learning_rate=learning_rate,
            activation_function=activation_function
        )

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> float:
        """Compute the neuron's output based on an array of inputs, corresponding to the ordering of weights
        as established in the weights attribute.

        Also caches function input, output and intermediate output in ._last_input, ._last_output and
        _last_output_before_activation..

        Args:
            inputs: the inputs from the previous layer's output, as ordered in weights.

        Returns:
            The neuron's output."""
        self._last_input = inputs
        self._last_output_before_activation = vectops.dot(self.weights, inputs) + self._bias
        self._last_output = self._activation_function(self._last_output_before_activation)
        return self._last_output
        # According to equation in figure 2.3 in reader.
        # Neuron activation function proposed in figure 2.4 would provide (almost) no performance boost
        # Because no optimized parallelized linear algebra operations are used, but pure python
        # implementations, and I prefer the readability of the current implementation.
        # If change to equation in 2.4 is desired, first element in weights should be 'bias weight'
        # ith 1 added as the first element of every inputs array.

    def __str__(self) -> str:
        return f"Neuron: b: {self._bias}. w: {self.weights} )"

    def __repr__(self) -> str:
        return self.__str__()

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this neuron expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return len(self.weights)

    @abc.abstractmethod
    def _output_error(self, target: ...) -> None:
        """Determine the error for a neuron, and save to self.error.

        Args:
            target:
                the 'target' for this layer, which would be an actual array of target values for an output Neuron,
                and a List of neurons in the next layer for a hidden neuron."""
        pass

    def _gradient(self, relevant_input: Union[int, float]) -> float:
        """Compute the gradient for one weight according to the last error of this Neuron, and the input
        corresponding to the earlier weight.

        Args:
            relevant_input:
                whatever this Neuron received as input for the weight that the gradient is being determined for.

        Returns:
            The gradient for the specified weight."""
        return relevant_input * self.last_error

    def _delta_weight(self, relevant_input: Union[int, float]) -> float:
        """Compute the delta for a weight (that should be subtracted from that weight) according to the learning
        rate and gradient for that weight.

        Args:
            relevant_input:
                whatever this Neuron received as input for the weight that the delta is being determined for.

        Returns:
            The delta for the weight."""
        return self._learning_rate * self._gradient(relevant_input)

    def _delta_bias(self) -> float:
        """Compute the delta for the bias (that should be subtracted) according to the learning rate and last error.

        Returns:
            The delta for the bias."""
        return self._learning_rate * self.last_error

    def __getitem__(self, item) -> Union[float, int]:
        """Convenience method to directly acces the weights/bias."""
        return self._bias if item == "b" else self.weights[item]

    def stage_deltas(self, target: ...) -> None:
        """Stage deltas for both the weights and bias through stochastic gradient descent to be applied
        after a single backpropagation iteration, by using _output_error(), _delta_bias() and _delta_weights,
        and save to staged_deltas.

        Args:
            target:
                the 'target' for this layer, which would be an actual array of target values for an output Neuron,
                and a List of neurons in the next layer for a hidden neuron."""
        self._output_error(target)
        self.staged_deltas = (
            self._delta_bias(),
            list(map(lambda inp: self._delta_weight(inp), self._last_input))
        )

    def update(self) -> None:
        """Apply the deltas staged in staged_deltas to both the weights and bias."""
        self._bias -= self.staged_deltas[0]
        for i, delta in enumerate(self.staged_deltas[1]):
            self.weights[i] -= delta


class OutputNeuron(BaseNeuron):

    def _output_error(self, target: List[Union[float, int]]) -> None:
        """Determine the error for an output neuron, and save to self.error.

        Args:
            target:
                The one-hot encoded target variable, corresponding to the input last passed to the network
                (and therefore also this Neuron.)"""
        relevant_target = target[self.position_in_layer]
        self.last_error = self._activation_function.deriv(self._last_output_before_activation) * \
                          -(relevant_target - self._last_output)


class HiddenNeuron(BaseNeuron):

    def _output_error(self, target: layer.NeuronLayer) -> None:
        """Determine the error for a hidden Neuron, and save to self.error.

        Args:
            target:
                the next layer in the network."""
        self.last_error = self._activation_function.deriv(
            self._last_output_before_activation
        ) * vectops.dot(
            map(lambda nrn: nrn.last_error, target.neurons),
            map(lambda nrn: nrn.weights[self.position_in_layer], target.neurons)
        )
