"""Module that implements a perceptron in it's simplest form, as specified in:

Aldewereld, H., van der Bijl, B., Bunk, J., van Moergestel, L. 2022. Machine Learning (reader).
Utrecht: Hogeschool Utrecht.

The perceptron learning rule is implemented as specified in:

Single-Layer Neural Networks and Gradient Descent. (2015, 24 march). Dr. Sebastian Raschka. Accessed on 25 february 2022
,from https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html#the-perceptron-learning-rule.
"""


from typing import Iterable, Union, List

import random

from ml.src.stats import spread
from ml.src.linalg import vectops
from ml.src.perceptron import abstract


class Perceptron(abstract.InOutPutNetworkI):
    """A perceptron.

    Attributes:
        _weights:
            (ordered) array that contains the weights corresponding to each input/perceptron in the previous layer.
        _bias: perceptron's bias.
        _learning_rate: the perceptron's learning rate for application in perceptron learning rule."""

    def __init__(self,
                 weights: List[Union[float or int]],
                 bias: float,
                 learning_rate: float = 0):
        """Initialize instance with _weights, _bias and _learning_rate."""
        self._weights: List[Union[float or int]] = weights
        self._bias: float = bias
        self._learning_rate: float = learning_rate

    @classmethod
    def random_instance(cls, weights_amount: int, learning_rate: float = 0):
        """Initialize an instance, with a certain amount of random weights and a bias, uniformly
        distributed between 0 and 1."""
        return Perceptron([random.uniform(-1, 1) for _ in range(weights_amount)],
                          random.uniform(-1, 1),
                          learning_rate=learning_rate)

    def feed_forward(self, inputs: Iterable[Union[float, int]]) -> int:
        """Compute the perceptron's output based on an array of inputs, corresponding to the ordering of weights
        as established in the _weights attribute.

        Args:
            inputs: the inputs from the previous layer's output, as ordered in _weights.

        Returns:
            The perceptron's output."""
        return int(vectops.dot(self._weights, inputs) + self._bias >= 0)
        # According to equation in figure 2.3 in reader.
        # Neuron activation function proposed in figure 2.4 would provide (almost) no performance boost
        # Because no optimized parallelized linear algebra operations are used, but pure python
        # implementations, and I prefer the readability of the current implementation.
        # If change to equation in 2.4 is desired, first element in weights should be 'bias weight'
        # ith 1 added as the first element of every inputs array.

    def _learning_rate_single_weight(self,
                                     error: int,
                                     inp: float = 1,
                                     ) -> float:
        """Apply the perceptron learning rule as specified in the blogpost by Dr. Sebastian Raschka to a single input to
        compute the required delta for the weights associated with that input.

        Args:
            error: the error of the output produced by the perceptron (expected output - actual output).
            inp: the input that was applied to this specific weight. 1 by default, in case of the bias being adjusted.

        Returns:
            the delta (positive or negative) that the weight corresponding to the provided input should be increment by.
        """
        return self._learning_rate * error * inp

    def update(self, inputs: Iterable[Union[float, int]], target: int) -> None:
        """Update the perceptron's weights and biases according to the perceptron learning rule, as
        described by Dr. Sebastian Raschka for a single training example.

        Args:
            inputs: the inputs of the training example.
            target: the expected output for the training example."""
        error: int = target - self.feed_forward(inputs)
        for index, inp in enumerate(inputs):
            self._weights[index] += self._learning_rate_single_weight(error, inp)
        self._bias += self._learning_rate_single_weight(error)

    def loss(self, inputs: Iterable[Iterable[Union[float, int]]], targets: Iterable[int]) -> float:
        """Compute the loss (mean-squared-error, as per the assignment specifications) for the predictions of this
        perceptron for a training dataset.

        Args:
            inputs: the inputs of the training dataset.
            targets: the expected outputs of the training dataset, in the same order as the inputs.

        Returns:
            the loss (mean squared error) of the perceptron's output when compared to the correct outputs."""
        outputs = [self.feed_forward(inp) for inp in inputs]
        return spread.mse(outputs, targets)

    def learn_iterations(self,
                         inputs: Iterable[Iterable[Union[float, int]]],
                         targets: Iterable[int],
                         iterations: int) -> None:
        """Train the perceptron a certain amount of iterations using .update() for every example in a training dataset.

        Args:
            inputs: the inputs of the training dataset.
            targets: the expected outputs of the training dataset, in the same order as the inputs.
            iterations: amount of times the entire training set should be applied to .update()"""
        for _ in range(iterations):
            for inp, target in zip(inputs, targets):
                self.update(inp, target)

    def learn_until_loss(self,
                         inputs: Iterable[Iterable[Union[float, int]]],
                         targets: Iterable[int],
                         loss_target: float,
                         ) -> float:
        """Train the perceptron using .update() on a training dataset until the loss (computed with .loss()) becomes
        lower then a provided threshold.

        WARNING: Will get stuck in an infinite loop when loss_target cannot be reached.

        Args:
            inputs: the inputs of the training dataset.
            targets: the expected outputs of the training dataset, in the same order as the inputs.
            loss_target
                the loss threshold below which the training dataset must score for the perceptron to stop learning.

        Returns:
            the loss (mean squared error) of the perceptron's output when compared to the correct outputs."""
        # Inefficient because examples are applied to perceptron both for updates
        # and computing loss, because this is how assignment specifies implementation should be,
        # and this is the cleanest way to implement it with these specification.
        # Teacher (Mark Pijnenburg) has given his blessing for this inefficient approach.
        while (current_loss := self.loss(inputs, targets)) > loss_target:
            for inp, target in zip(inputs, targets):
                self.update(inp, target)
        return current_loss

    def __str__(self) -> str:
        return f"Perceptron: b: {self._bias}. w: {self._weights} )"

    def __repr__(self) -> str:
        return self.__str__()

    def expected_number_of_inputs(self) -> int:
        """Determine the number of inputs that this perceptron expects to receive in .feed_forward().

        Returns:
            the number of inputs that should be passed to .feed_forward()."""
        return len(self._weights)
