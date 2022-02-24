"""Module that implements an interface that's shared between Perceptrons, PerceptronLayers,
and PerceptronNetworks that improves testing ability."""


import abc

import numpy as np


class InOutPutNetworkI(metaclass=abc.ABCMeta):
    """An interface for a component of a Neural/Perceptron Network that expects a certain number of inputs,
    allows those inputs to be applied to it in some way, and outputs a result based on them."""

    @abc.abstractmethod
    def expected_number_of_inputs(self) -> int:
        """The number of inputs that this component expects in it's .feed_forward().

        Returns:
            number of inputs that this component expects in it's .feed_forward()."""
        pass

    @abc.abstractmethod
    def feed_forward(self, inputs: np.array) -> np.array or float or int:
        """Apply input to the component, and output an output based on it.

        Args:
            inputs: inputs.

        Returns:
            component output. May be an array, or a scalar."""
        pass
