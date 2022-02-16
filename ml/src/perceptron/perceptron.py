"""Module that implements a perceptron in it's simplest form, as specified in
TODO(m-jeu): Citation to reader."""


import numpy as np


class Perceptron:
    def __init__(self,
                 weights: np.ndarray,
                 bias: float):
        self.weights: np.ndarray = weights
        self.bias: float = bias

    def activate(self, inputs: np.ndarray) -> int:
        return int(np.dot(self.weights, inputs) + self.bias >= 0)  # According to equation in figure 2.3 in reader.
        # Neuron activation function proposed in figure 2.4 is computationally expensive because of O(n) copy operation?
        # That is performed when attempting to insert element at index 0 in numpy array.
        # If change to equation in 2.4 is desired, first element in weights should be 'bias weight'
        # ith 1 added as the first element of every inputs array.

    def __str__(self) -> str:
        return f"( b: {self.bias} )"

    def __repr__(self) -> str:
        return self.__str__()
