import numpy as np

from ml.src.perceptron import abstract


class PerceptronNetwork(abstract.InOutPutNetworkI):

    def __init__(self, layers: np.ndarray) -> None:
        self._layers = layers

    def feed_forward(self, input: np.ndarray) -> np.ndarray:

        for layer in self._layers:  # FIXME(m-jeu): Get rid of for loop?
            input = layer.feed_forward(input)

        return input

    def expected_number_of_inputs(self) -> int:
        return self._layers[0].expected_number_of_inputs()
