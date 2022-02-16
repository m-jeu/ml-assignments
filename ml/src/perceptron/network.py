import numpy as np


class PerceptronNetwork:

    def __init__(self, layers: np.ndarray) -> None:
        self._layers = layers

    def feed_forward(self, input: np.ndarray) -> np.ndarray:

        for layer in self._layers:  # FIXME(m-jeu): Get rid of for loop?
            input = layer.feed_forward(input)

        return input
