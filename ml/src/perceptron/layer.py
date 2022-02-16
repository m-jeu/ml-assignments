import numpy as np


class PerceptronLayer:

    def __init__(self, perceptrons: np.ndarray) -> None:
        self._perceptrons: np.ndarray = perceptrons

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda p: p.feed_forward(inputs))(self._perceptrons)
