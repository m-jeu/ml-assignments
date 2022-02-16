import numpy as np


class PerceptronLayer:

    def __init__(self, perceptrons: np.ndarray) -> None:
        self._perceptrons: np.ndarray = perceptrons

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda p: p.activate(inputs))(self._perceptrons)
