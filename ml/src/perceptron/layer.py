import numpy as np

from ml.src.perceptron import abstract


class PerceptronLayer(abstract.InOutPutNetworkI):

    def __init__(self, perceptrons: np.ndarray) -> None:
        self._perceptrons: np.ndarray = perceptrons

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda p: p.feed_forward(inputs))(self._perceptrons)

    def expected_number_of_inputs(self) -> int:
        return self._perceptrons[0].expected_number_of_inputs()  # Assumes all perceptrons are equal.
