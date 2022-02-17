import abc

import numpy as np


class InOutPutNetworkI(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def expected_number_of_inputs(self) -> int:
        pass

    @abc.abstractmethod
    def feed_forward(self, inputs: np.array) -> np.array or float or int:
        pass