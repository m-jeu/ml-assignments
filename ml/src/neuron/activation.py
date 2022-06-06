"""Module that implements activation functions that cast a value
to a value between 0 and 1, such as the sigmoid function."""


import math
import abc
from typing import Union


# Implemented as abstract class, requiring extending class
# with actual activation functions instead of 1 "ActivationFunction" class
# with instances for every activation function because this approach allows
# having derivatives easily being able to call original function, but only if required.
# Which would be harder and less explicit with 1 singular class with 2 functions as attributes.
class ActivationFunction(metaclass=abc.ABCMeta):
    """An activation function, with a derivative."""

    @abc.abstractmethod
    def func(self, inp: Union[float, int]) -> Union[float, int]:
        """Function itself."""
        pass

    @abc.abstractmethod
    def deriv(self, inp: float) -> float:
        """Function's derivative."""
        pass

    def __call__(self, inp: Union[float, int]) -> Union[float, int]:
        return self.func(inp)


# Sigmoid derivative is more efficient when not written as lambda
class _Sigmoid(ActivationFunction):

    def func(self, inp: float) -> float:
        return 1 / (1 + (math.e ** (-inp)))

    def deriv(self, inp: float) -> float:
        output: float = self(inp)
        return output * (1 - output)


sigmoid: ActivationFunction = _Sigmoid()  # FIXME(m-jeu): Singleton pattern?


class _ReLU(ActivationFunction):
    def func(self, inp: float) -> float:
        return max(0, inp)

    def deriv(self, inp: float) -> float:
        return 0 if inp < 0 else 1


relu: ActivationFunction = _ReLU()
