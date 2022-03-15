"""Module that implements activation functions that cast a value
to a value between 0 and 1, such as the sigmoid function."""


import math


def sigmoid(z: float) -> float:  # FIXME(m-jeu): Reference in different module.
    """Sigmoid function as specified in (Aldewereld, H. et al.), that proportionally
    casts a value to a proportional value between 0 and 1.

    Args:
        z: sigmoid input.

    Returns:
        sigmoid output."""
    return 1 / (1 + (math.e ** -z))