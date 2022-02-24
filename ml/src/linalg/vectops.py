"""Module that implements certain linear algebra vector-specific functions.

Currently, there is no specific vector type implemented. These are instead represented by
any iterable that has some sense of ordering."""


from typing import Iterable, Union


def dot(v1: Iterable[Union[float, int]], v2: Iterable[Union[float, int]]) -> Union[float, int]:
    """Compute the dot-product for 2 vectors.

    Args:
        v1: vector 1.
        v2: vector 2.

    Returns:
        dot product for the 2 vectors."""
    return sum(map(lambda two_tup: two_tup[0] * two_tup[1], zip(v1, v2)))  # FIXME(m-jeu): Less ugly tuple unpacking?
