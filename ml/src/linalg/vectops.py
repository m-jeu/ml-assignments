"""Module that implements certain linear algebra vector-specific functions.

Currently, there is no specific vector type implemented. These are instead represented by
any iterable that has some sense of ordering."""


from typing import Iterable, Union, Any, List


def dot(v1: Iterable[Union[float, int]], v2: Iterable[Union[float, int]]) -> Union[float, int]:
    """Compute the dot-product for 2 vectors.

    Args:
        v1: vector 1.
        v2: vector 2.

    Returns:
        dot product for the 2 vectors."""
    return sum(map(lambda two_tup: two_tup[0] * two_tup[1], zip(v1, v2)))


def element_equal(v1: Iterable[Any], v2: Iterable[Any]) -> bool:
    """Check whether all elements in 2 iterables of the same length are exactly equal.

    Args:
        v1: iterable 1.
        v2: iterable 2.

    Returns:
        Whether all elements are exactly equal in both iterables."""
    return all(a == b for a, b in zip(v1, v2))


def elements_almost_equal(v1: Iterable[Union[float, int]],
                          v2: Iterable[Union[float, int]],
                          delta: float = 0.00001) -> bool:
    """Check whether all elements in 2 iterables of floats or ints are almost equal, within a given delta..

    Args:
        v1: iterable 1.
        v2: iterable 2.
        delta: permitted different between corresponding elements.

    Returns:
        Whether all elements are almost equal in both iterables."""
    return all(abs(a - b) < delta for a, b in zip(v1, v2))


def accuracy(predict: List[Iterable[int]],
             actual: List[Iterable[int]]) -> float:
    """Compute the accuracy between 2 (equal size) lists of iterables, representing one-hot-encoded prediction from a
    model, and their corresponding values.

    Args:
        predict: list of one-hot-encoded predictions.
        actual: list of actual values.

    Returns:
        Accuracy between 0 and 1."""
    where_same = []
    for it1, it2 in zip(predict, actual):
        where_same.append(all(elem1 == elem2 for elem1, elem2 in zip(it1, it2)))
    return sum(where_same) / len(predict)


def round_iterable(v: Iterable[float]) -> List[int]:
    """Round all the element in an iterable of floats, and return as list.

    Args:
        v: iterable of floats.

    Returns:
        List of rounded floats."""
    return [round(elem) for elem in v]
