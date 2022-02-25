from typing import Iterable, Union


def mse(it1: Iterable[Union[int, float]], it2: Iterable[Union[int, float]]) -> float:
    # FIXME(m-jeu): Input length verification???
    return sum(map(lambda two_tup: (two_tup[0] - two_tup[1]) ** 2, zip(it1, it2))) / len(it1)
