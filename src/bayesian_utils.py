import numpy as np
import random


def argmaxrand(a):
    """
    If a has only one max it is equivalent to argmax,
    otherwise it uniformly random selects a maximum
    """

    indices = np.where(np.array(a) == np.max(a))[0]
    return np.random.choice(indices)


def argmaxrand_dict(a: dict):
    if len(set(list(a.values()))) == 1:
        # i.e. if all choices have same value, choose random
        return random.choice(list(a.keys()))
    else:
        return max(a.items(), key=lambda x: x[1])[0]
