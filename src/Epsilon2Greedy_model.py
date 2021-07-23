"""
Epsilon2Greedy model, with 2 actions at nodes. Contrast it with Epsilon3Greedy.
"""
import numpy as np

from Epsilon3Greedy_model import Epsilon3Greedy


class Epsilon2Greedy(Epsilon3Greedy):

    def __init__(self, file_suffix='_Epsilon2GreedyTrajectories'):
        Epsilon3Greedy.__init__(self, file_suffix=file_suffix)

    def __random_action__(self):
        """
        No action to go back, only left and right
        :return: random action index
        """
        return np.random.choice(range(1, 3))


if __name__ == '__main__':
    pass
