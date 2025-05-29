"""
NOTE: This model will just make the agent stuck.

Epsilon2Greedy model, 2 actions at Level 0 - 5 nodes.
"""
from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy


class Epsilon2Greedy(EpsilonGreedy):

    def __init__(self, file_suffix='_Epsilon2GreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

    def get_valid_actions(self, state):
        """
        Get valid actions available at the "state".
        Note: back_action=False is not verified yet.
        """
        if state == HOME_NODE:
            return [1]
        if state in LVL_6_NODES:
            return [0]
        else:
            return [1, 2]


if __name__ == '__main__':
    pass
