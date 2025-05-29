"""
AA model (some e and AA action all the way - with some variations)
"""

import numpy as np
import parameters as p
from old_ez_models.EpsilonZGreedy_model import EpsilonZGreedy


class AA(EpsilonZGreedy):
    def __init__(self, file_suffix='_AATrajectories'):
        EpsilonZGreedy.__init__(self, file_suffix=file_suffix)

    def choose_action(self, Q, epsilon, *args, **kwargs):

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1, 1.0

        if (self.s in p.LVL_0_NODES) or (self.s in p.LVL_1_NODES):# or (self.s in p.LVL_2_NODES) or (self.s in p.LVL_3_NODES):
            action = self.__random_action__(self.s)
            self.duration = 0
            return action, 1.0

        assert self.duration >= 0
        if self.duration == 0 or kwargs['prev_action'] not in self.get_valid_actions(self.s):
            if np.random.random() <= epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__(self.s)
                self.duration -= 1
                print("epsilon", self.s, self.duration, action)
            else:
                action = self.__random_action__(self.s)
                print("random", self.s, self.duration, action)
        else:
            prev_action = kwargs['prev_action']
            if self.enable_alternate_action:
                # Take the same action (assuming ALTERNATING action means straight path) TODO: think more
                action = (3 - prev_action) % 3
                print("previous alt", self.s, self.duration, action)
            else:
                # OR Take the same action (assuming SAME action means straight path)
                action = prev_action
                print("previous same", self.s, self.duration, action)
            self.duration -= 1
        return action, 1.0

    def sample_duration(self):
        return 100


# Driver Code
if __name__ == '__main__':
    from sample_agent import run
    param_sets = {
        1: {"epsilon": 1.0},
    }
    run(AA(), param_sets, '/Users/usingla/mouse-maze/figs', '20000_randomInCentral1.0')