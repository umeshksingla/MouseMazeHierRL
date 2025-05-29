"""
"""

import random
import numpy as np

import parameters as p
from BaseModel import BaseModel
from TeOptions_model import TeOptions
from options_pre import all_options_dict


class TeAltOptions(TeOptions):

    def __init__(self, file_suffix='_TeAltOptionsTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0
        self.S = 128

        self.mu = 2
        self.options_dict = None    # Inner node Options not explicitly constructed for alternating case
        self.l6_options_dict = all_options_dict       # All sequences available at L6 nodes.

        self.episode_state_traj = []
        self.s = p.HOME_NODE

    def sample_option(self):
        assert self.s in p.LVL_6_NODES
        options_available = self.l6_options_dict[self.s][self.duration]
        return random.choice(options_available)

    def choose_action(self, prev_action):

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        assert self.duration >= 0
        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):
            self.duration = self.sample_duration()
            if self.s in p.LVL_6_NODES:
                self.execute_option(self.sample_option())
                self.duration = 1
                action = None
            else:
                action = self.__random_action__(self.s)
        else:
            action = (3 - prev_action) % 3    # alternating
        self.duration -= 1
        return action

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        d = min(d, 9)
        return d


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [{} for _ in range(10)]
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    runids = run(TeAltOptions(), param_sets, base_path, '50000', analyze=False)
    print(runids)
