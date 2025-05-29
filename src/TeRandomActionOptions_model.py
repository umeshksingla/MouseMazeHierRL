"""
"""

import random
import numpy as np

import parameters as p
from BaseModel import BaseModel
from TeOptions_model import TeOptions
from options_pre import all_options_dict


class TeRandomActionOptions(TeOptions):

    def __init__(self, file_suffix='_TeRandomActionOptionsTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0
        self.S = 128

        self.mu = 2
        self.options_dict = all_options_dict    # All sequences available at inner nodes.
        self.l6_options_dict = all_options_dict       # All sequences available at L6 nodes.
        self.episode_state_traj = []
        self.s = p.HOME_NODE

    def sample_option(self):
        if p.LVL_BY_NODE[self.s] == 6:
            options_available = self.l6_options_dict[self.s][self.duration]
        else:
            options_available = self.options_dict[self.s][self.duration]
        return random.choice(options_available)

    def choose_action(self, prev_action):

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        assert self.duration == 0
        self.duration = self.sample_duration()
        self.execute_option(self.sample_option())
        self.duration = 0
        action = None
        return action

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        d = min(d, 9)
        return d


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [{} for _ in range(10)]
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    runids = run(TeRandomActionOptions(), param_sets, base_path, '50000', analyze=False)
    print(runids)
