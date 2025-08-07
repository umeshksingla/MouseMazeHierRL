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
        TeOptions.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0
        self.S = 128

        self.options_dict = None    # Inner node Options not explicitly constructed for alternating case
        self.l6_options_dict = all_options_dict       # All sequences available at L6 nodes.

        self.episode_state_traj = []
        self.s = p.HOME_NODE         # p.HOME_NODE
        self.exit = p.HOME_NODE      # p.HOME_NODE

        np.random.seed()

    def sample_option(self):
        assert self.s in p.LVL_6_NODES
        options_available = self.l6_options_dict[self.s][self.duration]
        return random.choice(options_available)

    def choose_action(self, prev_action):

        if self.s == self.exit:
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
        self.mu = self.params['mu']
        d = np.random.zipf(a=self.mu)
        d = min(d, 9)
        return d


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [{'mu': 1.01}, {'mu': 1.25}, {'mu': 1.5}, {'mu': 1.75}, {'mu': 2}, {'mu': 2.25}, {'mu': 2.5}, {'mu': 2.75}, {'mu': 3}]*20
    param_sets = [{**d, 'model': f"TeAltOptions{d['mu']}"} for d in param_sets]
    print(param_sets)
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    runids = run(TeAltOptions(), param_sets, base_path, '50000', analyze=False)
    print(runids)
