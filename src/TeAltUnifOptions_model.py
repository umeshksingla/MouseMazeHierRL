"""
"""

import random
import numpy as np

import parameters as p
from BaseModel import BaseModel
from TeAltOptions_model import TeAltOptions
from options_pre import all_options_dict


class TeAltUnifOptions(TeAltOptions):

    def __init__(self, file_suffix='_TeAltUnifOptionsTrajectories'):
        TeAltOptions.__init__(self, file_suffix=file_suffix)

    def sample_duration(self):
        d = np.random.randint(1, 100)
        d = min(d, 2)
        return d


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [{}]*20
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    runids = run(TeAltUnifOptions(), param_sets, base_path, '50001', analyze=True)
    print(runids)
