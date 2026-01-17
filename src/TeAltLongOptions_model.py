"""
Temporally-Extended Greedy Model class with uniform distribution over action length durations (but only LONG durations >= 5)
 built on Alternative Options.
"""

import numpy as np

from TeAltOptions_model import TeAltOptions


class TeLongAltOptions(TeAltOptions):

    def __init__(self, file_suffix='_TeLongAltOptionsTrajectories'):
        TeAltOptions.__init__(self, file_suffix=file_suffix)

    def sample_duration(self):
        d = np.random.randint(5, 9)
        return d


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [{} for _ in range(10)]
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    runids = run(TeLongAltOptions(), param_sets, base_path, '50000', analyze=False)
    print(runids)
