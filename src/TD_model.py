"""
"""

import abc
import os
import numpy as np
import pickle
from pathlib import Path


from parameters import *


class TD:
    def __init__(self, main_dir):
        self.S = 128  # Number of states
        self.A = 3    # Number of max actions for a state
        
        self.main_dir = os.path.abspath(main_dir)
        self.stan_data_dir = os.path.join(self.main_dir, 'pre_reward_traj/real_traj')
        self.stan_results_dir = os.path.join(self.main_dir, 'stan_results')
        
        Path(self.stan_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.stan_results_dir).mkdir(parents=True, exist_ok=True)
        
        self.file_prefix = 'TD'
        
    @abc.abstractmethod
    def get_trajectory_data(self):
        pass

    def get_SAnodemap(self):
        '''
        Creates a mapping based on the maze layout where current states are linked to the next 3 future states

        Returns: SAnodemap, a 2D array of current state to future state mappings
                 Also saves SAnodemap in the main_dir as 'nodemap.p'
        Return type: ndarray[(S, A), int]
        '''
        S = self.S
        A = self.A

        # Return nodemap for state-action values
        SAnodemap = np.ones((S, A), dtype=int) * InvalidState
        for node in np.arange(S-1):
            # Shallow level node available from current node
            if node%2 == 0:
                SAnodemap[node,0] = (node - 2) / 2
            elif node%2 == 1:
                SAnodemap[node,0] = (node - 1) / 2
            if SAnodemap[node,0] == InvalidState:
                SAnodemap[node,0] = HomeNode

            if node not in lv6_nodes:
                # Deeper level nodes available from current node
                SAnodemap[node,1] = node*2 + 1
                SAnodemap[node,2] = node*2 + 2

        # Nodes available from entry point
        SAnodemap[HomeNode,0] = InvalidState
        SAnodemap[HomeNode,1] = 0
        SAnodemap[HomeNode,2] = InvalidState
        
        with open(os.path.join(self.main_dir, 'nodemap.p'),'wb') as f:
            pickle.dump(SAnodemap, f)

        return SAnodemap

