"""
Use this base class to define your model: from extracting data to loading states
and actions, to simulating agents, etc.
This is supposed to be only provide a skeleton, feel free to override any
function.

For an example, refer to TDLambdaXSteps_model.py file that inherits from
this class. Refer to TDlambda20.ipynb for an example usage.
"""

import abc
import os
import numpy as np
import pickle
from pathlib import Path

from parameters import *
from MM_Traj_Utils import *


class BaseModel:
    def __init__(self, file_suffix='BaseModel'):
        self.S = 129  # Number of states, including WaterPortState
        self.A = 3    # Number of max actions for a state
        self.file_suffix = file_suffix
        self.nodemap = self.get_SAnodemap()

    def extract_trajectory_data(self, orig_data_dir='../outdata/', save_dir=None):
        """
        save_dir: path to the directory where you want to save the pickled
        data object.
        """
        trajectory_data = []
        for mouseId, nickname in enumerate(RewNames):
            trajectory_data.append(self.__get_trajectory_data_by_nickname__(orig_data_dir, nickname))
        if save_dir:
            with open(os.path.join(save_dir, f'{self.file_suffix}.p'), 'wb') as f:
                pickle.dump(trajectory_data, f)
        return trajectory_data

    def __get_trajectory_data_by_nickname__(self, orig_data_dir, nickname):
        """
        Returns ALL trajectory data for a mouse
        """
        print(f"Returning all the trajectories for {nickname}.")
        tf = LoadTrajFromPath(os.path.join(orig_data_dir, nickname + '-tf'))
        trajectory_data = []
        for boutId, bout in enumerate(tf.no):
            trajectory_data.append(bout[:, 0].tolist())
        return trajectory_data

    @staticmethod
    def __load_trajectories__(data):
        # TrajS   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)

        N = len(data)
        B = max([len(n) for n in data])
        BL = max([len(b) for n in data for b in n])

        TrajS = np.ones((N, B, BL)) * InvalidState

        # over mouse
        for n in np.arange(len(data)):
            # over each of their bouts
            for b in np.arange(len(data[n])):
                # over each step of the bout
                for s in np.arange(len(data[n][b])):
                    TrajS[n, b, s] = data[n][b][s]
        return TrajS.astype(int)

    def load_trajectories_from_file(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return self.__load_trajectories__(data)

    def load_trajectories_from_object(self, trajectory_data):
        return self.__load_trajectories__(trajectory_data)

    def load_TrajA(self, TrajS, nodemap):
        # TrajA   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)
        # TrajA   : Matrix entries are action indices (1, 2 or 3) taken to transition from t to t+1 in TrajS
        #           extra space in the matrix is filled with an invalid action, 0.
        #           Action values of 1 is a transition from a deep node, s to shallow node sprime
        #           Action values 2 and 3 are transitions from a shallow node, s to deeper nodes, sprime
        N, B, BL = TrajS.shape
        TrajA = np.zeros((N, B, BL)).astype(int)
        for n in np.arange(N):
            for b in np.arange(B):
                for bl in np.arange(BL - 1):
                    if TrajS[n, b, bl + 1] == InvalidState or TrajS[n, b, bl + 1] == WaterPortNode:
                        break
                    TrajA[n, b, bl] = np.where(
                        nodemap[TrajS[n, b, bl], :] == TrajS[n, b, bl + 1]
                    )[0][0] + 1
        return TrajA

    def get_SAnodemap(self):
        """
        Creates a mapping based on the maze layout where current states are
        linked to the next 3 future states.

        Returns: SAnodemap, a 2D array of current state to future state mappings
                 Also saves SAnodemap in the main_dir as 'nodemap.p'
        Return type: ndarray[(S, A), int]
        """
        SAnodemap = np.ones((self.S, self.A), dtype=int) * INVALID_STATE
        for node in np.arange(self.S - 1):
            # Shallow level node available from current node
            if node % 2 == 0:
                SAnodemap[node, 0] = (node - 2) / 2
            elif node % 2 == 1:
                SAnodemap[node, 0] = (node - 1) / 2
            if SAnodemap[node, 0] == INVALID_STATE:
                SAnodemap[node, 0] = HOME_NODE

            if node not in NODE_LVL[6]:
                # Deeper level nodes available from current node
                SAnodemap[node, 1] = node * 2 + 1
                SAnodemap[node, 2] = node * 2 + 2

        # Nodes available from entry point
        SAnodemap[HOME_NODE, 0] = INVALID_STATE
        SAnodemap[HOME_NODE, 1] = 0
        SAnodemap[HOME_NODE, 2] = INVALID_STATE

        # Nodes at WaterPortState
        SAnodemap[WATER_PORT_STATE, 0] = INVALID_STATE
        SAnodemap[WATER_PORT_STATE, 1] = INVALID_STATE
        SAnodemap[WATER_PORT_STATE, 2] = INVALID_STATE
        SAnodemap[RWD_NODE, 1] = WATER_PORT_STATE
        return SAnodemap

    def simulate(self, sub_fits):
        """
        Simulate the agent with given set of parameters sub_fits.
        """
        raise NotImplementedError(
            "You need to define your own simulate function."
            " Base model doesn't have any.")

