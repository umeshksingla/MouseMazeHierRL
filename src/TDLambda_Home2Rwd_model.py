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

class TDLambda_Home2Rwd():
    def __init__(self, file_suffix='BaseModel'):
        self.S = 129  # Number of states, including WaterPortState
        self.A = 3    # Number of max actions for a state
        self.S0 = HOME_NODE
        self.file_suffix = file_suffix
        self.nodemap = self.get_SAnodemap()
        self.terminal_nodes = {HOME_NODE, WATER_PORT_STATE}

    def extract_trajectory_data(self):
        """
        Extracts the required trajectory data and pickle-dumps on the disk.
        """
        raise NotImplementedError(
            "You need to define your own data extract function. "
            "Base model doesn't have any.")

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
        for node in np.arange(self.S-1):
            # Shallow level node available from current node
            if node%2 == 0:
                SAnodemap[node,0] = (node - 2) / 2
            elif node%2 == 1:
                SAnodemap[node,0] = (node - 1) / 2
            if SAnodemap[node,0] == INVALID_STATE:
                SAnodemap[node,0] = HOME_NODE

            if node not in NODE_LVL[6]:
                # Deeper level nodes available from current node
                SAnodemap[node,1] = node*2 + 1
                SAnodemap[node,2] = node*2 + 2

        # Nodes available from entry point
        SAnodemap[HOME_NODE,0] = INVALID_STATE
        SAnodemap[HOME_NODE,1] = 0
        SAnodemap[HOME_NODE,2] = INVALID_STATE

        # Nodes at WaterPortState
        SAnodemap[WATER_PORT_STATE, 0] = INVALID_STATE
        SAnodemap[WATER_PORT_STATE, 1] = INVALID_STATE
        SAnodemap[WATER_PORT_STATE, 2] = INVALID_STATE
        SAnodemap[RWD_NODE, 1] = WATER_PORT_STATE
        return SAnodemap

    def softmax(self, s, V, beta):
        '''
        Selects an action (either 0, 1 or 2) to take from current state, s using the current set of state-values and beta parameter.
        :param V: state-values for all nodes, ndarray[S, float]
        :param beta: fixed or fitted inverse temperature parameter, beta
        :return:
        '''

        if s == RWD_NODE or s == HOME_NODE:
            prob = [0, 1, 0]
            a = 1
        elif INVALID_STATE in self.nodemap[s, :]:
            prob = [1, 0, 0]
            a = 0  # If the current state is an end node, a = 0 will make a transition to level 5 node
        else:
            softmaxEXP = []
            next_options = self.nodemap[s, :]
            for node in next_options:
                softmaxEXP.extend([np.exp(beta * V[node])])
            prob = softmaxEXP / np.sum(softmaxEXP)
            try:
                a = np.random.choice([0, 1, 2], 1, p=prob)[0]
            except:
                a = INVALID_STATE
                print('Error with probabilities. softmaxEXP: ', softmaxEXP, ' nodes: ', self.nodemap[s, :],
                      ' state-values: ', V[self.nodemap[s, :]])

        return a, prob

    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, MAX_BOUT_ATTEMPT, V_current, e_current):
        valid_episode = False
        fail_rate = 0

        while not valid_episode and fail_rate < MAX_BOUT_ATTEMPT:
            episode_traj = []
            s = self.S0
            V = V_current
            e = e_current

            while s not in self.terminal_nodes or not episode_traj:
                episode_traj.append(s)  # Record current state

                # Use softmax policy to select action, a at current state, s
                a, prob = self.softmax(s, V, beta)

                # Take action, observe reward and next state
                sprime = self.nodemap[s, a]
                if sprime == WATER_PORT_STATE:
                    R = 1  # Receive a reward of 1 when transitioning to the reward port
                else:
                    R = 0

                # Calculate error signal for current state
                td_error = R + gamma * V[sprime] - V[s]
                e[s] = 1

                # Propagate value to all other states
                for node in np.arange(self.S):
                    V[node] += alpha * td_error * e[node]
                    e[node] = gamma * lamda * e[node]

                # Update future state to current state
                s = sprime

                if len(episode_traj) > MAX_LENGTH:
                    fail_rate += 1
                    valid_episode = False
                    print('Trajectory too long. Aborting episode... Fail rate: ', fail_rate)
                    break
                else:
                    valid_episode = True

            if len(episode_traj) == 2:
                fail_rate += 1
                valid_episode = False
                print('Trajectory of 127 -> 0 -> 127 is too short. Aborting episode... Fail rate: ', fail_rate)
            else:
                episode_traj.append(s)  # Record terminal state which ended the bout

        return episode_traj, V, e, fail_rate

    def simulate(self, sub_fits, MAX_LENGTH=1000, MAX_BOUT_ATTEMPT=10, N_BOUTS_TO_GENERATE=100):
        """
        Simulate the agent with given set of parameters sub_fits.
        """
        V0mag = 0.1        # initial state-value for all nodes
        episodes_all_mice = {}
        V_all_mice = {}

        for mouseID in sub_fits:

            # Initialization for each mouse
            alpha = sub_fits[mouseID][0]    # learning rate
            beta = sub_fits[mouseID][1]     # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]    # discount factor
            lamda = sub_fits[mouseID][3]    # decay parameter for eligibility trace

            print("alpha, beta, gamma, lamda, mouseID, nick",
                  alpha, beta, gamma, lamda, mouseID)

            V = np.ones(self.S) * V0mag
            V[WATER_PORT_STATE] = 0  # setting state-values of terminal nodes to 0
            V[HOME_NODE] = 0
            e = np.zeros(self.S)     # eligibility trace vector for all states
            episodes = []

            for boutID in np.arange(N_BOUTS_TO_GENERATE):
                episode_traj, V_return, e_return, fail_rate = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, MAX_BOUT_ATTEMPT, V, e)
                if fail_rate < MAX_BOUT_ATTEMPT:
                    episodes.append(episode_traj)
                    V = V_return
                    e = e_return

                # Print stats
                if fail_rate:
                    print('Fail rate: ', fail_rate, ' with parameters: ', sub_fits[mouseID])

            episodes_all_mice[mouseID] = episodes
            V_all_mice[mouseID] = V

        return episodes_all_mice, V_all_mice


