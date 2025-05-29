"""
EpsilonGreedy model, 3 actions at Level 0 - 5 nodes.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from utils import break_simulated_traj_into_episodes, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level, calculate_visit_frequency
import evaluation_metrics as em


class EpsilonGreedy(BaseModel):

    def __init__(self, file_suffix='_EpsilonGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.S = 128  # total states
        self.episode_state_traj = []
        self.s = HOME_NODE  # start from s

    def get_action_probabilities(self, state, beta, V):
        raise Exception("wasn't supposed to be called")

    def __greedy_action__(self, state, Q):
        """
        Choose a valid action in state greedily based on Q-values
        :param state:
        :param Q:
        :return: greedy action index based on SA_nodemap
        """
        return self.__random_action__(state)
        action_values = dict([(a_i, round(Q[state, a_i], 10)) for a_i in self.get_valid_actions(state)])
        # print("action_values", action_values, action_values.keys())
        if len(set(list(action_values.values()))) == 1:
            # i.e. if all choices have same value, choose random
            greedy_action = random.choice(list(action_values.keys()))
        else:
            greedy_action = max(action_values.items(), key=lambda x: x[1])[0]
        return greedy_action

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        actions = self.get_valid_actions(state)
        return np.random.choice(actions)

    def choose_action(self, Q, *args, **kwargs):
        if np.random.random() <= self.epsilon:
            action = self.__random_action__(self.s)
        else:
            action = self.__greedy_action__(self.s, Q)
        return action, 1.0

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, _ = self.choose_action(Q)
            s_next = self.take_action(self.s, a)     # Take action
            if np.random.random() <= 0.8:   # make it go home forcibly just to have more data
                if s_next != HOME_NODE and self.s == 0:
                    s_next = HOME_NODE
            self.s = s_next

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        # print(episode_maze_trajs)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def generate_episode(self):
        raise Exception("Nope!")

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):

        print("Simulating agent with id", agentId)
        success = 1
        self.epsilon = params["epsilon"]     # epsilon

        print("epsilon, V, agentId", self.epsilon, agentId)
        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[HOME_NODE, :] = 0
        if self.S == 129:
            Q[RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []

        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH, Q)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)

        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
        }
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


if __name__ == '__main__':
    from sample_agent import run
    param_sets = [
        {"epsilon": 1},
        # 2: {"epsilon": 1},
        # 3: {"epsilon": 1},
        # 4: {"epsilon": 1},
    ]
    run(EpsilonGreedy(), param_sets, '/Users/usingla/mouse-maze/figs', '100000')

