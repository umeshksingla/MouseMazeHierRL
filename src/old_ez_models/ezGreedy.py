"""
EpsilonZGreedy model from Dabney et al 2020.

With the new action definition
    0: going forward straight
    1: going left
    2: going right
    3: going back

"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
from actions import actions_node_matrix
from maze_spatial_mapping import CELL_XY, NODE_CELL_MAPPING


class eZGreedy(BaseModel):

    def __init__(self, file_suffix='_EpsilonZGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0
        self.S = 128    # total states

        self.s = 1
        self.prev_s = 0
        self.prev_prev_s = HOME_NODE
        self.episode_state_traj = [self.prev_s, self.s]

        self.actions_node_matrix = actions_node_matrix

    def __random_action__(self):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        next_action_nodes = self.actions_node_matrix[self.prev_s][self.s]
        actions = [a for a, next_n in enumerate(next_action_nodes) if next_n != INVALID_STATE]
        return np.random.choice(actions)

    @staticmethod
    def level3_path(n, go_up=2):
        path = [n]
        while go_up:
            if n % 2 == 0:
                n -= 1
            n = n // 2
            go_up -= 1
            path.append(n)
        return path[::-1]

    def take_action(self, prev_n: int, n: int, a: int) -> int:
        return int(self.actions_node_matrix[prev_n][n][a])

    def choose_action(self, *args, **kwargs):
        # print("s", self.s, "valid act", self.get_valid_actions(self.s))

        # if self.s == HOME_NODE:
        #     # See https://www.notion.so/umeshksingla/original-ez-greedy-8eeb7d4a49104f52a6ba0a05c60ea6cd
        #     self.duration = 0
        #     return 1, 1.0

        if self.s in LVL_6_NODES:
            self.duration = 0
            return 3, 1.0

        assert self.duration >= 0
        prev_action = kwargs['prev_action']
        next_action_nodes = self.actions_node_matrix[self.prev_s][self.s]
        print(next_action_nodes)

        if self.duration == 0 or next_action_nodes[prev_action] == INVALID_STATE:
            if np.random.random() <= self.epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__()
                print("epsilon", self.s, self.duration, action)
                # self.duration -= 1
            else:
                action = self.__random_action__()
                self.duration = 0
                print("random", self.s, self.duration, action)
        else:
            if self.enable_alternate_action:
                # Take the same action (assuming ALTERNATING action means straight path) TODO: think more
                action = (3 - prev_action) % 3
                print("previous alt", self.s, self.duration, action)
            else:
                # OR Take the same action (assuming SAME action means straight path)

                # if self.s == self.prev_prev_s and (prev_action == 3 or prev_action == 0):     # back to forward action change at lv6-5
                #     prev_action = 3-prev_action
                # if self.s == self.prev_prev_s and prev_action == 0:  # back to forward action change at lv6-5
                #     prev_action = 3

                action = prev_action
                print("previous same", self.s, self.duration, action)

            self.duration -= 1
        return action, 1.0

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        self.sampled_durations.append(d)
        return d

    def generate_exploration_episode(self, MAX_LENGTH):

        # self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        print(self.actions_node_matrix[HOME_NODE][0])

        a = 1   # Take action 1 at HOME NODE
        print("Starting at", self.s, "with prev at", self.prev_s)

        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # acting
            a, _ = self.choose_action(prev_action=a)
            s_next = self.take_action(self.prev_s, self.s, a)  # Take action
            print(f"moving from {self.s} => {s_next}")
            self.prev_prev_s = self.prev_s
            self.prev_s = self.s
            self.s = s_next

            # Record current state
            self.episode_state_traj.append(self.s)

            if len(self.episode_state_traj)%1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        # print(episode_maze_trajs)

        # import matplotlib.pyplot as plt
        # d_values = np.array(self.sampled_durations)
        # print(d_values)
        # print(len(d_values))
        # unique, counts = np.unique(d_values, return_counts=True)
        # print("unique, counts raw", unique, counts)
        # n, bins, patches = plt.hist(d_values[d_values<30], 150, facecolor='blue', alpha=0.5)
        # plt.show()
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1

        self.enable_alternate_action = False     # params["enable_alternate_action"]
        self.epsilon = params["epsilon"]
        self.mu = params["mu"]

        print("epsilon, V, agentId", self.epsilon, agentId)
        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[HOME_NODE, :] = 0
        if self.S == 129:
            Q[RWD_STATE, :] = 0

        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH)
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
    from sample_agent import run, load
    param_sets = [
        {"epsilon": 0.3, "mu": 2},
        {"epsilon": 0.3, "mu": 2}
    ]
    runids = run(eZGreedy(), param_sets, '/Users/usingla/mouse-maze/figs', '39999')
    print(runids)
    base_path = '/Users/usingla/mouse-maze/figs/'
    load([
        ('eZGreedy', runids)
    ], base_path)

