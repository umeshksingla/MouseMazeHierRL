"""
EpsilonZGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency
import evaluation_metrics as em


class EpsilonZGreedy(EpsilonGreedy):

    def __init__(self, file_suffix='_EpsilonZGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = self.sample_duration()
        self.S = 128    # total states

        self.episode_state_traj = []
        self.prev_level3_node = None
        self.s = HOME_NODE  # start from s
        self.enable_LoS = False

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        actions = self.get_valid_actions(state)
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

    # def get_into_the_maze(self):
    #     # It will be stuck in L6 coz a direction that led an agent in cannot get it out
    #     assert LVL_BY_NODE[self.s] == 5
    #     assert len(self.episode_state_traj) >= 4
    #     path_3nodes_back = self.episode_state_traj[-3:]
    #     if self.level3_path(self.s, go_up=2) == path_3nodes_back: # i.e. followed the LoS path, not from level 5 opp side
    #         self.s = los_node_mapping_lvl5_6[self.s]
    #         self.episode_state_traj.append(self.s)
    #     return

    def choose_action(self, Q, epsilon, *args, **kwargs):
        # print("s", self.s, "valid act", self.get_valid_actions(self.s))

        if self.s == HOME_NODE:
            # See https://www.notion.so/umeshksingla/original-ez-greedy-8eeb7d4a49104f52a6ba0a05c60ea6cd
            self.duration = 0
            return 1, 1.0

        if self.enable_LoS:
            if self.s in LVL_5_NODES:
                path_3nodes_back = self.episode_state_traj[-3:]
                los_path_3nodes_back = self.level3_path(self.s, go_up=2)
                print(path_3nodes_back, los_path_3nodes_back)
                if los_path_3nodes_back == path_3nodes_back:
                    # TODO: make it go to LoS node with 80% prob
                    if np.random.random() <= 1.0:
                        print("LoS")
                        print("before", self.s)
                        self.s = los_node_mapping_lvl5_6[self.s]
                        print("after", self.s)
                        self.episode_state_traj.append(self.s)
                    else:
                        print("not LoS")

        assert self.duration >= 0
        if self.duration == 0 or kwargs['prev_action'] not in self.get_valid_actions(self.s):
            if np.random.random() <= epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__(self.s)
                self.duration -= 1
                if self.s in [15, 32, 66]:
                    print("epsilon", self.s, self.duration, action)
            else:
                action = self.__random_action__(self.s)
                if self.s in [15, 32, 66]:
                    print("random", self.s, self.duration, action)
        else:
            prev_action = kwargs['prev_action']
            # A straight path denotes alternating actions (TODO think more)
            if prev_action != 0:
                action = 1 if prev_action == 2 else 2
            else:
                action = prev_action
            self.duration -= 1
            if self.s in [15, 32, 66]:
                print("previous", self.s, self.duration, action)
        return action, 1.0

    # def choose_action(self, s, Q, epsilon, *args, **kwargs):
    #     if self.duration == 0 or kwargs['prev_action'] not in self.get_valid_actions(s):
    #         self.duration = self.sample_duration()
    #         action = self.__random_action__(s)
    #     else:
    #         action = kwargs['prev_action']
    #         self.duration -= 1
    #     return action, 1.0

    def sample_duration(self):
        # return 1
        d = 1+np.random.zipf(a=2)
        # print("duration chosen", d)
        self.sampled_durations.append(d)
        return d

    def generate_exploration_episode(self, alpha, gamma, lamda, epsilon, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # self.nodemap[0][1] = -1
        # self.nodemap[2][1] = -1
        # self.nodemap[1][2] = -1
        print(self.nodemap)

        # e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        a = None   # Take action 1 at HOME NODE
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, a_prob = self.choose_action(Q, epsilon, prev_action=a)
            s_next = self.take_action(self.s, a)     # Take action
            # print("action: ", a, f": {s} => {s_next}")

            # td_error = 0.0 + gamma * np.max([Q[s_next, a_i] for a_i in self.get_valid_actions(s_next)]) - Q[s, a]   # R = 0
            # e[s, a] += 1
            # for n in np.arange(self.S):
            #     Q[n, :] += alpha * td_error * e[n, :]
            #     e[n, :] = gamma * lamda * e[n, :]
            # Q[s, a] = self.is_valid_state_value(Q[s, a])
            self.s = s_next
            if len(self.episode_state_traj)%1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs = break_simulated_traj_into_episodes(self.episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same
        import matplotlib.pyplot as plt

        d_values = np.array(self.sampled_durations)
        print(d_values)
        print(len(d_values))
        unique, counts = np.unique(d_values, return_counts=True)
        print("unique, counts raw", unique, counts)
        n, bins, patches = plt.hist(d_values[d_values<30], 150, facecolor='blue', alpha=0.5)
        plt.show()
        return True, episode_state_trajs, episode_maze_trajs, 0.0


if __name__ == '__main__':
    pass
