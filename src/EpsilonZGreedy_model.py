"""
EpsilonZGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
import evaluation_metrics as em


class EpsilonZGreedy(BaseModel):

    def __init__(self, file_suffix='_EpsilonZGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = self.sample_duration()
        self.S = 128    # total states

        self.episode_state_traj = []
        self.prev_level3_node = None
        self.s = HOME_NODE  # start from s

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
            print("LoS")
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
                print("epsilon", self.s, self.duration, action)
            else:
                action = self.__random_action__(self.s)
                print("random", self.s, self.duration, action)
        else:
            prev_action = kwargs['prev_action']

            if self.enable_alternate_action:
                # Take the same action (assuming ALTERNATING action means straight path) TODO: think more
                action = (3 - prev_action) % 3
                print("previous alt", self.s, self.duration, action)
            else:
                # OR Take the same action (assuming SAME action means straight path)
                action = prev_action
                print("previous same", self.s, self.duration, action)

            self.duration -= 1
        return action, 1.0

    def sample_duration(self):
        # return 1
        d = 1+np.random.zipf(a=2)
        # print("duration chosen", d)
        self.sampled_durations.append(d)
        return d

    def generate_exploration_episode(self, epsilon, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        print(self.nodemap)

        a = None   # Take action 1 at HOME NODE
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, _ = self.choose_action(Q, epsilon, prev_action=a)
            s_next = self.take_action(self.s, a)     # Take action
            self.s = s_next

            if len(self.episode_state_traj)%1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        print(episode_maze_trajs)

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
        epsilon = params["epsilon"]     # epsilon
        self.enable_alternate_action = True     # params["enable_alternate_action"]
        self.enable_LoS = False     # params["enable_LoS"]

        print("epsilon, V, agentId", epsilon, agentId)
        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[HOME_NODE, :] = 0
        if self.S == 129:
            Q[RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []

        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(epsilon, MAX_LENGTH, Q)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)

        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes_state_trajs),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
            # "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency": calculate_normalized_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency_by_level": calculate_normalized_visit_frequency_by_level(
                all_episodes_state_trajs)
        }
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


if __name__ == '__main__':
    from sample_agent import run
    param_sets = {
        4: {"epsilon": 0.3},
        # 5: {"epsilon": 0.3},
    }
    run(EpsilonZGreedy(), param_sets, '/Users/usingla/mouse-maze/figs', '30005_e0.3')

