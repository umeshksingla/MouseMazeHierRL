"""
EpsilonZGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import parameters as p
import utils
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
import evaluation_metrics as em


class EpsilonZGreedy(BaseModel):

    def __init__(self, file_suffix='_EpsilonZGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0   # self.sample_duration()
        self.S = 128    # total states

        self.episode_state_traj = []
        self.visited_corners = []
        self.prev_s = None
        self.s = p.HOME_NODE

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        actions = self.get_valid_actions(state)
        return np.random.choice(actions)

    def choose_action(self, Q, *args, **kwargs):
        # print("s", self.s, "valid act", self.get_valid_actions(self.s))

        if self.s == p.HOME_NODE:
            # See https://www.notion.so/umeshksingla/original-ez-greedy-8eeb7d4a49104f52a6ba0a05c60ea6cd
            self.duration = 0
            return 1, 1.0

        if self.s in p.LVL_6_NODES:
            self.visited_corners.append(self.s)

        assert self.duration >= 0
        prev_action = kwargs['prev_action']

        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):
            if np.random.random() <= self.epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__(self.s)
                # self.duration -= 1
                # print(f"epsilon curr={self.s} sampled_dur={self.duration} action chosen = {action}")
            else:
                action = self.__random_action__(self.s)
                self.duration = 0
                # print(f"random curr={self.s} left_dur={self.duration} action chosen = {action}")
            if self.s in p.LVL_6_NODES:
                self.sampled_durations.append(self.duration)
        else:
            # if (self.prev_s in p.LVL_6_NODES) and (self.s in p.LVL_5_NODES):
            #     assert utils.get_parent_node(self.prev_s) == self.s
            #     opp_c = utils.get_the_other_children(self.s, self.prev_s)
            #     action = np.where(self.nodemap[self.s] == opp_c)[0][0]
            #     self.duration = 1

            # if self.memory_l5 != 'absent' and (self.prev_s in p.LVL_6_NODES) and (self.s in p.LVL_5_NODES):
            #     assert prev_action == 0
            #     # From L6 to opposite L6 component
            #     opp_c = utils.get_the_other_children(self.s, self.prev_s)
            #     # Memory component if already visited opposite
            #     x = {'strong': 1, 'weak': 2, 'mix': random.randint(1, 2)}[self.memory_l5]  # implement weak memory with randint if desired
            #     if self.visited_corners.count(opp_c) >= x:  # if already visited opposite node x times, go up the tree
            #         action = prev_action
            #         self.visited_corners.clear()
            #     else:  # else go to the opposite node
            #         action = np.where(self.nodemap[self.s] == opp_c)[0][0]

            # else:
            #     action = (3 - prev_action) % 3

            action = (3 - prev_action) % 3

            # if self.enable_alternate_action:    # Take the same action (assuming ALTERNATING action means straight path)
            #     # action = (3 - prev_action) % 3
            # else:                               # OR Take the same action (assuming SAME action means straight path)
            #     action = prev_action

            self.duration -= 1
        return action, 1.0

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        return d

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # print(self.nodemap)

        a = None   # Take action 1 at HOME NODE
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, _ = self.choose_action(Q, prev_action=a)
            s_next = self.take_action(self.s, a)     # Take action
            self.prev_s = self.s
            self.s = s_next

            if len(self.episode_state_traj)%5000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        # print(episode_maze_trajs)

        # plot durations sampled
        d_values = np.array(self.sampled_durations)
        unique, counts = np.unique(d_values, return_counts=True)
        print("unique, counts raw", unique, counts)
        n, bins, patches = plt.hist(d_values[d_values<30], bins=150, density=True, facecolor='blue', alpha=0.5)
        plt.title(f'mu={self.mu}')
        plt.savefig(f'../../figs/duration-mu={self.mu}_l6.png')
        # plt.show()
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        epsilon = params["epsilon"]     # epsilon
        # self.enable_alternate_action = True     # params["enable_alternate_action"]
        # self.memory_l5 = params["memory_l5"]      # 'strong', 'weak', 'absent'
        self.mu = params["mu"]
        self.epsilon = params["epsilon"]

        print("epsilon, V, agentId", epsilon, agentId)
        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[p.HOME_NODE, :] = 0
        if self.S == 129:
            Q[p.RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []

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
    from sample_agent import run, load
    param_sets = [
        # 3: {"epsilon": 0.4, "memory_l5": 'strong', "mu": 2},
        # 9: {"epsilon": 0.4, "memory_l5": 'weak', "mu": 2},
        # 4: {"epsilon": 0.4, "memory_l5": 'absent', "mu": 2},
        # 1: {"epsilon": 0.4, "memory_l5": 'strong', "mu": 1.9},
        # 10: {"epsilon": 0.4, "memory_l5": 'weak', "mu": 1.9},
        # 2: {"epsilon": 0.4, "memory_l5": 'absent', "mu": 1.9},
        # 5: {"epsilon": 0.4, "memory_l5": 'strong', "mu": 1.8},
        # 11: {"epsilon": 0.4, "memory_l5": 'weak', "mu": 1.8},
        # 6: {"epsilon": 0.4, "memory_l5": 'absent', "mu": 1.8},
        # 7: {"epsilon": 0.4, "memory_l5": 'strong', "mu": 1.7},
        # 12: {"epsilon": 0.4, "memory_l5": 'weak', "mu": 1.7},
        # 8: {"epsilon": 0.4, "memory_l5": 'absent', "mu": 1.7},

        # {"epsilon": 0.1, "memory_l5": 'strong', "mu": 2},
        # {"epsilon": 0.2, "memory_l5": 'strong', "mu": 2},
        # {"epsilon": 0.25, "memory_l5": 'strong', "mu": 2},
        # {"epsilon": 0.3, "memory_l5": 'strong', "mu": 2},
        # {"epsilon": 0.35, "memory_l5": 'strong', "mu": 2},

        # {"epsilon": 0.35, "mu": 1.9},
        # {"epsilon": 1, "mu": 1.7},
        # {"epsilon": 1, "mu": 1.6, "memory_l5": 'weak'},
        # {"epsilon": 1, "mu": 1.8, "memory_l5": 'weak'},
        # {"epsilon": 1, "mu": 2, "memory_l5": 'weak'},
        # {"epsilon": 1, "mu": 2.2, "memory_l5": 'weak'},
        # {"epsilon": 1, "mu": 2.4, "memory_l5": 'weak'},

        # {"epsilon": 1, "mu": 1.2, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 1.4, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 1.6, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 1.8, "memory_l5": 'absent'},
        {"epsilon": 0.3, "mu": 2,  'model': 'Levy'},
        # {"epsilon": 0.6, "mu": 2, "memory_l5": 'absent'},
        # {"epsilon": 0.6, "mu": 2, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 2.2, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 3.5, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 2.6, "memory_l5": 'absent'},
        # {"epsilon": 1, "mu": 2.8, "memory_l5": 'absent'},

        # {"epsilon": 1, "mu": 2.3},
        # {"epsilon": 1, "mu": 2},
        # {"epsilon": 1, "mu": 2},
        # {"epsilon": 1, "mu": 2},
        # {"epsilon": 0.3, "mu": 2.1},
        # {"epsilon": 0.3, "mu": 2.2},
        # {"epsilon": 0.3, "mu": 2.3},
        # {"epsilon": 0.3, "mu": 2.4},
        # {"epsilon": 0.3, "mu": 2},

    ]
    runids = run(EpsilonZGreedy(), param_sets, '/Users/usingla/mouse-maze/figs', '39999')
    print(runids)
    base_path = '/Users/usingla/mouse-maze/figs/'
    load([
        ('EpsilonZGreedy', runids)
    ], base_path)