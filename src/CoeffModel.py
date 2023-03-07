"""
Coeff distance model
"""

import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from scipy.special import logsumexp
from actions import actions_node_matrix
from maze_spatial_mapping import CELL_XY, NODE_CELL_MAPPING
from scipy.special import log_softmax, softmax

# def softmax_dict(d):
#     x = np.array(list(d.values()))
#     den_x = logsumexp(x, keepdims=True)
#     return {k: np.exp(v - den_x)[0] for k, v in d.items()}

LOW_VAL = -9999999


class CoeffModel(BaseModel):

    def __init__(self, file_suffix='_coeffTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.S = 128  # total states
        self.episode_state_traj = [0, random.choice([1, 2]), 0, random.choice([1, 2]), 0]
        self.s = self.episode_state_traj[-1]
        self.prev_s = self.episode_state_traj[-2]
        self.actions_node_matrix = actions_node_matrix

    name = 'coeff'

    # @staticmethod
    # def sample_key_from_dict(d):
    #     d = list(d.items())
    #     next_node = random.choices([k[0] for k in d], weights=[k[1] for k in d], k=1)[0]
    #     return next_node

    def take_action(self, prev_n: int, n: int, a: int) -> int:
        return int(self.actions_node_matrix[prev_n][n][a])

    def choose_action(self):

        # if self.s == HOME_NODE:
        #     return 0, 0.0

        tcell = self.episode_state_traj[-self.n_time_steps:]
        assert len(tcell) == self.n_time_steps
        assert tcell[-1] == self.s
        assert tcell[-2] == self.prev_s

        V = np.zeros(4)
        d = np.zeros(self.n_time_steps)
        next_possible_action_nodes = self.actions_node_matrix[self.prev_s][self.s]
        # print("At:", tcell, self.actions_node_matrix[self.prev_s][self.s], "At:", self.s)
        for a in range(len(next_possible_action_nodes)):
            next_n = next_possible_action_nodes[a]
            if next_n == INVALID_STATE:
                V[a] = LOW_VAL
                continue
            choice_cell_coors = CELL_XY[NODE_CELL_MAPPING[next_n]]
            for i in range(len(tcell)-1, 0, -1):
                d[i] = np.linalg.norm(choice_cell_coors - CELL_XY[NODE_CELL_MAPPING[tcell[i]]])
                # print("between", tcell[i], next_n, d[i])
            # print(d, self.coef)
            V[a] = d.dot(self.coef)
        prob_scores = softmax(V)
        # print("V, prob_scores", V, prob_scores)
        choices = [a for a in range(len(prob_scores)) if V[a] != LOW_VAL]
        weights = [prob_scores[a] for a in range(len(prob_scores)) if V[a] != LOW_VAL]
        # print("choices, weights", choices, weights)
        next_a = random.choices(choices, weights=weights)[0]
        # print("selected:", next_a)
        return next_a, np.log(prob_scores[next_a])

    def generate_exploration_episode(self, MAX_LENGTH):

        self.nodemap[WATERPORT_NODE][1] = -1    # No action to go to RWD_STATE

        print("Starting at", self.s, "with prev at", self.prev_s)
        LL = 0.0
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Act
            a, a_prob = self.choose_action()
            # print(ret)
            # s_next, s_next_prob = ret
            s_next = self.take_action(self.prev_s, self.s, a)  # Take action
            # print("traj till now", self.episode_state_traj)
            # print(f"moving from {self.s} => {s_next}")
            self.prev_s = self.s
            self.s = s_next
            LL += a_prob

            # Record current state
            self.episode_state_traj.append(self.s)

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, LL

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)
        self.coef = params['coef']
        self.n_time_steps = len(self.coef)
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        Q = np.zeros((self.S, self.A))  # Initialize state values
        _, episode_state_trajs, episode_maze_trajs, LL = self.generate_exploration_episode(MAX_LENGTH)
        all_episodes_state_trajs.extend(episode_state_trajs)
        all_episodes_pos_trajs.extend(episode_maze_trajs)
        print(all_episodes_state_trajs)

        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes_state_trajs),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
        }
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


# Driver Code
if __name__ == '__main__':
    from sample_agent import run
    param_sets = {
        # 1: {'coef': [0.05, 0.23, 0.19, -0.15, 0.13], 'model': 'B5coeffs'},
        # 2: {'coef': [-0.1, 0.08, 0.4, -0.27, 0.22], 'model': 'B6coeffs'},
        1: {'coef': [0.5, 0.0, 0.5, 0.0], 'model': 'randomcoeffs'},
        # 2: {'back_prob': 0.2},
        # 2: {'back_prob': 0.3},
        # 3: {'back_prob': 0.2},
    }
    run(CoeffModel(), param_sets, '/Users/usingla/mouse-maze/figs', '20000_random')

