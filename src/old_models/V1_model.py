"""
Our model: version 1
"""

import os
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy

import parameters as p
from BaseModel import BaseModel
from utils import get_outward_pref_order, get_parent_node, connect_path_node


class V1(BaseModel):
    """
    outward preference
    backward preference

    At L6: go tp L5
    """

    def __init__(self, file_suffix='_V1Trajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.S = 128  # total states

        # self.curr_directions = None
        self.episode_state_traj = []
        self.s = 0  # Start from 0
        self.prev_s = p.HOME_NODE

    @staticmethod
    def sample_key_from_dict(pref_order):
        pref_order = list(pref_order.items())
        next_node = random.choices([k[0] for k in pref_order], weights=[k[1] for k in pref_order], k=1)[0]
        return next_node

    @staticmethod
    def sample_key_from_dict_unweighted(pref_order):
        pref_order = list(pref_order.items())
        next_node = random.choices([k[0] for k in pref_order], k=1)[0]
        return next_node

    def forward_biases(self):

        # print("forward", self.s, self.prev_s)

        back_child = get_parent_node(self.s)
        assert self.prev_s == back_child

        if (self.s in p.LVL_1_NODES) or (self.s in p.LVL_0_NODES):  # 50-50
            pref_order = get_outward_pref_order(self.s, 0.5, self.FORWARD_GOBACK_PROB)
        elif (self.s in p.LVL_2_NODES) or (self.s in p.LVL_3_NODES) or (self.s in p.LVL_4_NODES) or (self.s in p.LVL_5_NODES): # tendency to go outwards
            pref_order = get_outward_pref_order(self.s, self.OUTWARD_PREFERENCE_PROB, self.FORWARD_GOBACK_PROB)
        else:
            raise Exception(f"Error in forward biases. At level 6 probably. self.s = {self.s}")

        # print("at", n, "got fwd pref_order", pref_order)
        next_node = self.sample_key_from_dict(pref_order)
        return next_node

    def backward_biases(self):

        valid_next_s = self.get_valid_next_states(self.s)
        # print("valid_next_s", valid_next_s, "prev_s", self.prev_s)

        pref_order = dict.fromkeys(valid_next_s, 0.5 * (1-self.BACKWARD_GOBACK_PROB))
        pref_order[self.prev_s] = self.BACKWARD_GOBACK_PROB   # i.e. go back

        # print("at", self.s, "got bkwd pref_order", pref_order)
        next_node = self.sample_key_from_dict(pref_order)
        return next_node

    def choose_action(self):

        if self.s == p.HOME_NODE:
            return 0

        if self.s in p.LVL_6_NODES:
            return get_parent_node(self.s)

        # print("lvls", LVL_BY_NODE[self.s], LVL_BY_NODE[self.prev_s], LVL_BY_NODE[self.s]!= LVL_BY_NODE[self.prev_s]+1)

        # if moving down the tree, forward biases apply with less prob to moving up
        if p.LVL_BY_NODE[self.s] == p.LVL_BY_NODE[self.prev_s] + 1:
            next_node = self.forward_biases()
        else:
            # if moving up the tree, backward biases apply with less prob of moving down where it just came from, and
            # rest 2 nodes equal pref
            next_node = self.backward_biases()

        # assert next_node in self.nodemap[self.s, :]
        return next_node

    def generate_exploration_episode(self, MAX_LENGTH):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # self.nodemap[0][0] = -1  # No action to go to HOME_NODE

        print("Starting at", self.s, "with prev at", self.prev_s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # Act
            s_next = self.choose_action()   # NOTE: there's a side effect to some versions where they change s
            # print("traj till now", self.episode_state_traj)
            # print(f"moving from {self.s} => {s_next}")
            self.prev_s = self.s
            self.s = s_next

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)

        self.BACKWARD_GOBACK_PROB = self.FORWARD_GOBACK_PROB = self.STRAIGHT_BACK_PROB = params['back_prob']
        self.OUTWARD_PREFERENCE_PROB = params['outward_pref']

        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        Q = np.zeros((self.S, self.A))  # Initialize state values
        _, episode_state_trajs, episode_maze_trajs, _ = self.generate_exploration_episode(MAX_LENGTH)
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


# Driver Code
if __name__ == '__main__':
    from sample_agent import run
    param_sets = [
        {'back_prob': 0.1, 'outward_pref': 0.75},
        {'back_prob': 0.2, 'outward_pref': 0.75},
    ]
    runids = run(V1(), param_sets, '/Users/usingla/mouse-maze/figs', '30001')
    print(runids)

