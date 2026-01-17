"""
A Base Temporally-Extended Greedy Model class.
"""

import numpy as np
import random

import parameters as p
from BaseModel import BaseModel
from utils import get_parent_node, connect_path_node, get_children, get_opp_child, get_parent_node_x_level_up
from options_pre import all_options_dict, straight_options_dict


class TeOptions(BaseModel):

    def __init__(self, file_suffix='_TeFixedActionOptionsTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0  # self.sample_duration()
        self.S = 128  # total states

        self.episode_state_traj = []
        self.s = p.HOME_NODE
        self.params = None

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        actions = self.get_valid_actions(state)
        return np.random.choice(actions)

    def make_it_go_to(self, target_node):

        temp_start_node = self.s
        temp_target_node = target_node
        if self.s == p.HOME_NODE:
            temp_start_node = 0
        if target_node == p.HOME_NODE:
            temp_target_node = 0

        path = connect_path_node(temp_start_node, temp_target_node)

        if self.s == p.HOME_NODE:
            path = [p.HOME_NODE] + path

        for n in path[1:]:
            self.episode_state_traj.append(n)

        if target_node == p.HOME_NODE:
            self.episode_state_traj.append(target_node)
        self.s = self.episode_state_traj[-1]
        return

    def execute_option(self, seq):
        self.episode_state_traj.extend(seq[1:])
        self.s = seq[-1]
        return

    def choose_action(self, prev_action):
        raise NotImplementedError

    def sample_duration(self):
        raise NotImplementedError

    def generate_exploration_episode(self, MAX_LENGTH):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        a = None  # Take action 1 at HOME NODE
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE  # since it's pure exploration

            # acting
            a = self.choose_action(prev_action=a)

            if a is not None:
                s_next = self.take_action(self.s, a)     # Take action
                self.s = s_next
                self.episode_state_traj.append(self.s)

        if self.s != p.HOME_NODE:
            self.make_it_go_to(p.HOME_NODE)

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        self.params = params
        success = 1
        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH)
        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
            "MAX_LENGTH": MAX_LENGTH,
        }
        return success, stats
