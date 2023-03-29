"""
Levy+IS: Written in terms of options at level 6 but specified alternate at other levels
"""

import numpy as np
import random

import parameters as p
from BaseModel import BaseModel
from utils import get_parent_node, connect_path_node, get_children, get_opp_child, get_parent_node_x_level_up
from options_pre import options_dict, straight_options_dict


class LWISOptL6Alternate(BaseModel):

    def __init__(self, file_suffix='_LWISOptL6AlternateTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0  # self.sample_duration()
        self.S = 128  # total states

        self.episode_state_traj = []
        self.s = p.HOME_NODE

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

    def sample_option(self):
        assert self.s in p.LVL_6_NODES
        assert self.duration >= 1
        if self.duration >= 9: self.duration = 9
        options_available = self.l6_options_dict[str(self.s)][str(self.duration)]
        print("options_available", options_available)
        return random.choice(options_available)

    def execute_option(self, seq):
        print("seq", seq)
        self.episode_state_traj.extend(seq[1:])
        self.s = seq[-1]
        return

    def choose_action(self, Q, *args, **kwargs):

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        assert self.duration >= 0
        prev_action = kwargs['prev_action']

        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):
            if np.random.random() <= self.epsilon:
                self.duration = self.sample_duration()
            else:
                self.duration = 1

            if self.s in p.LVL_6_NODES:
                self.execute_option(self.sample_option())   # composite actions
                self.duration = 1
                action = None
            else:
                action = self.__random_action__(self.s)
        else:
            action = (3 - prev_action) % 3
        self.duration -= 1
        return action

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        return d

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        a = None  # Take action 1 at HOME NODE
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE  # since it's pure exploration

            # acting
            a = self.choose_action(Q, prev_action=a)

            if a is not None:  # sometimes we have sequence of actions which returns action as None but follows it internally
                s_next = self.take_action(self.s, a)     # Take action
                self.s = s_next
                # Record current state
                self.episode_state_traj.append(self.s)

            if len(self.episode_state_traj) % 5000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        if self.s != p.HOME_NODE:
            self.make_it_go_to(p.HOME_NODE)

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1

        self.mu = params["mu"]
        self.epsilon = params["epsilon"]
        self.l6_options_type = params['l6options']

        if self.l6_options_type == 'all':
            self.l6_options_dict = options_dict
        elif self.l6_options_type == 'straight':
            self.l6_options_dict = straight_options_dict
        else:
            raise Exception('Invalid set of options for L6 specified.')

        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[p.HOME_NODE, :] = 0
        if self.S == 129:
            Q[p.RWD_STATE, :] = 0

        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH, Q)
        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
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
        # {"epsilon": 1.0, "mu": 2, 'l6options': 'straight', 'rew': False},
        {"epsilon": 1.0, "mu": 2, 'l6options': 'all', 'rew': True},
    ]

    runids = run(LWISOptL6Alternate(), param_sets, '/Users/usingla/mouse-maze/figs', '35000', analyze=True)
    print(runids)
    # base_path = '/Users/usingla/mouse-maze/figs/'

    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('ezg-custom', [52979, 232725, 587308])   # varying epsilon
    # ], base_path)
    #
    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('ezg-custom', [950365, 587773, 807980])    # varying mu
    # ], base_path)
