"""
EZgreedy and Custom fused: Written using options framework for level 6 choices
"""
import numpy as np
import random
import matplotlib.pyplot as plt

import parameters as p
from BaseModel import BaseModel
from utils import get_parent_node, connect_path_node, get_children, get_opp_child, get_parent_node_x_level_up


class Final2(BaseModel):

    def __init__(self, file_suffix='_Final2Trajectories'):
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
        path = connect_path_node(self.s, target_node)[1:]
        for n in path:
            self.episode_state_traj.append(n)
            self.s = self.episode_state_traj[-1]
            self.prev_s = self.episode_state_traj[-2]
        return

    def options_at_level_6(self, d):
        """
        Defines path options the agent has in all directions for a certain path length d. If the path length is longer
        than a possible path in the maze in a certain direction, it is still considered as one option because we can
        operate under the assymption that animals do not have an estimate of how far along a certain direction is
        reachable.
        """
        assert d >= 1
        choices = {
            '1': [
                get_parent_node_x_level_up(self.s, x=1)
            ],
            '2': [
                get_parent_node_x_level_up(self.s, x=2),
                get_opp_child(self.s)
            ],
            '3': [
                get_opp_child(self.s),  # len 2

                get_parent_node_x_level_up(self.s, x=3),
                get_opp_child(get_parent_node(self.s))
            ],
            '4': [
                get_opp_child(self.s),  # len 2

                get_parent_node_x_level_up(self.s, x=4),
                *get_children(get_opp_child(get_parent_node(self.s))),
                get_opp_child(get_parent_node_x_level_up(self.s, x=2))
            ],
            '5': [
                get_opp_child(self.s),  # len 2
                *get_children(get_opp_child(get_parent_node(self.s))),  # len 4

                get_parent_node_x_level_up(self.s, x=5),
                get_opp_child(get_parent_node_x_level_up(self.s, x=3)),
                *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))
            ],
            '6': [

                get_opp_child(self.s),  # 2
                *get_children(get_opp_child(get_parent_node(self.s))),  # 4

                get_parent_node_x_level_up(self.s, x=6),
                get_opp_child(get_parent_node_x_level_up(self.s, x=4)),
                *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3))),
                *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))[0]),
                *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))[1])
            ],
            '>=7': [
                p.HOME_NODE,
                *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=4))),
                get_opp_child(get_parent_node_x_level_up(self.s, x=5)),
                *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3)))[0]),
                *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3)))[1]),
                *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=5)))
            ]
        }

        options = choices[str(d) if d <= 6 else '>=7']
        return random.choice(options)

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
                self.make_it_go_to(self.options_at_level_6(self.duration))   # composite actions
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
        # print(self.nodemap)

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

    # param_sets = [
    #     {"epsilon": 0.8, "mu": 2, 'model': 'ezg-custom'},
    #     {"epsilon": 0.7, "mu": 2, 'model': 'ezg-custom'}
    # ]

    # param_sets = [
    #     {"epsilon": 1.0, "mu": 1.9, 'model': 'ezg-custom'},
    #     {"epsilon": 1.0, "mu": 1.95, 'model': 'ezg-custom'},
    #     {"epsilon": 1.0, "mu": 2, 'model': 'ezg-custom'},
    #     {"epsilon": 1.0, "mu": 2.05, 'model': 'ezg-custom'},
    #     {"epsilon": 1.0, "mu": 2.1, 'model': 'ezg-custom'}
    # ]

    # runids = run(Final2(), param_sets, '/Users/usingla/mouse-maze/figs', '40000', analyze=True)
    # print(runids)
    base_path = '/Users/usingla/mouse-maze/figs/'

    load([
        ('BiasedWalk4', [50173]),
        ('ezg-custom', [52979, 232725, 587308])   # varying epsilon
    ], base_path)
    #
    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('ezg-custom', [950365, 587773, 807980])    # varying mu
    # ], base_path)
