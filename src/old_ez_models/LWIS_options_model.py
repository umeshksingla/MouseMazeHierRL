"""
Levy+IS: Written in terms of options at all levels
"""
import numpy as np
import random

import parameters as p
from BaseModel import BaseModel
from utils import connect_path_node
from options_pre import all_options_dict, straight_options_dict


class LWIS_options(BaseModel):

    def __init__(self, file_suffix='_LWISOptionsTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        # self.duration = 0  # self.sample_duration()
        self.S = 128  # total states

        self.episode_state_traj = []
        self.s = p.HOME_NODE

    # def __random_action__(self, state):
    #     """
    #     Random action from the actions available in this state.
    #     :return: random action index
    #     """
    #     actions = self.get_valid_actions(state)
    #     return np.random.choice(actions)

    # def make_it_go_to(self, target_node):
    #     print(self.s, target_node)
    #     path = connect_path_node(self.s, target_node)[1:]
    #     for n in path:
    #         self.episode_state_traj.append(n)
    #         self.s = self.episode_state_traj[-1]
    #     return

    def options_any_level(self, s, d):

        # if self.s == p.HOME_NODE:
        #     return self.options_dict[str(s)][str(d)]

        assert d >= 1
        if d >= 9: d = 9

        if p.LVL_BY_NODE[s] == 6:
            # d = random.choice([1, 2])
            return self.l6_options_dict[s][d]
        else:
            return self.options_dict[s][d]

    def choose_option(self, Q):

        # if self.s == p.HOME_NODE:
        #     self.duration = 0
        #     return 1

        if np.random.random() <= self.epsilon:
            d = self.sample_duration()
            options = self.options_any_level(self.s, d)
            seq = random.choice(options)
            # option_i = options.find(seq)
            # print("opt", len(options))
            # print("random seq", seq)
        else:
            raise NotImplementedError
            # self.duration = 1
            # print("opt", len(options))
            # option_i = np.argmax(Q[self.s][self.duration])  # greedy for now
            # seq = options[option_i]
            # print("greedy seq", seq)
        return list(seq)

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        return d

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[p.WATERPORT_NODE][1] = -1      # No action to go to RWD_STATE

        a = None
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE    # since it's pure exploration

            # acting
            # a = self.choose_action(Q, prev_action=a)
            seq = self.choose_option(Q)
            self.episode_state_traj.extend(seq[1:])
            self.s = self.episode_state_traj[-1]

            # if a is not None:   # sometimes we have sequence of actions which returns action as None but follows it internally
            #     s_next = self.take_action(self.s, a)    # Take action
            #     self.s = s_next
            #     # Record current state
            #     self.episode_state_traj.append(self.s)

            if len(self.episode_state_traj) % 2000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        # if self.s != 0:
        #     self.make_it_go_to(0)

        self.episode_state_traj.append(p.HOME_NODE)
        # print(self.episode_state_traj)
        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):

        print("Simulating agent with id", agentId)
        success = 1

        self.mu = params["mu"]
        self.epsilon = params["epsilon"]
        self.options_type = params['options']
        self.l6_options_type = params['l6options']

        if self.options_type == 'all':
            self.options_dict = all_options_dict
        elif self.options_type == 'straight':
            self.options_dict = straight_options_dict
        else:
            raise Exception('Invalid set of options specified.')

        if self.l6_options_type == 'all':
            self.l6_options_dict = all_options_dict
        elif self.l6_options_type == 'straight':
            self.l6_options_dict = straight_options_dict
        else:
            raise Exception('Invalid set of options for L6 specified.')

        Q = dict.fromkeys(range(self.S))
        # for n in range(self.S):
        #     Q[n] = dict.fromkeys(range(1, 10))
        #     for d in range(1, 10):
        #         print(n, d, self.options_dict[str(n)][str(d)])
        #         Q[n][d] = [0]*len(self.options_dict[str(n)][str(d)])
        # print(Q)
        Q = np.ones((self.S, self.A)) * 0.5  # Initialize state values
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

    param_sets = [
        # {"epsilon": 1.0, "mu": 1.9, 'model': 'ezg-custom'},
        # {"epsilon": 1.0, "mu": 1.95, 'model': 'ezg-custom'},
        # {"epsilon": 1.0, "mu": 2, 'model': 'final2', 'options': 'straight',  'rew': False},
        {"epsilon": 1.0, "mu": 2, 'options': 'straight', 'l6options': 'all', 'rew': False},
        # {"epsilon": 1.0, "mu": 2, 'options': 'straight', 'rew': False},
        # {"epsilon": 1.0, "mu": 2.05, 'model': 'ezg-custom'},
        # {"epsilon": 1.0, "mu": 2.1, 'model': 'ezg-custom'}
    ]

    runids = run(LWIS_options(), param_sets, '/Users/us3519/mouse-maze/figs', '50000', analyze=True)
    print(runids)
    # base_path = '/Users/usingla/mouse-maze/figs/'
    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('ezg-custom', runids)   # varying epsilon
    # ], base_path)

