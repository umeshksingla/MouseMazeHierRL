"""
EZ+switch probabilities between 3 states (explore, leave, reward) built upon Final2_model.


|from / to:   | leave | drink | explore |
|:--|:-:|:-:|:-:|
|**leave**    |   |0.51 ± 0.14|0.49 ± 0.14|
|**drink**    |0.10 ± 0.05|  |0.90 ± 0.05|
|**explore**  |0.40 ± 0.11|0.60 ± 0.11|   |


"""
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

import parameters as p
from parameters import LEAVE, DRINK, EXPLORE, mode_labels
from LWISOptL6Alternate_model import LWISOptL6Alternate
from utils import get_parent_node, connect_path_node, get_children, get_opp_child, get_parent_node_x_level_up
from options_pre import options_dict, straight_options_dict


class LWISOptL6AlternateReward(LWISOptL6Alternate):

    def __init__(self, file_suffix='_LWISOptL6AlternateRewardTrajectories'):
        LWISOptL6Alternate.__init__(self, file_suffix=file_suffix)

        self.current_mode = EXPLORE
        self.before_first_reward_mode_transition = {
            # mode: other modes        mean-probability
            LEAVE: ([EXPLORE], [1.0]),
            EXPLORE: ([EXPLORE, LEAVE], [0.95, 0.05]),
        }

        self.after_first_reward_mode_transition = {
            # mode: other modes        mean-probability
            DRINK: ([EXPLORE, LEAVE], [0.9, 0.1]),
            LEAVE: ([DRINK, EXPLORE], [0.51, 0.49]),
            EXPLORE: ([EXPLORE, DRINK, LEAVE], [0.85, 0.6*0.15, 0.4*0.15]),
        }

        self.mode_transition = self.before_first_reward_mode_transition
        self.first_reward = False
        modes = [EXPLORE, LEAVE, DRINK]
        self.mode_switches = dict.fromkeys(modes)

    def switch_phase(self):
        assert self.first_reward
        self.mode_transition = self.after_first_reward_mode_transition
        print(f"SWITCHED PHASE at step {len(self.episode_state_traj)}")
        return

    def choose_action(self, *args, **kwargs):

        if not self.first_reward and self.s == p.WATERPORT_NODE:
            self.first_reward = True
            # self.switch_phase()

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        assert self.duration >= 0
        prev_action = kwargs['prev_action']
        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):

            prev_mode = self.current_mode
            modes, transition_probs = self.mode_transition[prev_mode]
            new_mode = np.random.choice(modes, p=transition_probs)
            # print("new_mode", mode_labels[self.current_mode], [mode_labels[i] for i in modes], transition_probs, mode_labels[new_mode])
            self.current_mode = new_mode

            if prev_mode != new_mode:
                self.mode_switches[prev_mode][new_mode] += 1

            if self.current_mode == EXPLORE:
                self.duration = self.sample_duration()
                if self.s in p.LVL_6_NODES:  # composite intense search actions at L6
                    # self.make_it_go_to(self.sample_option(self.s, self.duration)[-1])
                    self.execute_option(self.sample_option())
                    self.duration = 1
                    action = None
                else:   # alternate actions at L0-5
                    action = self.__random_action__(self.s)
            elif self.current_mode == LEAVE:
                self.duration = 1
                self.make_it_go_to(p.HOME_NODE)     # !!!!!!!!
                action = None
            else:
                self.duration = 1
                self.make_it_go_to(p.WATERPORT_NODE)
                action = None
        else:
            action = (3 - prev_action) % 3
        self.duration -= 1
        return action

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        return d

    def generate_exploration_episode(self, MAX_LENGTH):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        a = None
        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE  # since it's pure exploration

            # acting
            a = self.choose_action(prev_action=a)
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

        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH)
        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
        }
        print("self.mode_switches")
        print(self.mode_switches)
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


if __name__ == '__main__':
    from sample_agent import run, load

    param_sets = [
        {"epsilon": 1.0, "mu": 2, 'l6options': 'all', 'rew': True},
    ]
    runids = run(LWISOptL6AlternateReward(), param_sets, '/Users/usingla/mouse-maze/figs', '36000', analyze=True)
    print(runids)

    # base_path = '/Users/usingla/mouse-maze/figs/'
    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('ezg-custom', [52979, 232725, 587308])   # varying epsilon
    # ], base_path)
