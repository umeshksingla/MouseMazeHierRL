"""
EZgreedy with brute force long actions

Only for unrewarded case.
"""
import numpy as np
import random

import parameters as p
from BaseModel import BaseModel
from utils import get_parent_node, connect_path_node, get_children, get_opp_child


class EZBruteForceLongActions(BaseModel):

    def __init__(self, file_suffix='_FinalTrajectories'):
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

    def choose_action(self, Q, *args, **kwargs):

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        assert self.duration >= 0
        prev_action = kwargs['prev_action']

        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):

            while self.s in p.LVL_6_NODES:  # as long as it is at a level 6 node
                n_step = random.choices(['1', '2', '4', '4_out', '6', '5_out', '6_out'], weights=[0.5, 0.2, 0.16, 0.036, 0.054, 0.045, 0.005])[0]
                if n_step == '1':           # takes it to level 5
                    self.make_it_go_to(get_parent_node(self.s))
                elif n_step == '2':         # takes it to level 6
                    self.make_it_go_to(get_opp_child(self.s))
                elif n_step == '4':         # takes it to level 6
                    self.make_it_go_to(random.choice(get_children(get_opp_child(get_parent_node(self.s)))))
                elif n_step == '4_out':     # takes it to level 4
                    self.make_it_go_to(get_opp_child(get_parent_node(get_parent_node(self.s))))
                elif n_step == '6':         # takes it to level 4
                    self.make_it_go_to(random.choice(get_children(get_opp_child(get_parent_node(get_parent_node(get_parent_node(self.s)))))))
                elif n_step == '5_out':     # takes it to level 1
                    self.make_it_go_to(1)
                elif n_step == '6_out':     # takes it to level 0
                    self.make_it_go_to(0)
                else:
                    raise Exception(f'wrong sampled step {n_step}')

            if np.random.random() <= self.epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__(self.s)
            else:
                action = self.__random_action__(self.s)
                self.duration = 0
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

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a = self.choose_action(Q, prev_action=a)
            s_next = self.take_action(self.s, a)  # Take action
            print("s, a, s'", self.s, a, s_next)
            self.s = s_next

            if len(self.episode_state_traj) % 5000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

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

    param_sets = [
                     {"epsilon": 0.3, "mu": 2, 'model': 'EZBruteForceLongActions', 'rew': False}
                 ]
    runids = run(EZBruteForceLongActions(), param_sets, '/Users/usingla/mouse-maze/figs', '40000')
    print(runids)
