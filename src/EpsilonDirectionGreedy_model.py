"""
EpsilonDirectionGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency
import evaluation_metrics as em


class EpsilonDirectionGreedy(EpsilonGreedy):

    def __init__(self, file_suffix='_EpsilonDirectionGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.direction_intervals = [(i-90, i) for i in range(90, 450, 90)]
        self.curr_directions = self.sample_directions()
        self.duration = self.sample_duration()

    def get_interval(self, target_angle):
        for i in self.direction_intervals:
            if i[0] <= target_angle <= i[1]:
                return i

    def get_valid_directions(self, target_angle):
        """ returns primary and secondary direction """
        vertical_dir = 'up' if 0 <= target_angle <= 180 else 'down'
        horizontal_dir = 'left' if 90 <= target_angle <= 270 else 'right'

        target_interval = self.get_interval(target_angle)
        mid = abs(target_interval[1]+target_interval[0])/2

        if target_interval in [(0, 90), (180, 270)]:
            if target_angle >= mid:
                return vertical_dir, horizontal_dir
            else:
                return horizontal_dir, vertical_dir
        else:
            if target_angle >= mid:
                return horizontal_dir, vertical_dir
            else:
                return vertical_dir, horizontal_dir

    def get_action_direction_mapping(self, state):
        """ mapping of direction to actions in the nodemap for each state"""
        node_lvl = LVL_BY_NODE[state]
        if node_lvl == 6:
            if state % 2 == 0:
                directions = ['left', '', '']      # e.g. 84, 108, 72
            else:
                directions = ['right', '', '']     # e.g. 99, 111, 83
        elif node_lvl % 2 == 0:
            # i.e. level is 0, 2, 4
            if state % 2 == 0:
                directions = ['left', 'up', 'down']    # e.g. 6, 28, 16
            else:
                directions = ['right', 'up', 'down']  # e.g. 5, 27, 21
        else:
            # i.e. level is 1, 3, 5
            if state % 2 == 0:
                directions = ['up', 'left', 'right']    # e.g. 10, 42, 2
            else:
                directions = ['down', 'left', 'right']  # e.g. 13, 1, 35
        return directions

    def choose_action(self, s, Q, epsilon, random_dir, version):
        if version == 1:
            return self.choose_action_1(s, Q, random_dir)
        if version == 2:
            return self.choose_action_2(s, Q, epsilon, random_dir)
        if version == 3:
            return self.choose_action_3(s, Q, epsilon, random_dir)
        if version == 4:
            return self.choose_action_4(s, Q, epsilon, random_dir)

    def choose_action_4(self, s, Q, epsilon, random_dir):
        raise NotImplementedError
        """
        Version 4

        if duration is still left and possible to go in the prev direction:
            continue the direction
            duration -= 1
        else:
            sample a duration
            sample a direction

        """
        print("Version 3")

        if s == HOME_NODE:
            self.duration = self.sample_duration()
            self.curr_directions = self.sample_directions()
            return 1, 1.0   # Go to node 0

        state_valid_directions = self.get_action_direction_mapping(s)
        dir1, dir2 = self.curr_directions
        print("state", s, "current", dir1, dir2, "- valid ones", state_valid_directions)

        if self.duration != 0 and (dir1 in state_valid_directions or dir2 in state_valid_directions):
                if random_dir:
                    chosen_dir = random.choice(tuple({dir1, dir2}.intersection(state_valid_directions)))
                else:
                    # choose directions in a priority order
                    if dir1 in state_valid_directions:
                        chosen_dir = dir1
                    else:
                        chosen_dir = dir2
                action = state_valid_directions.index(chosen_dir)
                print("random direction chosen", chosen_dir)
                assert action in self.get_valid_actions(s)
                self.duration -= 1
        else:
            self.duration = self.sample_duration()
            self.curr_directions = self.sample_directions()
            return self.choose_action_3(s, Q, epsilon, random_dir)
        return action, 1.0

    def choose_action_3(self, s, Q, epsilon, random_dir):
        """
        Version 3

        if duration is still left and possible to go in the current sampled direction:
            continue the direction      # 2 variants
            duration -= 1
        else:
            sample a duration
            sample a direction

        """
        print("Version 3")

        if s == HOME_NODE:
            self.duration = self.sample_duration()
            self.curr_directions = self.sample_directions()
            return 1, 1.0   # Go to node 0

        state_valid_directions = self.get_action_direction_mapping(s)
        dir1, dir2 = self.curr_directions
        print("state", s, "current", dir1, dir2, "- valid ones", state_valid_directions)

        if self.duration != 0 and (dir1 in state_valid_directions or dir2 in state_valid_directions):
                if random_dir:
                    chosen_dir = random.choice(tuple({dir1, dir2}.intersection(state_valid_directions)))
                else:
                    # choose directions in a priority order
                    if dir1 in state_valid_directions:
                        chosen_dir = dir1
                    else:
                        chosen_dir = dir2
                action = state_valid_directions.index(chosen_dir)
                print("random direction chosen", chosen_dir)
                assert action in self.get_valid_actions(s)
                self.duration -= 1
        else:
            self.duration = self.sample_duration()
            self.curr_directions = self.sample_directions()
            return self.choose_action_3(s, Q, epsilon, random_dir)
        return action, 1.0

    def choose_action_2(self, s, Q, epsilon, random_dir):
        """
        Version 2

        if random() <= epsilon
            sample a random direction and follow
        else
            if one can follow an action using prev direction
                do it
            else
                sample a random direction and follow

        """
        print("Version 2")
        if s == HOME_NODE:
            self.curr_directions = self.sample_directions()
            return 1, 1.0   # Go to node 0

        if np.random.random() <= epsilon:
            print("random direction")
            self.curr_directions = self.sample_directions()
            return self.choose_action_2(s, Q, epsilon, random_dir)
        else:
            print("prev sampled direction")
            state_valid_directions = self.get_action_direction_mapping(s)
            dir1, dir2 = self.curr_directions
            print("state", s, "current", dir1, dir2, "- valid ones", state_valid_directions)
            if dir1 in state_valid_directions or dir2 in state_valid_directions:
                if random_dir:
                    chosen_dir = random.choice(tuple({dir1, dir2}.intersection(state_valid_directions)))
                else:
                    # choose directions in a priority order
                    if dir1 in state_valid_directions:
                        chosen_dir = dir1
                    else:
                        chosen_dir = dir2
                action = state_valid_directions.index(chosen_dir)
                print("random direction chosen", chosen_dir)
                assert action in self.get_valid_actions(s)
            else:
                print("if prev invalid, random")

                # i.e. I have got stuck, go to level 3 - "planning"?
                assert LVL_BY_NODE[s] == 6
                while LVL_BY_NODE[s] != 3:
                    s = self.take_action(s, 0)    # go to parent node
                    self.episode_state_traj.append(s)

                self.curr_directions = self.sample_directions()
                return self.choose_action_2(s, Q, epsilon, random_dir)
        return action, 1.0

    def choose_action_1(self, s, Q, random_dir):
        """
        Version 1

        if one can follow an action using prev direction
            do it
        else
            if random() <= epsilon
                sample a random direction and follow
            else
                take greedy

        """
        print("Version 1")
        if s == HOME_NODE:
            self.curr_directions = self.sample_directions()
            return 1, 1.0   # Go to node 0

        state_valid_directions = self.get_action_direction_mapping(s)
        dir1, dir2 = self.curr_directions
        print("state", s, "current", dir1, dir2, "- valid ones", state_valid_directions)

        if dir1 in state_valid_directions or dir2 in state_valid_directions:
            if random_dir:
                chosen_dir = random.choice(tuple({dir1, dir2}.intersection(state_valid_directions)))
            else:
                # choose directions in a priority order
                if dir1 in state_valid_directions:
                    chosen_dir = dir1
                else:
                    chosen_dir = dir2
            action = state_valid_directions.index(chosen_dir)
            print("random direction chosen", chosen_dir)
            assert action in self.get_valid_actions(s)
        else:

            # i.e. I have got stuck, go to level 3 - "planning"?
            assert LVL_BY_NODE[s] == 6
            while LVL_BY_NODE[s] != 3:
                print(s)
                s = self.take_action(s, 0)  # go to parent node
                self.episode_state_traj.append(s)

            self.curr_directions = self.sample_directions()
            print(s, "node level for sampling direction", LVL_BY_NODE[s])
            print("random")
            return self.choose_action_1(s, Q, random_dir)
        return action, 1.0

    def sample_directions(self):
        angle = np.random.randint(360)
        while angle % 45 == 0:
            angle = np.random.randint(360)
        d = self.get_valid_directions(angle)
        print("angle chosen", angle, d)
        return d

    @staticmethod
    def sample_duration():
        d = np.random.zipf(2, 1)[0]
        print("duration chosen", d)
        return d

    def generate_exploration_episode(self, alpha, gamma, lamda, epsilon, random_dir, version, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        print(self.nodemap)
        self.episode_state_traj = []
        e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        s = HOME_NODE  # Start from HOME
        a = 1   # Take action 1 at HOME NODE
        print("Starting at", s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(s)

            # acting
            a, a_prob = self.choose_action(s, Q, epsilon, random_dir, version)
            s_next = self.take_action(s, a)     # Take action
            print("action: ", a, f": {s} => {s_next}")

            # update Q values
            td_error = 0.0 + gamma * np.max([Q[s_next, a_i] for a_i in self.get_valid_actions(s_next)]) - Q[s, a]   # R = 0
            e[s, a] += 1
            for n in np.arange(self.S):
                Q[n, :] += alpha * td_error * e[n, :]
                e[n, :] = gamma * lamda * e[n, :]

            Q[s, a] = self.is_valid_state_value(Q[s, a])

            s = s_next
            if len(self.episode_state_traj)%1000 == 0:
                print("current state", s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')

        episode_state_trajs = break_simulated_traj_into_episodes(self.episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same
        print(episode_maze_trajs)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        alpha = params["alpha"]         # learning rate
        gamma = params["gamma"]         # discount factor
        lamda = params["lamda"]         # eligibility trace-decay
        epsilon = params["epsilon"]     # epsilon
        random_dir = params["random_dir"]  # epsilon
        version = params["version"]     # switch between algos to choose action
        initial_v = 1   # params["V"]

        print("alpha, gamma, lamda, epsilon, V, agentId", alpha, gamma, lamda, epsilon, initial_v, agentId)
        Q = np.zeros((self.S, self.A)) * initial_v  # Initialize state values
        Q[HOME_NODE, :] = 0
        Q[RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        LL = 0.0
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, epsilon, random_dir, version, MAX_LENGTH, Q)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)
            LL += episode_ll
        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes_state_trajs),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
            "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs)
        }
        return success, stats


if __name__ == '__main__':
    pass
