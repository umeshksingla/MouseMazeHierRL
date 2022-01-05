"""
EpsilonZGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency
import evaluation_metrics as em


class EpsilonZGreedy(EpsilonGreedy):

    def __init__(self, file_suffix='_EpsilonZGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.duration = self.sample_duration()

    def choose_action(self, s, Q, epsilon, *args, **kwargs):
        if self.duration == 0:
            if np.random.random() <= epsilon:
                self.duration = self.sample_duration()
                action = self.__random_action__(s)
                # print("random")
            else:
                action = self.__greedy_action__(s, Q)
                # print("greedy")
        else:
            if kwargs['prev_action'] in self.get_valid_actions(s):
                action = kwargs['prev_action']
                self.duration -= 1
                # print("prev")
            else:
                # print("prev invalid")
                self.duration = 0
                return self.choose_action(s, Q, epsilon)
        return action, 1.0

    @staticmethod
    def sample_duration():
        d = np.random.zipf(2, 1)[0]
        # print("duration chosen", d)
        return d

    def generate_exploration_episode(self, alpha, gamma, lamda, epsilon, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        episode_state_traj = []
        e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        s = HOME_NODE  # Start from HOME
        a = 1   # Take action 1 at HOME NODE
        print("Starting at", s)
        while len(episode_state_traj) <= MAX_LENGTH:
            assert s != RWD_STATE   # since it's pure exploration

            # Record current state
            episode_state_traj.append(s)

            # acting
            a, a_prob = self.choose_action(s, Q, epsilon, prev_action=a)
            s_next = self.take_action(s, a)     # Take action
            # print("action: ", a, f": {s} => {s_next}")

            # update Q values
            td_error = 0.0 + gamma * np.max([Q[s_next, a_i] for a_i in self.get_valid_actions(s_next)]) - Q[s, a]   # R = 0
            e[s, a] += 1
            for n in np.arange(self.S):
                Q[n, :] += alpha * td_error * e[n, :]
                e[n, :] = gamma * lamda * e[n, :]

            Q[s, a] = self.is_valid_state_value(Q[s, a])

            s = s_next
            if len(episode_state_traj)%1000 == 0:
                print("current state", s, "step", len(episode_state_traj))
                # print("Q", Q)

            # print("===========================")

        print('Max trajectory length reached. Ending this trajectory.')

        episode_state_trajs = break_simulated_traj_into_episodes(episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same

        return True, episode_state_trajs, episode_maze_trajs, 0.0


if __name__ == '__main__':
    pass
