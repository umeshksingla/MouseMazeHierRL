"""
TDLambdaXStepsPrevNode model:
Take only the last X steps before a reward as training data and each state is
a combination of current node and prev node (in an attempt to include direction)
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from parameters import *
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from MM_Traj_Utils import *


class TDLambdaXStepsPrevNodeRewardReceived(TDLambdaXStepsRewardReceived):

    def __init__(self, X = 20, file_suffix='_XStepsRewardReceivedTrajectories'):
        TDLambdaXStepsRewardReceived.__init__(self, X, file_suffix)
        self.nodes = 130
        self.S = self.nodes * self.nodes

        self.h, self.inv_h = dict(), dict()
        self.construct_node_tuples_to_number_map()

        self.home_terminal_state = self.get_number_from_node_tuple((HomeNode, InvalidState))
        self.reward_terminal_state = self.get_number_from_node_tuple((RewardNode, WaterPortNode))
        self.RewardTupleState = self.get_number_from_node_tuple((57, RewardNode))
        self.terminal_nodes = {self.home_terminal_state, self.reward_terminal_state}

    def get_action_probabilities(self, state, beta, V):
        # Use softmax policy to select action, a at current state, s

        curr = self.get_node_tuple_from_number(state)[1]
        if curr in lvl6_nodes:
            action_prob = [1, 0, 0]
        else:
            betaV = [np.exp(beta * V[self.get_number_from_node_tuple((curr, val))])
                     for val in self.nodemap[curr, :]]
            action_prob = []
            for action in np.arange(self.A):
                if np.isinf(betaV[action]):  # TODO: ?
                    action_prob.append(1)
                elif np.isnan(betaV[action]):
                    action_prob.append(0)
                else:
                    action_prob.append(betaV[action] / np.nansum(betaV))

            # Check for invalid probabilities
            for i in action_prob:
                if np.isnan(i):
                    raise Exception('Invalid action probabilities ', action_prob, betaV, state)

            if np.sum(action_prob) < 0.999:
                raise Exception('Invalid action probabilities, failed summing to 1: ',
                      action_prob, betaV, state)

        return action_prob

    def get_initial_state(self) -> int:
        return self.get_number_from_node_tuple((HomeNode, 0))
        # a=list(range(self.S))
        # a.remove(28)
        # a.remove(57)
        # a.remove(115)
        # a.remove(RewardNode)
        # return np.random.choice(a)    # Random initial state

    def construct_node_tuples_to_number_map(self):
        """
        Transition i->j
        """
        c = 0
        for i in range(-1, self.nodes):
            for j in range(-1, self.nodes):
                self.h[c] = (i, j)
                self.inv_h[(i, j)] = c
                c += 1
        # print(self.h, self.inv_h, c)
        return

    def get_node_tuple_from_number(self, c: int) -> tuple:
        return self.h[c]

    def get_number_from_node_tuple(self, t: tuple) -> int:
        return self.inv_h[t]

    def convert_nodes_to_state_traj(self, t):
        return [self.get_number_from_node_tuple((i, j)) for i, j in zip(t, t[1:])]

    def convert_state_traj_to_node(self, l):
        # print(l)
        return [self.get_node_tuple_from_number(n)[0] for n in l] +\
               [self.get_node_tuple_from_number(l[-1])[1]]

    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, V, e):

        def take_action(s: int, a: int) -> int:
            prev, curr = self.get_node_tuple_from_number(s)
            s_ = self.get_number_from_node_tuple((curr, self.nodemap[curr, a]))
            return s_

        s = self.get_initial_state()
        LL = 0.0
        first_reward = -1
        episode_traj = []
        # valid_episode = False
        # while s not in self.terminal_nodes:
        while True:
            if len(episode_traj)%100 == 0:
                print("current state", self.get_node_tuple_from_number(s)[1], "step", len(episode_traj))

            episode_traj.append(s)  # Record current state

            if s in self.terminal_nodes:
                # print("entering again", s, self.get_node_tuple_from_number(s))
                # episode_traj.append("e")
                s = self.get_initial_state()

            if s != self.RewardTupleState:
                action_prob = self.get_action_probabilities(s, beta, V)
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]    # Choose action
                s_next = take_action(s, a)  # Take action
                LL += np.log(action_prob[a])
                # print("s, s_next, a, action_prob", self.get_node_tuple_from_number(s), self.get_node_tuple_from_number(s_next), a, action_prob)
            else:
                s_next = self.reward_terminal_state

            R = 1 if s == self.RewardTupleState else 0   # Observe reward

            assert isinstance(s, int)
            assert isinstance(s_next, int)

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            for n in np.arange(self.S):
                V[n] += alpha * td_error * e[n]
                e[n] = gamma * lamda * e[n]

            # print("V reward", V[self.RewardTupleState])

            if np.isnan(V[s]):
                print('Warning invalid state-value: ', s, s_next, V[s], V[s_next], alpha, beta, gamma, R)
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5

            if s == self.RewardTupleState:
                print('Reward Reached!')
                if first_reward == -1:
                    first_reward = len(episode_traj)
                    print("First reward:", len(episode_traj))
            #     episode_traj.append("r")
                # valid_episode = True

            if len(episode_traj) > MAX_LENGTH + first_reward:
                print('Trajectory too long. Aborting episode.')
                # valid_episode = True
                break

            s = s_next

        # print(episode_traj)
        maze_episode_traj = self.convert_state_traj_to_node(episode_traj)

        # split at HomeNode or RewardNode
        # print(maze_episode_traj)
        episodes = []
        epi = []
        for i in maze_episode_traj:
            if i == HomeNode:
                epi.append(i)
                if len(epi) > 2:
                    episodes.append(epi)
                epi = []
            elif i == RewardNode:
                epi.append(i)
                if len(epi) > 2:
                    episodes.append(epi)
                epi = []
            else:
                epi.append(i)
        if epi:
            episodes.append(epi)

        # if len(maze_episode_traj) >= 5:
        #     return True, maze_episode_traj
        # else:
        #     return False, maze_episode_traj
        return True, episodes, LL

    def simulate(self, sub_fits, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):

        stats = {}
        episodes_all_mice = defaultdict(dict)
        loglikelihoods = defaultdict(int)
        episode_cap = 500   # max attempts at generating a bout episode
        success = 1

        for mouseID in sub_fits:

            alpha = sub_fits[mouseID][0]    # learning rate
            beta = sub_fits[mouseID][1]     # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]    # discount factor
            lamda = sub_fits[mouseID][3]    # eligibility trace

            print("alpha, beta, gamma, lamda, mouseID, nick",
                  alpha, beta, gamma, lamda, mouseID, RewNames[mouseID])

            # V = np.ones(self.S+1)*0.1  # Initialize state-action values
            V = np.random.rand(self.S + 1)
            V[self.home_terminal_state] = 0     # setting action-values of maze entry to 0
            V[self.reward_terminal_state] = 0

            e = np.zeros(self.S+1)    # eligibility trace vector for all states

            episodes = []
            invalid_episodes = []
            count_valid, count_total = 0, 1
            LL = 0.0
            while len(episodes) < N_BOUTS_TO_GENERATE:

                # Begin generating episode
                episode_attempt = 0
                valid_episode = False
                while not valid_episode and episode_attempt <= episode_cap:
                    episode_attempt += 1
                    valid_episode, episode_traj, episode_ll = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, V, e)
                    if valid_episode:
                        episodes.extend(episode_traj)
                        LL += episode_ll
                    else:   # retry
                        raise Exception("Inside invalid episode")
                    print("===")
                # print("=============")
            episodes_all_mice[mouseID] = episodes
            loglikelihoods[mouseID] = LL
            stats[mouseID] = {
                "mouse": RewNames[mouseID],
                "MAX_LENGTH": MAX_LENGTH,
                "count_total": count_total,
                "V": V,
            }
            print(V)
            print(len(V))
        return episodes_all_mice, [], loglikelihoods, success, stats


def test1():
    a = TDLambdaXStepsPrevNodeRewardReceived()
    t = [10, 21, 44, 90, 44, 89, 44, 90, 44, 21, 43]
    l = [a.inv_h[(i, j)] for i, j in zip(t, t[1:])]
    new = [a.h[n][0] for n in l] + [a.h[l[-1]][1]]
    assert new == t


if __name__ == '__main__':
    test1()
