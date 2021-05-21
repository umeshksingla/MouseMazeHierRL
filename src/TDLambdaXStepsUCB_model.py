"""
TDLambdaXSteps model:
Take only the last X steps before a reward as training data
and add a UCB component for exploration.
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from parameters import *
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from MM_Traj_Utils import *

class TDLambdaXStepsUCBRewardReceived(TDLambdaXStepsRewardReceived):
    
    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, V, e, c, t, N):

        s = self.get_initial_state()
        episode_traj = []
        valid_episode = False
        while s not in self.terminal_nodes:

            episode_traj.append(s)  # Record current state

            if s != RewardNode:
                #print(c * np.sqrt(np.log(t)/N))
                action_prob = self.get_action_probabilities(s, beta, V + c * np.sqrt(np.log(t)/N))
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]  # Choose action
                s_next = int(self.nodemap[s, a])           # Take action
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
            else:
                s_next = WaterPortNode

            R = 1 if s == RewardNode else 0  # Observe reward

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            N[s] += 1
            t[0] += 1
            for node in np.arange(self.S):
                V[node] += alpha * td_error * e[node]
                e[node] = gamma * lamda * e[node]

            # print("V[s]", s, V[s])
            if np.isnan(V[s]):
                print('Warning invalid state-value: ', s, s_next, V[s], V[s_next], alpha, beta, gamma, R)
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5

            if s == RewardNode:
                print('Reward Reached. Recording episode.')
                valid_episode = True
                break

            if len(episode_traj) > MAX_LENGTH:
                print('Trajectory too long. Aborting episode.')
                valid_episode = False
                break

            s = s_next

        return valid_episode, [episode_traj]
    
    
    
    def simulate(self, sub_fits, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=100):
        """
        Model predictions (sample predicted trajectories) using fitted parameters sub_fits.
        You can use this to generate simulated data for parameter recovery as well.

        sub_fits: 
            dictionary of fitted parameters and log likelihood for each mouse. 
            {0:[alpha_fit, beta_fit, gamma_fit, lambda_fit, c_fit, LL], 1:[]...}
        MAX_LENGTH:
            max length of an episode to simulate
        N_BOUTS_TO_GENERATE:
            number of bout episodes to generate

        Returns:
        episodes_all_mice:
            dictionary of trajectories simulated by a model using fitted 
            parameters for specified mice. e.g. {0:[0,1,3..], 1:[]..}
        invalid_episodes_all_mice:
            dictionary of trajectories simulated by a model using fitted
            parameters for specified mice. e.g. {0:[0,1,3..], 1:[]..}
            where they didn't reach the reward in even MAX_LENGTH steps
        success:
            int. either 0 or 1 to flag when the model fails to generate
            trajectories adhering to certain bounds: fitted parameters, 
            number of episodes, trajectory length, etc.
        stats:
            dict. some stats on generated traj
        """

        stats = {}
        episodes_all_mice = defaultdict(dict)
        invalid_episodes_all_mice = defaultdict(dict)
        episode_cap = 500   # max attempts at generating a bout episode
        success = 1

        for mouseID in sub_fits:

            alpha = sub_fits[mouseID][0]    # learning rate
            beta = sub_fits[mouseID][1]     # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]    # discount factor
            lamda = sub_fits[mouseID][3]    # eligibility trace
            c = sub_fits[mouseID][4]

            print("alpha, beta, gamma, lamda, mouseID, c, nick",
                  alpha, beta, gamma, lamda, mouseID, c, RewNames[mouseID])

            V = np.random.rand(self.S+1)  # Initialize state-action values
            V[HomeNode] = 0     # setting action-values of maze entry to 0
            V[RewardNode] = 0   # setting action-values of reward port to 0

            e = np.zeros(self.S+1)    # eligibility trace vector for all states
            
            t = np.ones(1)
            N = np.ones(self.S+1)

            episodes = []
            invalid_episodes = []
            count_valid, count_total = 0, 1
            while len(episodes) < N_BOUTS_TO_GENERATE:

                # Back-up a copy of state-values to use in case the next episode has to be discarded
                V_backup = np.copy(V)
                e_backup = np.copy(e)
                t_backup = np.copy(t)
                N_backup = np.copy(N)

                # Begin generating episode
                episode_attempt = 0
                valid_episode = False
                while not valid_episode and episode_attempt <= episode_cap:
                    episode_attempt += 1
                    valid_episode, episode_traj = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, V, e, c, t, N)
                    count_valid += int(valid_episode)
                    count_total += 1
                    if valid_episode:
                        episodes.extend(episode_traj)
                    else:   # retry
                        V = np.copy(V_backup)   # TODO: maybe not discard invalid trajs?
                        e = np.copy(e_backup)
                        t = np.copy(t_backup)
                        N = np.copy(N_backup)
                        invalid_episodes.extend(episode_traj)
                    print("===")
                if not count_valid:
                    print('Failed to generate episodes for mouse ', mouseID)
                    success = 0
                    break
                # print("=============")
            episodes_all_mice[mouseID] = episodes
            invalid_episodes_all_mice[mouseID] = invalid_episodes
            stats[mouseID] = {
                "mouse": RewNames[mouseID],
                "MAX_LENGTH": MAX_LENGTH,
                "count_valid": count_valid,
                "count_total": count_total,
                "fraction_valid": round(count_valid/count_total, 3) * 100,
                # "invalid_initial_state_counts": invalid_initial_state_counts
            }
        return episodes_all_mice, invalid_episodes_all_mice, success, stats
