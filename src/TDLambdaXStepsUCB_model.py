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
    
    def __init__(self, X = 20, file_suffix='_XStepsRewardReceivedTrajectories'):
        TDLambdaXStepsRewardReceived.__init__(self, file_suffix)
        self.terminal_nodes = {HomeNode}
    
    def get_initial_state(self):
        return 0
    
    #def get_action_probabilities(self, state, beta, V):
    #    if state == HomeNode:
    #        action_prob = [0, 1, 0]
    #    else:
    #        action_prob = TDLambdaXStepsRewardReceived.get_action_probabilities(self, state, beta, V)
    #    return action_prob
    
    def generate_episode(self, alpha, beta, gamma, lamda, V, e, c, t, N, TimeEachMove=1):

        s = self.get_initial_state()
        episode_traj = []
        episode_traj_wo_rew = []
        #valid_episode = False
        TimeFromLastReward = 90
        
        #start from the home node
        N[HomeNode] += 1
        t[0] += 1
        while s not in self.terminal_nodes:

            episode_traj.append(s)  # Record current state
            if s != WaterPortNode:
                episode_traj_wo_rew.append(s)#Do not Record the Water Port Node
            
            if s == WaterPortNode:
                s_next = RewardNode
                R = 1
                TimeFromLastReward = 0
            elif (s == RewardNode) & (TimeFromLastReward >= 90):
                s_next = WaterPortNode
                R = 0
            else:
                #print(c * np.sqrt(np.log(t)/N))
                action_prob = self.get_action_probabilities(s, beta, V + c * np.sqrt(np.log(t)/N))
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]  # Choose action
                s_next = int(self.nodemap[s, a])           # Take action
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
                TimeFromLastReward += TimeEachMove #time
                R = 0

            #R = 1 if s == RewardNode else 0  # Observe reward

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            N[s] += 1
            t[0] += 1
            for node in np.arange(self.S):
                V[node] += alpha * td_error * e[node]
                e[node] = gamma * lamda * e[node]

            #print('Water Port Node: ' + str(V[WaterPortNode]))
            # print("V[s]", s, V[s])
            if np.isnan(V[s]):
                print('Warning invalid state-value: ', s, s_next, V[s], V[s_next], alpha, beta, gamma, R)
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5

            if s == WaterPortNode:
                print('Reward Reached')
            #if s == RewardNode:
            #    print('Reward Reached. Recording episode.')
            #    valid_episode = True
            #    break

            #if len(episode_traj) > MAX_LENGTH:
            #    print('Trajectory too long. Aborting episode.')
            #    valid_episode = False
            #    break

            s = s_next
            
        #if s == HomeNode:
        #    episode_traj.append(s)
        
        #e[s] += 1
        N[s] += 1

        return [episode_traj], [episode_traj_wo_rew]
    
    
    
    def simulate(self, sub_fits, N_MOVES = 10000):#:MAX_LENGTH=25, N_BOUTS_TO_GENERATE=100):
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
        episodes_all_mice_wo_rew:
            is similar to episodes_all_mice but excludes the reward nodes in order for
            plotting

        stats:
            dict. some stats on generated traj
        """

        stats = {}
        episodes_all_mice = defaultdict(dict)
        episodes_all_mice_wo_rew = defaultdict(dict)
        episode_cap = 500   # max attempts at generating a bout episode
        success = 1

        for mouseID in sub_fits:

            alpha = sub_fits[mouseID][0]    # learning rate
            beta = sub_fits[mouseID][1]     # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]    # discount factor
            lamda = sub_fits[mouseID][3]    # eligibility trace
            c = sub_fits[mouseID][4]        # degree of exploration
            if len(sub_fits[mouseID]) > 5:
                TimeEachMove = sub_fits[mouseID][5] 
            else:
                TimeEachMove = 1

            print("alpha, beta, gamma, lamda, mouseID, c, speed, nick",
                  alpha, beta, gamma, lamda, mouseID, c, TimeEachMove, RewNames[mouseID])

            V = np.random.rand(self.S+1)  # Initialize state-action values
            #V = np.zeros(self.S+1)
            V[HomeNode] = 0     # setting action-values of maze entry to 0
            V[RewardNode] = 0   # setting action-values of reward port to 0
            V[WaterPortNode] = 0

            e = np.zeros(self.S+1)    # eligibility trace vector for all states
            
            t = np.ones(1)
            N = np.ones(self.S+1)

            episodes = []
            episodes_wo_rew = []
            n_bouts = 0

            while t <= N_MOVES:
                episode_traj, episode_traj_wo_rew = self.generate_episode(alpha, beta, gamma, lamda, V, e, c, t, N, TimeEachMove)
                n_bouts += 1
                episodes.extend(episode_traj)
                episodes_wo_rew.extend(episode_traj_wo_rew)

            episodes_all_mice[mouseID] = episodes
            episodes_all_mice_wo_rew[mouseID] = episodes_wo_rew
            stats[mouseID] = {
                "mouse": RewNames[mouseID],
                "n_moves": t,
                "n_bouts": n_bouts,
                "state_values": V.round(4),
                "visit_frequency": N.round(4)
                # "invalid_initial_state_counts": invalid_initial_state_counts
            }
        return episodes_all_mice, episodes_all_mice_wo_rew, stats
