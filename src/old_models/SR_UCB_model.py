"""
SR-UCB model:
Switch between an SR model and a pure UCB model.
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from parameters import *
from BaseModel import BaseModel
#from MM_Traj_Utils import *

class SR_UCB(BaseModel):
    
    def __init__(self):
        super().__init__(self)
        self.T = self.get_transition_matrix()
        self.terminal_nodes = {HomeNode}
        
    def get_transition_matrix(self):
        n_states=len(ALL_MAZE_NODES)
        T = np.zeros((n_states, n_states))
        for state in np.arange(n_states):
            mask_valid_next_states = np.logical_and(np.logical_and(self.nodemap[state] != INVALID_STATE, \
                                                                   self.nodemap[state] != RWD_STATE), \
                                                    self.nodemap[state] != HOME_NODE)
            n_possibilities = float(sum(mask_valid_next_states))

            for next_state in self.nodemap[state][mask_valid_next_states]:
                T[state, next_state] = float(1 / n_possibilities)
        return T
    
    def get_initial_state(self):
        return 0
    
    #def get_action_probabilities(self, state, beta, V):
    #    if state == HomeNode:
    #        action_prob = [0, 1, 0]
    #    else:
    #        action_prob = TDLambdaXStepsRewardReceived.get_action_probabilities(self, state, beta, V)
    #    return action_prob
    
    def get_action_probabilities(self, state, beta, V):
        """
        Softmax policy to select action, a at current state, s
        """
        if state in lvl6_nodes:
            action_prob = [1, 0, 0]
        else:
            betaV = [np.exp(beta * V[int(future_state)]) for future_state in self.nodemap[state, :]]
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
                    print('Invalid action probabilities ', action_prob, betaV, state)

            if np.sum(action_prob) < 0.999:
                print('Invalid action probabilities, failed summing to 1: ',
                      action_prob, betaV, state)

        return action_prob
    
    def get_action_probabilities_greedy(self, state, V):
        """
        Select the action with the highest value, a at the current state, s 
        """
        action_prob = [0, 0, 0]
        if state in lvl6_nodes:
            action_prob = [1, 0, 0]
        else:
            V_actions = [V[int(future_state)] for future_state in self.nodemap[state, :]]
            max_ind = np.where(V_actions == np.max(V_actions))[0].astype(int)
            for ind in max_ind:
                action_prob[ind] = 1 / np.size(max_ind)
        return action_prob
            
    
    def generate_episode(self, beta, gamma, tau, c, t, N, SR_UCB, time_each_move=0.3, time_from_last_rwd=90):\
        
        def check_value_function(V):
            # TODO: When would the cases below happen? Add explanation
            if np.isnan(V[s]):
                print('Warning invalid state-value: ')
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5
            return V
        
        def model_next_step(time_from_last_rwd, SR_UCB, k = 0.01):#,max_switch_prob = 0.3):#the model for the next step
            if SR_UCB[0] == 0:#if the current model is SR
                if time_from_last_rwd == 0:#switch to the UCB model if the agent just gets the reward
                    return 0, 1
                else:
                    return 1, 0
            elif SR_UCB[0] == 1:#if the current model is UCB
                #if time_from_last_rwd == 0:
                    #SR_prob = 0
                #else:
                    #SR_prob_cumu_this = 1 - np.exp(-k * time_from_last_rwd)
                    #SR_prob_cumu_last = 1 - np.exp(-k * (time_from_last_rwd - 1))
                    #SR_prob = SR_prob_cumu_this - SR_prob_cumu_last
                #UCB_prob = 1 - SR_prob
                return 0.01, 0.99
            #return SR_prob, UCB_prob
            
        
        s = self.get_initial_state()
        episode_state_traj = [s]
        episode_maze_traj = [s]
        value_SR_hist = []
        value_UCB_hist = []
        model_hist = []
        M = np.linalg.inv(np.eye(self.T.shape[0]) - gamma * self.T)  # matrix M with expected future occupancies in each line
        #tau = .05
        rwd = np.zeros(self.S-2)
        
        #start from the home node
        N[HomeNode] += 1
        t[0] += 1
        while s not in self.terminal_nodes:
            
            V_SR = np.pad(M @ rwd, (0,2))  # pad because value function array includes values for all states, including home node and waterport_state
            value_SR_hist.append(V_SR)
            V_SR = check_value_function(V_SR)
            
            V_UCB = c * np.sqrt(np.log(t)/N)
            value_UCB_hist.append(V_UCB)

            if s == RWD_STATE:
                s_next = WATERPORT_NODE
                # R = 1
                rwd[WATERPORT_NODE] = 0  # reset the periodic drive to go to waterport
                time_from_last_rwd = 0
                # print(rwd)
            elif (s == WATERPORT_NODE) & (time_from_last_rwd > 0):# 90):
                s_next = RWD_STATE
                # R = 0
            else:
                SR_prob, UCB_prob = model_next_step(time_from_last_rwd, SR_UCB)
                SR_UCB[0] = np.random.choice(range(2), 1, p = [SR_prob, UCB_prob])[0]
                model_hist.append(SR_UCB[0])
                if SR_UCB[0] == 0:
                    V = V_SR
                    action_prob = self.get_action_probabilities_greedy(s, V)
                elif SR_UCB[0] == 1:
                    V = V_UCB
                    action_prob = self.get_action_probabilities(s, beta, V)
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]  # Choose action
                s_next = int(self.nodemap[s, a])           # Take action
                time_from_last_rwd += time_each_move #time
                # R = 0
                
            episode_state_traj.append(s_next)  # Record next state
            if s_next in ALL_VISITABLE_NODES:
                episode_maze_traj.append(s_next)  # Record next state

            N[s] += 1
            t[0] += 1

            # Update state-values
            rwd[WATERPORT_NODE] = rwd[WATERPORT_NODE] + tau*(1 - rwd[WATERPORT_NODE])

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

        return episode_state_traj, episode_maze_traj, value_SR_hist, value_UCB_hist, model_hist, time_from_last_rwd
    
    
    
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
        episodes_state_traj_all_mice = defaultdict(dict)
        episodes_maze_traj_all_mice = defaultdict(dict)
        episodes_value_SR_hist_all_mice = defaultdict(dict)
        episodes_value_UCB_hist_all_mice = defaultdict(dict)
        models_hist_all_mice = defaultdict(dict)
        episode_cap = 500   # max attempts at generating a bout episode
        success = 1

        for mouseID in sub_fits:
            
            #alpha = sub_fits[mouseID][0]
            beta = sub_fits[mouseID][0]             # softmax exploration - exploitation
            gamma = sub_fits[mouseID][1]            # discount factor
            tau = sub_fits[mouseID][2]              # reward rate
            c = sub_fits[mouseID][3]                # degree of exploration
            if len(sub_fits[mouseID]) > 4:
                time_each_move = sub_fits[mouseID][4] 
            else:
                time_each_move = 0.3
            if len(sub_fits[mouseID]) > 5:
                homestay = sub_fits[mouseID][5]
            else:
                homestay = 50

            print("beta, gamma, tau, c, mouseID, speed, nick",
                  beta, gamma, tau, c, mouseID, time_each_move, RewNames[mouseID])

            #V = np.random.rand(self.S+1)  # Initialize state-action values
            #V = np.zeros(self.S+1)
            #V[HomeNode] = 0     # setting action-values of maze entry to 0
            #V[RewardNode] = 0   # setting action-values of reward port to 0
            #V[WaterPortNode] = 0
            
            t = np.ones(1)
            N = np.ones(self.S+1)
            SR_UCB = np.ones(1)

            episodes_state_traj = []
            episodes_maze_traj = []
            episodes_value_SR_hist = []
            episodes_value_UCB_hist = []
            models_hist = []
            n_bouts = 0

            time_from_last_rwd = 90
            while t <= N_MOVES:
                episode_state_traj, episode_maze_traj,  value_SR_hist, value_UCB_hist, model_hist, time_from_last_rwd = self.generate_episode(beta, gamma, tau, c, t, N, SR_UCB, time_each_move, time_from_last_rwd)
                n_bouts += 1
                episodes_state_traj.extend(episode_state_traj)
                episodes_maze_traj.extend(episode_maze_traj)
                episodes_value_SR_hist.extend(value_SR_hist)
                episodes_value_UCB_hist.extend(value_UCB_hist)
                models_hist.extend(model_hist)
                time_from_last_rwd += np.random.exponential(1/homestay)

            episodes_state_traj_all_mice[mouseID] = episodes_state_traj
            episodes_maze_traj_all_mice[mouseID] = episodes_maze_traj
            episodes_value_SR_hist_all_mice[mouseID] = episodes_value_SR_hist
            episodes_value_UCB_hist_all_mice[mouseID] = episodes_value_UCB_hist
            models_hist_all_mice[mouseID] = models_hist
            stats[mouseID] = {
                "mouse": RewNames[mouseID],
                "n_moves": t,
                "n_bouts": n_bouts,
                #"state_values": V.round(4),
                "visit_frequency": N.round(4)
                # "invalid_initial_state_counts": invalid_initial_state_counts
            }
        return episodes_state_traj_all_mice, episodes_maze_traj_all_mice, episodes_value_SR_hist_all_mice, episodes_value_UCB_hist_all_mice, models_hist_all_mice, stats
