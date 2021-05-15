"""
TDLambdaXSteps model:
Take only the last X steps before a reward as training data.
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from parameters import *
from BaseModel import BaseModel
from MM_Traj_Utils import *


class TDLambdaXStepsRewardReceived(BaseModel):

    def __init__(self, X = 20, file_suffix='_XStepsRewardReceivedTrajectories'):
        BaseModel.__init__(self, file_suffix)
        self.X = X
        self.terminal_nodes = {HomeNode, WaterPortNode}

    def extract_trajectory_data(self, orig_data_dir='../outdata/', save_dir=None):
        """
        save_dir: path to the directory where you want to save the pickled
        data object.
        """
        trajectory_data = []
        for mouseId, nickname in enumerate(RewNames):
            trajectory_data.append(self.__get_trajectory_data_by_nickname__(orig_data_dir, nickname))
        if save_dir:
            with open(os.path.join(save_dir, f'{self.file_suffix}.p'), 'wb') as f:
                pickle.dump(trajectory_data, f)
        return trajectory_data

    def __get_trajectory_data_by_nickname__(self, orig_data_dir, nickname):
        tf = LoadTrajFromPath(os.path.join(orig_data_dir, nickname + '-tf'))
        trajectory_data = []

        for boutId, reward_frames in enumerate(tf.re):
            bout_trajectory = tf.no[boutId]
            prev_idx = -1

            for each in reward_frames:
                start_frame, end_frame = each

                # Look for the current reward visit in the trajectory
                idx = np.searchsorted(bout_trajectory[:, 1], start_frame, side='left')

                # Take only the last X steps before the current reward visit.
                # (Taking into account if the previous reward visit was within X
                # steps Or if there haven't been X steps till now.)
                lastXsteps_ = bout_trajectory[max(prev_idx+1, idx-self.X):idx , 0]

                # Check if RWD_NODE was visited within X steps but no reward was received
                rew_node_visit_in_lastXsteps = np.where(lastXsteps_[:-1] == RewardNode)[0]
                last_rew_node_visit = rew_node_visit_in_lastXsteps[-1] if rew_node_visit_in_lastXsteps.size > 0 else -1

                # If yes, then consider only the later part of the trajectory
                traj = lastXsteps_[last_rew_node_visit+1:]
                # And append WaterPortNode at the end to denote the receipt of a reward.
                traj = np.append(traj, WaterPortNode)

                trajectory_data.append(traj)
                prev_idx = idx
        return trajectory_data

    def get_action_probabilities(self, state, beta, V, nodemap):
        # Use softmax policy to select action, a at current state, s
        if state in lv6_nodes:
            action_prob = [1, 0, 0]
        else:
            betaV = [np.exp(beta * V[int(val)]) for val in nodemap[state, :]]
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

    def get_initial_state(self):
        a=list(range(self.S))
        a.remove(28)
        a.remove(57)
        a.remove(115)
        a.remove(RewardNode)
        return np.random.choice(a)    # Random initial state

    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, V, e):

        nodemap = self.get_SAnodemap()

        s = self.get_initial_state()
        episode_traj = []
        valid_episode = False
        while s not in self.terminal_nodes:

            episode_traj.append(s)  # Record current state

            if s != RewardNode:
                action_prob = self.get_action_probabilities(s, beta, V, nodemap)
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]  # Choose action
                s_next = int(nodemap[s, a])           # Take action
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
            else:
                s_next = WaterPortNode

            R = 1 if s == RewardNode else 0  # Observe reward

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
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
                # print('Reward Reached. Recording episode.')
                valid_episode = True
                break

            if len(episode_traj) > MAX_LENGTH:
                # print('Trajectory too long. Aborting episode.')
                valid_episode = False
                break

            s = s_next

        return valid_episode, episode_traj

    def simulate(self, sub_fits, MAX_LENGTH=25):
        """
        Model predictions (sample predicted trajectories) using fitted parameters sub_fits.
        You can use this to generate simulated data for parameter recovery as well.

        sub_fits: 
            dictionary of fitted parameters and log likelihood for each mouse. 
            {0:[alpha_fit, beta_fit, gamma_fit, lambda_fit, LL], 1:[]...}
        MAX_LENGTH:
            max length of an episode to simulate

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
        N_BOUTS_TO_GENERATE = 100  # number of bout episodes to generate

        for mouseID in sub_fits:

            alpha = sub_fits[mouseID][0]    # learning rate
            beta = sub_fits[mouseID][1]     # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]    # discount factor
            lamda = sub_fits[mouseID][3]    # eligibility trace

            print("alpha, beta, gamma, lamda, mouseID, nick",
                  alpha, beta, gamma, lamda, mouseID, RewNames[mouseID])

            V = np.random.rand(self.S+1)  # Initialize state-action values
            V[HomeNode] = 0     # setting action-values of maze entry to 0
            V[RewardNode] = 0   # setting action-values of reward port to 0

            e = np.zeros(self.S)    # eligibility trace vector for all states

            episodes = []
            invalid_episodes = []
            count_valid, count_total = 0, 1
            while len(episodes) < N_BOUTS_TO_GENERATE:

                # Back-up a copy of state-values to use in case the next episode has to be discarded
                V_backup = np.copy(V)
                e_backup = np.copy(e)

                # Begin generating episode
                episode_attempt = 0
                valid_episode = False
                while not valid_episode and episode_attempt <= episode_cap:
                    episode_attempt += 1
                    valid_episode, episode_traj = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, V, e)
                    count_valid += int(valid_episode)
                    count_total += 1
                    if valid_episode:
                        episodes.append(episode_traj)
                    else:   # retry
                        V = np.copy(V_backup)   # TODO: maybe not discard invalid trajs?
                        e = np.copy(e_backup)
                        invalid_episodes.append(episode_traj)
                    # print("===")
                if not count_valid:
                    print('Failed to generate episodes for mouse ', mouseID)
                    success = 0
                    break
                # print("=============")
            episodes_all_mice[mouseID] = dict([(i, epi) for i, epi in enumerate(episodes)])
            invalid_episodes_all_mice[mouseID] = dict([(i, epi) for i, epi in enumerate(invalid_episodes)])
            stats[mouseID] = {
                "mouse": RewNames[mouseID],
                "MAX_LENGTH": MAX_LENGTH,
                "count_valid": count_valid,
                "count_total": count_total,
                "fraction_valid": round(count_valid/count_total, 3) * 100,
                # "invalid_initial_state_counts": invalid_initial_state_counts
            }
        return episodes_all_mice, invalid_episodes_all_mice, success, stats
