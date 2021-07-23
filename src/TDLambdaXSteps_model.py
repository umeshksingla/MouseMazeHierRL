"""
TDLambdaXSteps model:
Take only the last X steps before a reward as training data.
"""

import numpy as np
import pickle
import os
import sys
from collections import defaultdict

from BaseModel import BaseModel
from MM_Traj_Utils import LoadTrajFromPath
from parameters import HOME_NODE, RWD_STATE, LVL_6_NODES, WATERPORT_NODE, RewNames
from utils import break_simulated_traj_into_episodes


class TDLambdaXStepsRewardReceived(BaseModel):

    def __init__(self, X = 20, file_suffix='_XStepsRewardReceivedTrajectories'):
        """
        :param X (int): number of steps before the reward was consumed at the waterport
        :param file_suffix:  # TODO explain what this argument is for
        """
        BaseModel.__init__(self, file_suffix)
        self.X = X

    def __get_trajectory_data_by_nickname__(self, orig_data_dir, nickname):
        """
        Returns only the last X steps before a reward as training data.
        """
        print(f"Returning only the last X steps before every reward for {nickname}.")
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

                # Check if WATERPORT_NODE was visited within X steps but no reward was received
                rew_node_visit_in_lastXsteps = np.where(lastXsteps_[:-1] == WATERPORT_NODE)[0]
                last_rew_node_visit = rew_node_visit_in_lastXsteps[-1] if rew_node_visit_in_lastXsteps.size > 0 else -1

                # If yes, then consider only the later part of the trajectory
                traj = lastXsteps_[last_rew_node_visit+1:]
                # And append WaterPortNode at the end to denote the receipt of a reward.
                traj = np.append(traj, RWD_STATE)

                trajectory_data.append(traj.tolist())
                prev_idx = idx
        return trajectory_data

    def get_action_probabilities(self, state, beta, V):
        """
        Softmax policy to select action, a at current state, s
        :param state:
        :param beta:
        :param V:
        :return: list of action_probabilities for three possible actions []
        """
        if state in LVL_6_NODES:
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

    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, V, et):
        """

        :param alpha:
        :param beta:
        :param gamma:
        :param lamda:
        :param MAX_LENGTH:
        :param V:
        :param et:
        :return: valid_episode, episodes, LL
        """

        s = self.get_initial_state()
        episode_traj = []
        LL = 0.0
        first_reward = -1
        while True:

            episode_traj.append(s)  # Record current state

            if s in self.terminal_nodes:
                print(f"reached {s}, entering again")
                s = self.get_initial_state()

            if s != WATERPORT_NODE:
                a, a_prob = self.choose_action(s, beta, V)  # Choose action
                s_next = self.take_action(s, a)  # Take action
                LL += np.log(a_prob)    # Update log likelihood
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
            else:
                s_next = RWD_STATE

            R = 1 if s == WATERPORT_NODE else 0  # Observe reward

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            # et[s] += 1
            for node in np.arange(self.S):
                V[node] += alpha * td_error * et[node]
                # et[node] = gamma * lamda * et[node]

            V[s] = self.is_valid_state_value(V[s])

            if s == WATERPORT_NODE:
                print('Reward Reached!')
                if first_reward == -1:
                    first_reward = len(episode_traj)
                    print("First reward:", len(episode_traj))

            if len(episode_traj) > MAX_LENGTH + first_reward:
                print('Trajectory too long. Aborting episode.')
                break

            s = s_next

            if len(episode_traj)%100 == 0:
                print("current state", s, "step", len(episode_traj))

        episodes = break_simulated_traj_into_episodes(episode_traj)
        return True, episodes, LL

    def simulate(self, agentId, sub_fits, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        """
        Model predictions (sample predicted trajectories) using fitted parameters sub_fits.
        You can use this to generate simulated data for parameter recovery as well.

        agentId:
            any agent identifier
        sub_fits: 
            dictionary of fitted parameters and log likelihood for each mouse. 
            {0:[alpha_fit, beta_fit, gamma_fit, lambda_fit, LL], 1:[]...}
        MAX_LENGTH:
            max length of a trajectory to simulate
        N_BOUTS_TO_GENERATE:
            number of bout episodes to generate

        Returns:
        success:
            int. either 0 or 1 to flag when the model fails to generate
            trajectories adhering to certain bounds: fitted parameters, 
            number of episodes, trajectory length, etc.
        stats:
            dict. some stats on generated traj along with episodes
        """

        success = 1

        alpha = sub_fits[0]    # learning rate
        beta = sub_fits[1]     # softmax exploration - exploitation
        gamma = sub_fits[2]    # discount factor
        lamda = sub_fits[3]    # eligibility trace

        print("alpha, beta, gamma, lamda, agentId",
              alpha, beta, gamma, lamda, agentId)

        V = np.random.rand(self.S+1)  # Initialize state values
        V[HOME_NODE] = 0     # setting state-value of maze entry to 0
        V[RWD_STATE] = 0
        et = np.zeros(self.S+1)    # eligibility trace vector for all states

        all_episodes = []
        LL = 0.0
        while len(all_episodes) < N_BOUTS_TO_GENERATE:
            # Begin generating a bout
            _, episodes, episode_ll = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, V, et)
            all_episodes.extend(episodes)
            LL += episode_ll

        stats = {
            "agentId": agentId,
            "episodes": all_episodes,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes),
            "V": V,
        }
        return success, stats
