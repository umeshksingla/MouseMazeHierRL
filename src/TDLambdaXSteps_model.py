"""
"""
import numpy as np
import pickle
import os

from parameters import InvalidState, RewNames, RewardNode
from TD_model import TD
from MM_Traj_Utils import *


class TDLambdaXStepsRewardReceived(TD):

    def __init__(self, file_suffix='_XStepsRewardReceivedTrajectories'):
        TD.__init__(self, file_suffix)
        self.X = 20

    def extract_and_save_trajectory_data(self, output_dir):
        """
        """
#         N = 10
#         TrajSize = 3000
#         TrajNo = 20
#         TrajS = np.ones((N,TrajNo,TrajSize)) * InvalidState
        trajectory_data = []
        for mouseId, nickname in enumerate(RewNames):
            trajectory_data.append(self.__get_trajectory_data_by_nickname__(nickname))
#             break
        with open(os.path.join(output_dir, f'{self.file_suffix}.p'),'wb') as f:
            pickle.dump(trajectory_data, f)
        return trajectory_data

    def __get_trajectory_data_by_nickname__(self, nickname):
        tf = LoadTraj(nickname + '-tf')
        
        trajectory_data = []

        for boutId, reward_frames in enumerate(tf.re):
            bout_trajectory = tf.no[boutId]
            prev_idx = -1
            
#             print(f"boutId {boutId}")
            for each in reward_frames:
                start_frame, end_frame = each
                
                # Look for the current reward visit in the trajectory
                idx =  np.searchsorted(bout_trajectory[:, 1], start_frame, side='left') 
                
                # Take the last X steps from the trajectory till the current reward visit
                # Or if previous reward visit was within X steps, then max of those
                # Or if there haven't been X steps till now
                lastXsteps_ = bout_trajectory[max(prev_idx+1, idx-self.X):idx , 0]

                # Check if RWD_NODE was visited within X steps but no reward was received
                rew_node_visit_in_lastXsteps = np.where(lastXsteps_[:-1] == RewardNode)[0]
                last_rew_node_visit = rew_node_visit_in_lastXsteps[-1] if rew_node_visit_in_lastXsteps.size > 0 else -1

                # If yes, then consider only the later part of the trajectory
                traj = lastXsteps_[last_rew_node_visit+1:]
                trajectory_data.append(traj)
                prev_idx = idx
#                 print(f" ===> {idx} {each} {traj} {rew_node_visit_in_lastXsteps}")
        return trajectory_data

    def simulate(self, sub_fits, orig_data):
        pass
#         '''
#         Model predictions (sample predicted trajectories) using fitted parameters sub_fits.

#         You can use this to generate simulated data for parameter recovery as well.

#         sub_fits: dictionary of fitted parameters and log likelihood for each rewarded mouse. 
#                        best_sub_fits{0:[alpha_fit, beta_fit, gamma_fit, LL], 1:[]...., 9:[]}
#         orig_data: str, file path for real trajectory data

#         Returns: state_hist_AllMice, dictionary of trajectories simulated by a model using fitted parameters for all Rew mice
#                  state_hist_AllMice{0:[0,1,3..], 1:[]..}

#                  int valid_bouts, counter to record the number of bouts that were simulated corresponding to real trajectory
#                                   data used for fitting

#                  int success, either 0 or 1 to flag when the model fails to generate simulated trajectories adhering
#                               to certain bounds: fitted parameters, number of episodes, trajectory length
#         '''
#             # Set environment parameters

#         S = 127
#         A = 3
#         RT = 1
#         nodemap = self.get_SAnodemap(S, A)  # rows index the current state, columns index 3 available neighboring states
#         state_hist_AllMice = {}
#         valid_bouts = []
#         avg_count = 1
#         episode_cap = 500
#         value_cap = 1e5
#         success = 1

#         TrajS = pickle.load(open(orig_data,'rb')).astype(int)

#         for mouseID in np.arange(10):
#             # Set model parameters
#             alpha = sub_fits[mouseID][0]  # learning rate
#             beta = sub_fits[mouseID][1]   # softmax exploration - exploitation
#             gamma = sub_fits[mouseID][2]
#             R = 0

#             # number of episodes to train over which are real bouts beginning at node 0 
#             # and exploring deeper into the maze, which is > than a trajectory length of 2 (node 0 -> node 127)
#             valid_boutID = np.where(TrajS[mouseID,:,2]!=InvalidState)[0]
#             N = len(valid_boutID)
#             valid_bouts.extend([N])

#             for count in np.arange(avg_count):
#                 # Initialize model parameters
#                 V = np.random.rand(S+1)  # state-action values
#                 V[HomeNode] = 0  # setting action-values of maze entry to 0
#                 V[RewardNode] = 0  # setting action-values of reward port to 0
#                 state_hist_mouse = {}
#                 R_visits = 0

#                 for n in np.arange(N):
#                     valid_episode = False
#                     episode_attempt = 0

#                     # Extract from real mouse trajectory the terminal node in current bout and trajectory length
#                     end = np.where(TrajS[mouseID,valid_boutID[n]]==InvalidState)[0][0]
#                     valid_traj = TrajS[mouseID,valid_boutID[n],0:end]

#                     # Back-up a copy of state-values to use in case the next episode has to be discarded
#                     V_backup = np.copy(V)

#                     # Begin episode
#                     while not valid_episode and episode_attempt < episode_cap:
#                         # Initialize starting state,s0 to node 0
#                         s = 0
#                         state_hist = []

#                         while s!=HomeNode and s!=RewardNode:
#                             # Record current state
#                             state_hist.extend([s])

#                             # Use softmax policy to select action, a at current state, s
#                             if s in lv6_nodes:
#                                 aprob = [1,0,0]
#                             else:
#                                 betaV = [np.exp(beta*V[int(val)]) for val in nodemap[s,:]]
#                                 aprob = []
#                                 for atype in np.arange(3):
#                                     if np.isinf(betaV[atype]):
#                                         aprob.extend([1])
#                                     elif np.isnan(betaV[atype]):
#                                         aprob.extend([0])
#                                     else:
#                                         aprob.extend([betaV[atype]/np.nansum(betaV)])

#                             # Check for invalid probabilities
#                             for i in aprob:
#                                 if np.isnan(i):
#                                     print('Invalid action probabilities ', aprob, betaV, s)
#                                     print(alpha, beta, gamma, mouseID, n)
#                             if np.sum(aprob) < 0.999:
#                                 print('Invalid action probabilities, failed summing to 1: ', aprob, betaV, s)
#                             a = np.random.choice([0,1,2],1,p=aprob)[0]

#                             # Take action, observe reward and next state
#                             sprime = int(nodemap[s,a])
#                             if sprime == RewardNode:
#                                 R = 1  # Receive a reward of 1 when transitioning to the reward port
#                             else:
#                                 R = 0

#                             # Update action-value of previous state value, V[s]
#                             V[s] += alpha * (R + gamma*V[sprime] - V[s])
#                             if np.isnan(V[s]):
#                                 print('Warning invalid state-value: ', s, sprime, V[s], V[sprime], alpha, beta, gamma, R)
#                             elif np.isinf(V[s]):
#                                 print('Warning infinite state-value: ', V)
#                             elif V[s]>value_cap:
#                                 #print('Warning state value exceeded upper bound. Might approach infinity')
#                                 V[s] = value_cap

#                             # Shift state values for the next time step
#                             s = sprime

#                             # Check whether to abort the current episode
#                             if len(state_hist) > len(valid_traj):
#                                 #print('Trajectory too long. Aborting episode')
#                                 break
#                         state_hist.extend([s])

#                         # Find actual end node for mouse trajectory in the current bout/episode
#                         if s == valid_traj[-1]:
#                             if (len(state_hist) > 2) and (len(state_hist) > 0.5 * len(valid_traj)) and (len(state_hist) <= len(valid_traj)):
#                                 state_hist_mouse[n] = state_hist
#                                 valid_episode = True
#                         else:
#                             R = 0
#                             V = np.copy(V_backup)
#                             #print('Rejecting episode of length: ', len(state_hist), ' for mouse ', mouseID, ' bout ', valid_boutID[n], ' traj length ', len(valid_traj))
#                             episode_attempt += 1

#                     if episode_attempt >= episode_cap:
#                         print('Failed to generate episodes for mouse ', mouseID, ' with parameter set: ', alpha, beta, gamma)
#                         success = 0
#                         break
#                 state_hist_AllMice[mouseID] = state_hist_mouse

#         return state_hist_AllMice, valid_bouts, success
