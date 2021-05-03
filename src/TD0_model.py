"""
"""

import numpy as np
import pickle

from parameters import *
from TD_model import TD
from MM_Traj_Utils import *


class TD0MultipleVisits(TD):
    
    def __init__(self, main_dir):
        TD.__init__(self, main_dir)

    def get_trajectory_data(self):
        """
        Extracting trajectories of each rewarded mouse up until the first drink is obtained at the reward node

        Segments of real rewarded mice trajectories are extracted. This data will be used for model fitting.

        Returns: 
        1. Saves trajectories as a pickle file
        'rewMICE_multiple_visit.p' in the stan_data_dir directory
        : ndarray[(N, TrajNo, TrajSize), int]
        
        2. Saves the number of unrewarded water port visits in 'nonRew_RVisits.p'
        : ndarray[(N, TrajNo), int]
        """

        N = 10
        TrajSize = 3000
        TrajNo = 20
        TrajS = np.ones((N,TrajNo,TrajSize)) * InvalidState
        
        nonRew_RVisits = np.zeros((N,TrajNo), dtype=int)

        for mouseID, nickname in enumerate(RewNames):
            tf = LoadTraj(nickname+'-tf')
            reward_found = False
            for boutID, reFrames in enumerate(tf.re):
                waterport_visit_frames = tf.no[boutID][np.where(tf.no[boutID][:,0]==116)[0],1]
                if len(reFrames) != 0:
                    # find the number of steps till the first reward
                    for step, entry in enumerate(tf.no[boutID]):
                        node, frame = entry
                        if node==116:
                            wID = np.where(waterport_visit_frames==frame)[0][0]
                            if len(waterport_visit_frames)==1 and waterport_visit_frames[wID] <= reFrames[0][0]:
                                reFirst = step
                                TrajS[mouseID,boutID,0:reFirst+1] = tf.no[boutID][0:reFirst+1,0] 
                            elif waterport_visit_frames[-1]==frame and waterport_visit_frames[wID] <= reFrames[0][0]:
                                reFirst = step
                                TrajS[mouseID,boutID,0:reFirst+1] = tf.no[boutID][0:reFirst+1,0] 
                            elif waterport_visit_frames[wID] <= reFrames[0][0] and reFrames[0][0] <= waterport_visit_frames[wID+1]:
                                reFirst = step
                                TrajS[mouseID,boutID,0:reFirst+1] = tf.no[boutID][0:reFirst+1,0] 
                            reward_found = True
                            break
                        else:
                            TrajS[mouseID,boutID,step] = tf.no[boutID][step,0]

                else:
                    TrajS[mouseID,boutID,0:len(tf.no[boutID][:,0])] = tf.no[boutID][:,0]

                if reward_found:
                    break

            # Save number of unsuccessful reward node visits
            for boutID in np.arange(TrajNo):
                nonRew_RVisits[mouseID,boutID] = len(np.where(TrajS[mouseID,boutID,:]==RewardNode)[0])

                # Checking if the bout is rewarded
                if TrajS[mouseID,boutID+1,0] == InvalidState:
                    nonRew_RVisits[mouseID,boutID] -= 1
                    break
        pickle.dump(nonRew_RVisits, open(self.stan_data_dir + 'nonRew_RVisits.p','wb'))
        pickle.dump(TrajS, open(self.stan_data_dir + 'rewMICE_multiple_visit.p','wb'))  
        return


class TD0FirstVisit(TD):
    
    def __init__(self, main_dir):
        TD.__init__(self, main_dir)
        self.file_prefix = '_first_visit'

    def get_trajectory_data(self):
        """
        Extracting trajectories of each rewarded mouse up until the first visit to the reward node

        Segments of real rewarded mice trajectories are extracted. This data will be used for model fitting.

        Returns: 
        1. Saves trajectories as a pickle file
        'rewMICE_first_visit.p' in the stan_data_dir directory: ndarray[(N, TrajNo, TrajSize), int]
        """

        N = 10
        TrajSize = 3000
        TrajNo = 20
        TrajS = np.ones((N,TrajNo,TrajSize)) * InvalidState

        for mouseID, nickname in enumerate(RewNames):
            tf = LoadTraj(nickname+'-tf')
            reward_found = False
            for boutID in np.arange(len(tf.no)):
                # find the number of steps till the first reward
                for step, entry in enumerate(tf.no[boutID]):
                    node, frame = entry
                    if node==116:
                        TrajS[mouseID,boutID,step] = tf.no[boutID][step,0]
                        reward_found = True
                        break
                    else:
                        TrajS[mouseID,boutID,step] = tf.no[boutID][step,0]
                if reward_found:
                    break
        file_name = self.stan_data_dir + f'rewMICE{self.file_prefix}.p'
        with open(file_name, 'wb') as f:
            pickle.dump(TrajS, f)
        print(f"Trajectory Data written to {file_name}")
        return
    
    def simulate(self, sub_fits, orig_data):
        '''
        Model predictions (sample predicted trajectories) using fitted parameters sub_fits.

        You can use this to generate simulated data for parameter recovery as well.

        sub_fits: dictionary of fitted parameters and log likelihood for each rewarded mouse. 
                  sub_fits{0:[alpha_fit, beta_fit, gamma_fit, LL], 1:[]...., 9:[]}
        orig_data: str, file path for real trajectory data

        Returns: state_hist_AllMice, dictionary of trajectories simulated by a model using fitted parameters for all Rew mice
                 state_hist_AllMice{0:[0,1,3..], 1:[]..}

                 int valid_bouts, counter to record the number of bouts that were simulated corresponding to real trajectory
                                  data used for fitting

                 int success, either 0 or 1 to flag when the model fails to generate simulated trajectories adhering
                              to certain bounds: fitted parameters, number of episodes, trajectory length
        '''
            # Set environment parameters

        RT = 1
        nodemap = self.get_SAnodemap()  # rows index the current state, columns index 3 available neighboring states
        state_hist_AllMice = {}
        valid_bouts = []
        avg_count = 1
        episode_cap = 500
        value_cap = 1e5
        success = 1

        TrajS = pickle.load(open(orig_data, 'rb')).astype(int)

        for mouseID in np.arange(10):
            # Set model parameters
            alpha = sub_fits[mouseID][0]  # learning rate
            beta = sub_fits[mouseID][1]   # softmax exploration - exploitation
            gamma = sub_fits[mouseID][2]
            R = 0

            # number of episodes to train over which are real bouts beginning at node 0 
            # and exploring deeper into the maze, which is > than a trajectory length of 2 (node 0 -> node 127)
            valid_boutID = np.where(TrajS[mouseID,:,2]!=InvalidState)[0]
            N = len(valid_boutID)
            valid_bouts.extend([N])

            for count in np.arange(avg_count):
                # Initialize model parameters
                V = np.random.rand(self.S)  # state-action values
                V[HomeNode] = 0  # setting action-values of maze entry to 0
                V[RewardNode] = 0  # setting action-values of reward port to 0
                state_hist_mouse = {}
                R_visits = 0

                for n in np.arange(N):
                    valid_episode = False
                    episode_attempt = 0

                    # Extract from real mouse trajectory the terminal node in current bout and trajectory length
                    end = np.where(TrajS[mouseID,valid_boutID[n]]==InvalidState)[0][0]
                    valid_traj = TrajS[mouseID,valid_boutID[n],0:end]

                    # Back-up a copy of state-values to use in case the next episode has to be discarded
                    V_backup = np.copy(V)

                    # Begin episode
                    while not valid_episode and episode_attempt < episode_cap:
                        # Initialize starting state,s0 to node 0
                        s = 0
                        state_hist = []

                        while s!=HomeNode and s!=RewardNode:
                            # Record current state
                            state_hist.extend([s])

                            # Use softmax policy to select action, a at current state, s
                            if s in lv6_nodes:
                                aprob = [1,0,0]
                            else:
                                betaV = [np.exp(beta*V[int(val)]) for val in nodemap[s,:]]
                                aprob = []
                                for atype in np.arange(3):
                                    if np.isinf(betaV[atype]):
                                        aprob.extend([1])
                                    elif np.isnan(betaV[atype]):
                                        aprob.extend([0])
                                    else:
                                        aprob.extend([betaV[atype]/np.nansum(betaV)])

                            # Check for invalid probabilities
                            for i in aprob:
                                if np.isnan(i):
                                    print('Invalid action probabilities ', aprob, betaV, s)
                                    print(alpha, beta, gamma, mouseID, n)
                            if np.sum(aprob) < 0.999:
                                print('Invalid action probabilities, failed summing to 1: ', aprob, betaV, s)
                            a = np.random.choice([0,1,2],1,p=aprob)[0]

                            # Take action, observe reward and next state
                            sprime = int(nodemap[s,a])
                            if sprime == RewardNode:
                                R = 1  # Receive a reward of 1 when transitioning to the reward port
                            else:
                                R = 0

                            # Update action-value of previous state value, V[s]
                            V[s] += alpha * (R + gamma*V[sprime] - V[s])
                            if np.isnan(V[s]):
                                print('Warning invalid state-value: ', s, sprime, V[s], V[sprime], alpha, beta, gamma, R)
                            elif np.isinf(V[s]):
                                print('Warning infinite state-value: ', V)
                            elif V[s]>value_cap:
                                #print('Warning state value exceeded upper bound. Might approach infinity')
                                V[s] = value_cap

                            # Shift state values for the next time step
                            s = sprime

                            # Check whether to abort the current episode
                            if len(state_hist) > len(valid_traj):
                                #print('Trajectory too long. Aborting episode')
                                break
                        state_hist.extend([s])

                        # Find actual end node for mouse trajectory in the current bout/episode
                        if s == valid_traj[-1]:
                            if (len(state_hist) > 2) and (len(state_hist) > 0.5 * len(valid_traj)) and (len(state_hist) <= len(valid_traj)):
                                state_hist_mouse[n] = state_hist
                                valid_episode = True
                        else:
                            R = 0
                            V = np.copy(V_backup)
                            #print('Rejecting episode of length: ', len(state_hist), ' for mouse ', mouseID, ' bout ', valid_boutID[n], ' traj length ', len(valid_traj))
                            episode_attempt += 1

                    if episode_attempt >= episode_cap:
                        print('Failed to generate episodes for mouse ', mouseID, ' with parameter set: ', alpha, beta, gamma)
                        success = 0
                        break
                state_hist_AllMice[mouseID] = state_hist_mouse

        return state_hist_AllMice, valid_bouts, success
