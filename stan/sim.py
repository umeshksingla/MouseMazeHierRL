# Imports
from __future__ import print_function
import pickle
import numpy as np
import os, sys
from itertools import product
sys.path.append('scrap_code')

module_path = 'src'
if module_path not in sys.path:
    sys.path.append(module_path)
data_path = 'traj_data'
if data_path not in sys.path:
    sys.path.append(data_path)

# Markus's code
from MM_Maze_Utils import *
from MM_Traj_Utils import *
from parameters import *
from utility import *
from src import utils

nodemap = utils.get_SAnodemap()

# Duplicate of function on TD_Models.ipynb
def TDlambda_Rvisit(sub_fits, fit_group, fit_group_data, max_bouts, max_traj, EndNode=-1):
    '''
    Predict trajectories using TD-lambda model. In this version home and reward node are terminal states.
    Predicted trajectories can't be longer than its corresponding bout in real mouse trajectories.
    Can set MatchEndNode = True to set an additional constraint on generating predicted trajectories with the same end node as the real counterpart
    '''
    state_hist_AllMice = {}
    episode_cap = 500
    success = 1
    MatchBouts = False
    MatchEndNode = True
    MatchTrajLength = False

    if fit_group == 'Rew':
        TrajS = pickle.load(open(fit_group_data, 'rb')).astype(int)

    for mouseID in np.arange(N):
        # Set model parameters
        alpha, beta, gamma, lamda = sub_fits[mouseID]
        if MatchBouts:
            NumBouts = len(np.where(TrajS[mouseID, :, 0] != InvalidState)[0])
        else:
            NumBouts = max_bouts

        # Initialize model parameters
        if init == 'ZERO':
            V = np.zeros(S)  # state-action values
        elif init == 'RAND':
            V = np.random.rand(S)
        elif init == 'ONES':
            V = np.random.rand(S)
        V[WaterPortState] = 0  # setting action-values of reward port to 0
        e = np.zeros(S)  # eligibility trace vector for all states
        state_hist_mouse = {}

        for boutID in np.arange(NumBouts):
            valid_episode = False
            abort_episode = False
            episode_attempt = 0

            # Extract from real mouse trajectory the terminal node in current bout and trajectory length
            if MatchEndNode or MatchTrajLength:
                end = np.where(TrajS[mouseID, boutID] == InvalidState)[0][0]
                valid_traj = TrajS[mouseID, boutID, 0:end]
                EndNode = valid_traj[-1]
                if EndNode == RewardNode:
                    EndNode = WaterPortState
            if MatchTrajLength:
                max_traj = len(valid_traj)

            # Back-up a copy of state-values to use in case the next episode has to be discarded
            V_backup = np.copy(V)
            e_backup = np.copy(e)

            # Begin episode
            while not valid_episode and episode_attempt < episode_cap:

                init_step = True
                sprime = StartNode
                state_hist = []

                while sprime != HomeNode and sprime != WaterPortState:

                    if init_step:
                        # Initialize starting state,s0 to node 0
                        s = HomeNode
                        init_step = False
                    else:
                        # Update future state to current state
                        s = sprime

                    # Record current state
                    state_hist.extend([s])

                    # Use softmax policy to select action, a at current state, s
                    a, prob = softmax(s, V, beta)

                    # Take action, observe reward and next state
                    sprime = nodemap[s, a]
                    if sprime == WaterPortState:
                        R = RewardNodeMag  # Receive a reward of 1 when transitioning to the reward port
                    else:
                        R = 0

                    # Calculate error signal for current state
                    td_error = R + gamma * V[sprime] - V[s]
                    e[s] = 1

                    # Propagate value to all other states
                    for node in np.arange(S):
                        V[node] += alpha * td_error * e[node]
                        e[node] = gamma * lamda * e[node]

                    if np.isnan(V[s]):
                        print('Warning invalid state-value: ', s, sprime, V[s], V[sprime], sub_fits)

                    # Check whether to abort the current episode
                    # Trajectory can't be 0 -> 127 and can't be longer than the maximum trajectory length
                    if len(state_hist) > max_traj:
                        # print('Trajectory too long. Aborting episode')
                        abort_episode = True
                        print('Incorrect trajectory length: ', len(state_hist))
                        break
                    else:
                        abort_episode = False

                # End of episode
                state_hist.extend([sprime])

                if abort_episode or len(state_hist) <= 2:
                    # Don't save predicted trajectory and attempt episode again
                    V = np.copy(V_backup)
                    e = np.copy(e_backup)
                    episode_attempt += 1
                elif not abort_episode:
                    if not MatchEndNode:
                        state_hist_mouse[boutID] = state_hist
                        valid_episode = True
                    elif MatchEndNode:
                        # Checking if predicted trajectory meets another minimum requirement
                        # Trajectory must end at the same terminal node as the real trajectory bout
                        if sprime == EndNode:
                            state_hist_mouse[boutID] = state_hist
                            valid_episode = True
                        else:
                            V = np.copy(V_backup)
                            e = np.copy(e_backup)
                            episode_attempt += 1
                            #print('Invalid episode: Requirements are to end at ', realTerminalNode, ' with length ', len(valid_traj))
                            #print('Predicted Trajectory statistics: ends at ', s, ' with length ', len(state_hist))

                if episode_attempt >= episode_cap:
                    # If there's a failure in generating a valid episode,
                    # quit predicting trajectories for the current parameter set
                    print('Failed to generate episodes for mouse ', mouseID, ' with parameter set: ', sub_fits[mouseID])
                    success = 0
                    return {}, success
        state_hist_AllMice[mouseID] = state_hist_mouse

    return state_hist_AllMice, success

# Load environment variables
main_dir = os.getenv('MAIN_DIR') + '/'
pred_traj_dir = main_dir+'traj_data/pred_traj/'+os.getenv('DATA_DIR')+'/'
real_data = main_dir+'traj_data/real_traj/'+os.getenv('REAL_DATA')
max_traj = int(os.getenv('MAX_TRAJ'))
max_bouts = int(os.getenv('MAX_BOUTS'))
params = os.getenv('PARAMS').split(',')
sweep_range = os.getenv('SWEEP_RANGE')
exec('sweep_range='+sweep_range)
param_range = os.getenv('PARAM_RANGE')
exec('param_range='+param_range)
param_specific = os.getenv('PARAM_SPECIFIC')
exec('param_specific='+param_specific)
init = os.getenv('INIT')

if not os.path.exists(pred_traj_dir):
    print('Creating data directory')
    os.mkdir(pred_traj_dir)

# Load values for free parameters
param_lists = []
for paramID, param in enumerate(params):
    if sweep_range[paramID]:
        exec(str(param)+'_range='+str(list(np.arange(param_range[paramID][0], param_range[paramID][1], param_range[paramID][2]))))
    else:
        exec(str(param)+'_range='+str(param_specific[paramID]))
    exec('param_lists.append('+str(param)+'_range'+')')
exec('param_sets='+'list(product('+str(param_lists)[1:-1]+'))')
print('Parameter lists: ', param_lists)

true_param = {}
for set_counter, param_values in enumerate(param_sets):
    print('Now simulating: set ', set_counter, ' with ', param_values)
    true_param[set_counter] = param_values
    subfits = dict(zip(np.arange(N), [param_values] * N))
    state_hist_AllMice, success = TDlambda_Rvisit(subfits,'Rew',real_data, max_bouts, max_traj)

    if success == 0:
        print('Not saving set ', set_counter)
    elif success == 1:
        # Converting predicted trajectories from a dictionary format to array format padded with InvalidState
        simTrajS = np.ones((N, max_bouts, max_traj), dtype=int) * InvalidState
        for mouseID in np.arange(N):
            for boutID in np.arange(len(state_hist_AllMice[mouseID])):
                simTrajS[mouseID, boutID, 0:len(state_hist_AllMice[mouseID][boutID])] = state_hist_AllMice[mouseID][boutID]
        pickle.dump(simTrajS,open(pred_traj_dir+'set'+str(set_counter)+'.pkl','wb'))

# Save parameter sets
pickle.dump(true_param,open(pred_traj_dir+'true_param.pkl','wb'))
print('Predicted trajectory generation complete.')