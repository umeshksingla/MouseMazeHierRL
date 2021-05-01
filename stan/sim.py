# Imports
from __future__ import print_function
import pickle
import numpy as np
import os, sys
from itertools import product

module_path = 'src'
if module_path not in sys.path:
    sys.path.append(module_path)
data_path = 'traj_data'
if data_path not in sys.path:
    sys.path.append(data_path)

# Markus's code
from MM_Maze_Utils import *
from MM_Traj_Utils import *

# Some lists of nicknames for mice
RewNames=['B1','B2','B3','B4','C1','C3','C6','C7','C8','C9']
UnrewNames=['B5','B6','B7','D3','D4','D5','D6','D7','D8','D9']
AllNames=RewNames+UnrewNames
UnrewNamesSub=['B5','B6','B7','D3','D4','D5','D7','D8','D9'] # excluding D6 which barely entered the maze

# Define cell numbers of end/leaf nodes
lv6_nodes = list(range(63,127))
lv5_nodes = list(range(31,63))
lv4_nodes = list(range(15,31))
lv3_nodes = list(range(7,15))
lv2_nodes = list(range(3,7))
lv1_nodes = list(range(1,3))
lv0_nodes = list(range(0,1))
lvl_dict = {0:lv0_nodes, 1:lv1_nodes, 2:lv2_nodes, 3:lv3_nodes, 4:lv4_nodes, 5:lv5_nodes, 6:lv6_nodes}

InvalidState = -1
RewardNode = 116
HomeNode = 127
StartNode = 0
S = 128
A = 3
RewardNodeMag = 1
N = 10
nodemap = pickle.load(open('nodemap.p', 'rb'))  # retrieving a list of next states for every node in the maze
main_dir = '/sphere/mattar-lab/stan/'

# Duplicate of function on TD_Models.ipynb
def TDlambda_Rvisit(sub_fits, fit_group, fit_group_data):
    '''
    Predict trajectories using TD-lambda model. In this version home and reward node are terminal states.
    Predicted trajectories can't be longer than its corresponding bout in real mouse trajectories.
    Can set MatchEndNode = True to set an additional constraint on generating predicted trajectories with the same end node as the real counterpart
    '''
    N = 10
    state_hist_AllMice = {}
    episode_cap = 500
    value_cap = 1e5
    success = 1
    MatchEndNode = False

    if fit_group == 'Rew':
        TrajS = pickle.load(open(fit_group_data, 'rb')).astype(int)

    for mouseID in np.arange(N):
        # Set model parameters
        alpha, beta, gamma, lamda = sub_fits[mouseID]
        TrajNo = len(np.where(TrajS[mouseID, :, 0] != InvalidState)[0])

        # Initialize model parameters
        if init == 'ZERO':
            V = np.zeros(S)  # state-action values
        V[HomeNode] = 0  # setting action-values of maze entry to 0
        V[RewardNode] = 0  # setting action-values of reward port to 0
        e = np.zeros(S)  # eligibility trace vector for all states
        state_hist_mouse = {}

        for bout in np.arange(TrajNo):
            valid_episode = False
            abort_episode = False
            episode_attempt = 0

            # Extract from real mouse trajectory the terminal node in current bout and trajectory length
            end = np.where(TrajS[mouseID, bout] == InvalidState)[0][0]
            valid_traj = TrajS[mouseID, bout, 0:end]

            # Back-up a copy of state-values to use in case the next episode has to be discarded
            V_backup = np.copy(V)
            e_backup = np.copy(e)

            # Begin episode
            while not valid_episode and episode_attempt < episode_cap:
                # Initialize starting state,s0 to node 0
                s = StartNode
                state_hist = []

                while s != HomeNode and s != RewardNode:
                    # Record current state
                    state_hist.extend([s])

                    # Use softmax policy to select action, a at current state, s
                    betaV = []
                    for node in nodemap[s, :]:
                        if node == InvalidState:
                            betaV.extend([0])
                        else:
                            betaV.extend([np.exp(beta * V[node])])
                    prob = betaV / np.sum(betaV)
                    try:
                        a = np.random.choice([0, 1, 2], 1, p=prob)[0]
                    except:
                        print('Error with probabilities. betaV: ', betaV, ' nodes: ', nodemap[s, :], ' state-values: ', V[nodemap[s, :]])

                    # Take action, observe reward and next state
                    sprime = nodemap[s, a]
                    if sprime == RewardNode:
                        R = RewardNodeMag  # Receive a reward of 1 when transitioning to the reward port
                    else:
                        R = 0

                    # Calculate error signal for current state
                    td_error = R + gamma * V[sprime] - V[s]
                    e[s] += 1

                    # Propagate value to all other states
                    for node in np.arange(S):
                        V[node] += alpha * td_error * e[node]
                        e[node] = gamma * lamda * e[node]

                    if np.isnan(V[s]):
                        print('Warning invalid state-value: ', s, sprime, V[s], V[sprime], sub_fits)
                    elif np.isinf(V[s]):
                        print('Warning infinite state-value: ', V)
                    elif V[s] > value_cap:
                        # print('Warning state value exceeded upper bound. Might approach infinity')
                        V[s] = value_cap

                    # Update future state to current state
                    s = sprime

                    # Check whether to abort the current episode
                    if len(state_hist) > len(valid_traj):
                        # print('Trajectory too long. Aborting episode')
                        abort_episode = True
                        V = np.copy(V_backup)
                        e = np.copy(e_backup)
                        episode_attempt += 1
                        break
                    else:
                        abort_episode = False
                state_hist.extend([s])

                if abort_episode:
                    # Don't save predicted trajectory and attempt episode again
                    pass
                else:
                    if not MatchEndNode:
                        valid_episode = True
                    elif MatchEndNode:
                        # Checking if predicted trajectory meets another minimum requirement
                        # Trajectory must end at the same terminal node as the real trajectory bout
                        realTerminalNode = valid_traj[-1]
                        if s == realTerminalNode:
                            state_hist_mouse[mouseID] = state_hist
                            valid_episode = True
                        else:
                            V = np.copy(V_backup)
                            e = np.copy(e_backup)
                            episode_attempt += 1
                            #print('Invalid episode: Requirements are to end at ', realTerminalNode, ' with length ', len(valid_traj))
                            #print('Predicted Trajectory statistics: ends at ', s, ' with length ', len(state_hist))

            if episode_attempt >= episode_cap:
                print('Failed to generate episodes for mouse ', mouseID, ' with parameter set: ', alpha, beta, gamma)
                success = 0
                break
        state_hist_AllMice[mouseID] = state_hist_mouse

    return state_hist_AllMice, success

# Load environment variables
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

true_param = {}
subfits = {}
for set_counter, param_values in enumerate(param_sets):
    print('Now simulating: set ', set_counter, ' with ', param_values)
    true_param[set_counter] = param_values
    subfits = dict(zip(np.arange(N), [param_values] * N))
    state_hist_AllMice, success = TDlambda_Rvisit(subfits,'Rew',real_data)

    if success == 0:
        print('Not saving set ', set_counter)
    elif success == 1:
        # Converting predicted trajectories from a dictionary format to array format padded with InvalidState
        simTrajS = np.ones((N, max_bouts, max_traj)) * InvalidState
        for mouseID in np.arange(N):
            for boutID in np.arange(len(state_hist_AllMice[mouseID])):
                simTrajS[mouseID, boutID, 0:len(state_hist_AllMice[mouseID][boutID])] = state_hist_AllMice[mouseID][
                    boutID]
        pickle.dump(simTrajS,open(pred_traj_dir+'set'+str(set_counter)+'.p','wb'))

# Save parameter sets
pickle.dump(true_param,open(pred_traj_dir+'true_param.p','wb'))