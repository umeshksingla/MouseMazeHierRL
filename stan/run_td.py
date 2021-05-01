import os, sys
import pystan
import pickle
import numpy as np
import shutil

# Set directories for access
main_dir = '/sphere/mattar-lab/stan/'
stan_model_dir = main_dir + 'test_models/'
traj_dir = main_dir + 'traj_data/'
main_log_dir = main_dir + 'logs/'

# Load environment variables
traj_type = os.getenv('TRAJ_TYPE')
results_dir = main_dir + 'stan_results/' + os.getenv('RESULTS_DIR') + '/'
if traj_type == 'simulated':
    data_dir = traj_dir + 'pred_traj/' + os.getenv('DATA_DIR') + '/'
    set = int(os.getenv('SET'))
    input_traj = 'set'+str(set)+'.p'
    log_dir = os.getenv('LOG_DIR')
    true_params = pickle.load(open(data_dir + 'true_param.p', 'rb'))
elif traj_type == 'real':
    data_dir = traj_dir + 'real_traj/'
    input_traj = os.getenv('TRAJ_DATA')
    log_file = os.getenv('LOG_FILE')
stan_file = stan_model_dir + os.getenv('STAN_FILE')
free_params = os.getenv('PARAMS').split(',')
upper_bounds = os.getenv('UPPER_BOUNDS')
exec('upper_bounds='+upper_bounds)
init = os.getenv('INIT')  # 'ZERO': for all state values to be zero, 'RAND1': for state values to be randomly assigned [0,1]

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

print('Free params: ', free_params)
nodemap = pickle.load(open('nodemap.p', 'rb'))  # retrieving a list of next states for every node in the maze

# Loading nodes of mice trajectories
'''
TrajS   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)
          Matrix entries are node labels and extra space in the matrix is filled with -1
TrajA   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)
          Matrix entries are action indices (1, 2 or 3) taken to transition from i to i+1 in TrajS
          extra space in the matrix is filled with an invalid action, 0.
          Action values of 1 is a transition from a deep node, s to shallow node sprime
          Action values 2 and 3 are transitions from a shallow node, s to deeper nodes, sprime
nodemap: 127 by 3 matrix of (node position, 3 possible node positions accessible from the current node)
          Matrix entries are node labels and extra space in the matrix is filled with nan
'''

sim_data_file = input_traj
if os.path.isfile(data_dir + sim_data_file):
    print('Valid dataset for fitting')
    if traj_type == 'simulated':
        save_dir_fit = results_dir + 'set' + str(set) + '/'
    elif traj_type == 'real':
        save_dir_fit = results_dir
    if not os.path.exists(save_dir_fit):
        os.mkdir(save_dir_fit)

    TrajS = pickle.load(open(data_dir + sim_data_file, 'rb')).astype(int)
    print('Loading', traj_type, 'data from ', data_dir + sim_data_file)

    N = np.shape(TrajS)[0]          # setting the number of rewarded mice
    B = np.shape(TrajS)[1]          # setting the maximum number of bouts until the first reward was sampled
    BL = np.shape(TrajS)[2]         # setting the maximum bout length until the first reward was sampled
    S = 128                         # number of states/nodes in the maze
    A = 3                           # number of actions available at each state
    RewardNodeMag = 1               # reward magnitude at the reward port (node 116)
    InvalidState = -1              # padding to indicate invalid nodes on nodemap and invalid states in trajectories
    StartNode = 0                  # node at which all episodes begin
    HomeNode = 127                 # node at maze entrance
    RewardNode = 116               # node where liquid reward is located

    # Storing actions taken at each state, s to transition to state s'
    TrajA = np.zeros(np.shape(TrajS)).astype(int)
    for n in np.arange(N):
        for b in np.arange(B):
            for bl in np.arange(BL - 1):
                if TrajS[n, b, bl + 1] != -1 and TrajS[n, b, bl] != 127:
                    TrajA[n, b, bl] = np.where(nodemap[TrajS[n, b, bl], :] == TrajS[n, b, bl + 1])[0][0] + 1

    print('Initializing values')
    if init == 'ZERO':
        V0 = np.zeros((N, S))
    elif init == 'RAND1':
        V0 = np.random.rand(N, S)  # initialized state values for each mouse
    elif init == 'ONES':
        V0 = np.ones((N, S))
    V0[:, HomeNode] = 0  # setting action-values of terminal state, 127 to 0
    V0[:, RewardNode] = 0  # setting action-values of terminal state, 116 to 0

    # Run STAN model
    model_data = {'N': N,
                  'B': B,
                  'BL': BL,
                  'S': S,
                  'A': A,
                  'RewardNodeMag': RewardNodeMag,
                  'V0': V0,
                  'nodemap': nodemap,
                  'InvalidState': InvalidState,
                  'HomeNode': HomeNode,
                  'StartNode': StartNode,
                  'RewardNode': RewardNode,
                  'TrajS': TrajS,
                  'TrajA': TrajA,
                  'NUM_PARAMS': len(upper_bounds),
                  'UB': upper_bounds}

    if traj_type == 'simulated':
        print('Fitting to simulated data from file ', sim_data_file, ' with true parameters ', true_params[set], '(alpha, beta, gamma)')
    elif traj_type == 'real':
        print('Fitting to real data from file ', sim_data_file, ' free parameters:', free_params)

    sm = pystan.StanModel(file=stan_file)
    # other STAN settings: , control={'max_treedepth':15, 'adapt_delta':0.9}
    fit = sm.sampling(data=model_data, iter=2000, chains=4, warmup=250, control={'max_treedepth':15, 'adapt_delta':0.9}, n_jobs=1)

    # Save fit
    with open(save_dir_fit + 'fit.pkl', 'wb') as f:
        pickle.dump({'model': sm, 'fit': fit}, f, protocol=-1)
    fit_data = open(save_dir_fit + 'fit_data.txt', 'w')
    print(fit, file=fit_data)
    fit_data.close()

    # Save fit diagnostics
    diag_fitdata = open(save_dir_fit + 'diagnostics_log.txt', 'w')
    diag_treedepth = pystan.diagnostics.check_treedepth(fit, verbose=3)
    diag_energy = pystan.diagnostics.check_energy(fit, verbose=3)
    diag_div = pystan.diagnostics.check_div(fit, verbose=3)
    diag_neff = pystan.diagnostics.check_n_eff(fit, verbose=3)
    diag_r = pystan.diagnostics.check_rhat(fit, verbose=3)
    print('No treedepth issues: ', diag_treedepth, file=diag_fitdata)
    print('No energy issues: ', diag_energy, file=diag_fitdata)
    print('No divergence issues: ', diag_div, file=diag_fitdata)
    print('No effective sample size issues: ', diag_neff, file=diag_fitdata)
    print('No R issues: ', diag_r, file=diag_fitdata)
    diag_fitdata.close()

    # Extract best fit parameters and log likelihoods
    fit = pickle.load(open(save_dir_fit+'fit.pkl','rb'))['fit']
    best_sub_fits = {}
    best_group_fits = {}
    summary_dict = fit.summary()
    for k in np.arange(10):
        best_sub_fits[k] = np.zeros(len(free_params)+1)
        for id, param in enumerate(free_params):
            summary_param_type = np.where(summary_dict['summary_rownames'] == param + '_sub_phi[' + str(k + 1) + ']')[0][0]
            summary_val = summary_dict['summary'][summary_param_type][0]
            best_sub_fits[k][id] = summary_val
        summary_param_type = np.where(summary_dict['summary_rownames'] == 'log_LL[' + str(k + 1) + ']')[0][0]
        summary_val = summary_dict['summary'][summary_param_type][0]
        best_sub_fits[k][len(free_params)] = summary_val
    pickle.dump(best_sub_fits, open(save_dir_fit+'best_sub_fits.p',"wb"))

    # Save group fits
    for id, param in enumerate(free_params):
        summary_param_type = np.where(summary_dict['summary_rownames'] == param + '_mu_phi')[0][0]
        summary_val = summary_dict['summary'][summary_param_type][0]
        best_group_fits[id] = summary_val
    pickle.dump(best_group_fits, open(save_dir_fit+'best_group_fits.p',"wb"))

    # Copy log file from main directory to stan_result dir
    if traj_type == 'simulated':
        shutil.copyfile(main_log_dir + log_dir + '/log' + str(set) + '.out', save_dir_fit + 'log' + str(set) + '.out')
    elif traj_type == 'real':
        shutil.copyfile(main_log_dir + log_file + '.out', save_dir_fit + log_file + '.out')