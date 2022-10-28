import os, sys
import pystan
import pickle
import numpy as np
import shutil
module_path = 'src'
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('scrap_code')
from utility import *
from parameters import *

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
    input_traj = 'set'+str(set)+'.pkl'
    log_dir = os.getenv('LOG_DIR')
    true_params = pickle.load(open(data_dir + 'true_param.pkl', 'rb'))
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

# Loading nodes of mice trajectories
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
    print('Number of subjects: ', np.shape(TrajS)[0], ' Max Number of Bouts: ', np.shape(TrajS)[1], ' Max TrajSize: ', np.shape(TrajS)[2])

    N = np.shape(TrajS)[0]          # setting the number of rewarded mice
    B = np.shape(TrajS)[1]          # setting the maximum number of bouts until the first reward was sampled
    BL = np.shape(TrajS)[2]         # setting the maximum bout length until the first reward was sampled

    # Storing actions taken at each state, s to transition to state s'
    TrajA = state_to_action(TrajS, N, B, BL)

    print('Initializing values')
    if init == 'ZERO':
        V0 = np.zeros((N, S))
    elif init == 'RAND':
        V0 = np.random.rand(N, S)  # initialized state values for each mouse
    elif init == 'ONES':
        V0 = np.ones((N, S))
    V0[:, WaterPortState] = 0  # setting action-values of terminal state to 0
    e0 = np.zeros(S)

    # Run STAN model
    model_data = {'N': N,
                  'B': B,
                  'BL': BL,
                  'S': S,
                  'A': A,
                  'RewardNodeMag': RewardNodeMag,
                  'V0': V0,
                  'e0': e0,
                  'nodemap': nodemap,
                  'InvalidState': InvalidState,
                  'HomeNode': HomeNode,
                  'WaterPortState': WaterPortState,
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
    pickle.dump(best_sub_fits, open(save_dir_fit+'best_sub_fits.pkl',"wb"))

    # Save group fits
    for id, param in enumerate(free_params):
        summary_param_type = np.where(summary_dict['summary_rownames'] == param + '_mu_phi')[0][0]
        summary_val = summary_dict['summary'][summary_param_type][0]
        best_group_fits[id] = summary_val
    pickle.dump(best_group_fits, open(save_dir_fit+'best_group_fits.pkl',"wb"))

    # Copy log file from main directory to stan_result dir
    if traj_type == 'simulated':
        shutil.copyfile(main_log_dir + log_dir + '/log' + str(set) + '.out', save_dir_fit + 'log' + str(set) + '.out')
    elif traj_type == 'real':
        shutil.copyfile(main_log_dir + log_file + '.out', save_dir_fit + log_file + '.out')