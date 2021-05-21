"""
Run the model
"""

import os
import pystan
import pickle
import numpy as np
from pathlib import Path

import sys
module_path = '../src'
if module_path not in sys.path:
    sys.path.append(module_path)

from parameters import HomeNode, RewardNode, InvalidState, WaterPortNode
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived

# Load environment variables
# input files
stan_file = os.path.abspath(os.getenv('STAN_FILE'))

# output files
results_dir = os.path.abspath(os.getenv('RESULTS_DIR'))
Path(results_dir).mkdir(exist_ok=True, parents=True)
print(results_dir, stan_file)

free_params = os.getenv('PARAMS').split(',')
print('Free params: ', free_params)
upper_bounds = os.getenv('UPPER_BOUNDS')
exec('upper_bounds='+upper_bounds)
init_state_value_type = os.getenv('INIT')


def main():
    model = TDLambdaXStepsRewardReceived()
    nodemap = model.get_SAnodemap()

    S = model.S
    A = model.A

    #  Loading nodes of mice trajectories
    trajectory_data = model.extract_trajectory_data(orig_data_dir='../outdata')
    TrajS = model.load_trajectories_from_object(trajectory_data)
    N, B, BL = TrajS.shape
    print("TrajS.shape", N, B, BL)

    # Loading actions taken at each state, s to transition to state s'
    TrajA = model.load_TrajA(TrajS, nodemap)

    print('Initializing values')
    if init_state_value_type == 'ZERO':
        V0 = np.zeros((N, S))
    elif init_state_value_type == 'RAND':
        V0 = np.random.rand(N, S)  # initialized state values for each mouse
    elif init_state_value_type == 'ONES':
        V0 = np.ones((N, S))
    V0[:, HomeNode] = 0         # setting state-values of terminal state HomeNode to 0
    V0[:, WaterPortNode] = 0    # setting state-values of terminal state WaterPortNode to 0

    # STAN model data
    model_data = {'N': N,
                  'B': B,
                  'BL': BL,
                  'S': S,
                  'A': A,
                  'RewardNodeMag': 1,
                  'V0': V0,
                  'nodemap': nodemap,
                  'InvalidState': InvalidState,
                  'HomeNode': HomeNode,
                  'StartNode': 0,               # node at which all episodes begin
                  'RewardNode': RewardNode,     # reward magnitude at the reward port (node 116)
                  'WaterPortNode': WaterPortNode,
                  'TrajS': TrajS,
                  'TrajA': TrajA,
                  'NUM_PARAMS': len(upper_bounds),
                  'UB': upper_bounds}

    sm = pystan.StanModel(file=stan_file)

    # Run STAN model
    fit = sm.sampling(
        data=model_data, iter=2000, chains=4, warmup=250,
        control={'max_treedepth':15, 'adapt_delta':0.9},
        n_jobs=4
    )
    return sm, fit, N, B, BL


def save(sm, fit, N):
    summary_dict = fit.summary()
    print(fit)
    print(summary_dict)

    # Save model and fit object and fit data
    with open(results_dir + 'model.pkl', 'wb') as f:
        pickle.dump(sm, f, protocol=-1)
    with open(results_dir + 'fit.pkl', 'wb') as f:
        pickle.dump(fit, f, protocol=-1)
    with open(results_dir + 'fit_data.txt', 'w') as f:
        print(fit, file=f)

    # Save fit diagnostics
    diag_fitdata = open(results_dir + 'diagnostics_log.txt', 'w')
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
    best_sub_fits = {}
    best_group_fits = {}
    for k in np.arange(N):
        best_sub_fits[k] = np.zeros(len(free_params)+1)
        for id, param in enumerate(free_params):
            summary_param_type = np.where(summary_dict['summary_rownames'] == param + '_sub_phi[' + str(k + 1) + ']')[0][0]
            summary_val = summary_dict['summary'][summary_param_type][0]
            best_sub_fits[k][id] = summary_val
        summary_param_type = np.where(summary_dict['summary_rownames'] == 'log_LL[' + str(k + 1) + ']')[0][0]
        summary_val = summary_dict['summary'][summary_param_type][0]
        best_sub_fits[k][len(free_params)] = summary_val
    pickle.dump(best_sub_fits, open(results_dir+'best_sub_fits.p',"wb"))

    # Save group fits
    for id, param in enumerate(free_params):
        summary_param_type = np.where(summary_dict['summary_rownames'] == param + '_mu_phi')[0][0]
        summary_val = summary_dict['summary'][summary_param_type][0]
        best_group_fits[id] = summary_val
    pickle.dump(best_group_fits, open(results_dir+'best_group_fits.p',"wb"))
    return


if __name__ == '__main__':
    sm, fit, N, _, _ = main()
    save(sm, fit, N)
