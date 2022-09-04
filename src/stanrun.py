import sys
import time

import pystan
import pandas as pd
import numpy as np
import pickle
import random
import datetime


start_time = time.time()

stan_file = 'distancemodel.stan'
run_id = 'fd'

print(">>>> Model from", stan_file)
print(">>>> Run id", run_id)


from MM_Traj_Utils import LoadTrajFromPath
import utils
import parameters as p
import maze_spatial_mapping as msm
import actions

tf = LoadTrajFromPath(p.OUTDATA_PATH + 'B5-tf')
episodes = utils.convert_traj_to_episodes(tf)[1:20]

# episodes = [[0,   2,   5,  12,  25,  52, 105,  52,  25,  51, 103]]

B = len(episodes)
MAX_BL = max([len(e) for e in episodes])

TrajS = np.ones((B, MAX_BL)) * p.INVALID_STATE
TrajL = np.zeros(B).astype(int)
TrajA = np.ones((B, MAX_BL-1)) * p.INVALID_STATE

for e, epi in enumerate(episodes):
    TrajL[e] = len(epi)
    print(e, epi, TrajL[e])
    TrajS[e, :TrajL[e]] = np.array(epi)

    for t in range(1, len(epi)-1):
        prev_n = epi[t-1]
        curr_n = epi[t]
        next_n = epi[t+1]
        TrajA[e][t] = np.where((actions.actions_node_matrix[prev_n][curr_n] == next_n))[0]
        print(f"t={t}:", prev_n, curr_n, next_n, actions.actions_node_matrix[prev_n][curr_n], TrajA[e][t])

TrajS = TrajS.astype(int)
TrajA = TrajA.astype(int)

print(TrajS)
print(TrajA)

print(TrajS.shape)
print(TrajA.shape)

model_data = {
    'K': 4,
    'B': B,
    'N_CELLS': len(p.ALL_VISITABLE_CELLS),
    'N_NODES': len(p.ALL_VISITABLE_NODES),
    'N_ACTIONS': 4,
    'MAX_BL': MAX_BL,
    'TrajS': TrajS,
    'TrajL': TrajL,
    'TrajA': TrajA,

    'NODE_CELL_MAPPING': msm.NODE_CELL_MAPPING,
    'ACTION_NODE_MATRIX': actions.actions_node_matrix,
    'CELL_XY': msm.CELL_XY,
    'INVALID_STATE': p.INVALID_STATE,

}

print(model_data)

sm = pystan.StanModel(file=stan_file)
iters = 1

# Run STAN model
fit = sm.sampling(
    data=model_data, chains=1, n_jobs=1,
)

summary_dict = fit.summary()
print(fit)

diag_treedepth = pystan.diagnostics.check_treedepth(fit, verbose=3)
diag_energy = pystan.diagnostics.check_energy(fit, verbose=3)
diag_div = pystan.diagnostics.check_div(fit, verbose=3)
diag_neff = pystan.diagnostics.check_n_eff(fit, verbose=3)
diag_r = pystan.diagnostics.check_rhat(fit, verbose=3)
print('No treedepth issues: ', diag_treedepth)
print('No energy issues: ', diag_energy)
print('No divergence issues: ', diag_div)
print('No effective sample size issues: ', diag_neff)
print('No R issues: ', diag_r)

draws = fit.extract()
# Save model and fit object and fit data
results_dir = './outputs/'
with open(results_dir + f'{run_id}-model_{stan_file}.pkl', 'wb') as f:
    pickle.dump(sm, f, protocol=-1)
with open(results_dir + f'{run_id}-fit_{stan_file}.pkl', 'wb') as f:
    pickle.dump(fit, f, protocol=-1)
with open(results_dir + f'{run_id}-summary_{stan_file}.pkl', 'wb') as f:
    pickle.dump(summary_dict, f, protocol=-1)
with open(results_dir + f'{run_id}-extract_{stan_file}.pkl', 'wb') as f:
    pickle.dump(draws, f, protocol=-1)

time_elapsed = time.time() - start_time
print("**** END ****", run_id, "time_elapsed", time_elapsed / 60, "minutes")
