import os
import sys
import time

import pystan
import numpy as np
import pickle
import random
import datetime


start_time = time.time()

stan_models_path = './stan/'
stan_model = 'distancemodel.stan'
run_id = 'B5'

print(">>>> Model from", stan_model)
print(">>>> Run id", run_id)


from MM_Traj_Utils import LoadTrajFromPath
import utils
import parameters as p
import maze_spatial_mapping as msm
import actions

tf = LoadTrajFromPath(p.OUTDATA_PATH + f'{run_id}-tf')
episodes = utils.convert_traj_to_episodes(tf)[1:200]

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

print(TrajS.shape)
print(TrajA.shape)

model_data = {
    'K': 6,
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

sm = pystan.StanModel(file=stan_models_path+stan_model)

# Run STAN model
fit = sm.sampling(
    data=model_data, chains=4, n_jobs=1,
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
os.makedirs(results_dir, exist_ok=True)
with open(results_dir + f'{run_id}-model_{stan_model}.pkl', 'wb') as f:
    pickle.dump(sm, f, protocol=-1)
with open(results_dir + f'{run_id}-fit_{stan_model}.pkl', 'wb') as f:
    pickle.dump(fit, f, protocol=-1)
with open(results_dir + f'{run_id}-summary_{stan_model}.pkl', 'wb') as f:
    pickle.dump(summary_dict, f, protocol=-1)
with open(results_dir + f'{run_id}-extract_{stan_model}.pkl', 'wb') as f:
    pickle.dump(draws, f, protocol=-1)

time_elapsed = time.time() - start_time
print("**** END ****", run_id, "time_elapsed", time_elapsed / 60, "minutes")


# B5 - 4
# Inference for Stan model: anon_model_fb6547903c17ff209af2f1d3028c45ae.
# 4 chains, each with iter=2000; warmup=1000; thin=1;
# post-warmup draws per chain=1000, total post-warmup draws=4000.
#
#        mean se_mean     sd    2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# c[1]   0.05  3.7e-4   0.02 -1.2e-3   0.03   0.05   0.06   0.09   4033    1.0
# c[2]   0.22  2.3e-4   0.01    0.19   0.21   0.22   0.23   0.25   3146    1.0
# c[3]   0.26  4.1e-4   0.02    0.22   0.25   0.26   0.27    0.3   2374    1.0
# c[4]   -0.1  2.8e-4   0.01   -0.13  -0.11   -0.1  -0.09  -0.07   2725    1.0
# lp__  -9905    0.03   1.37   -9909  -9906  -9905  -9904  -9904   1938    1.0
#
# Samples were drawn using NUTS at Mon Sep  5 14:30:01 2022.
# No treedepth issues:  True
# No energy issues:  True
# No divergence issues:  True
# No effective sample size issues:  True
# No R issues:  True
# **** END **** B5 time_elapsed 65.1930110692978 minutes

# B5 - 5
# Inference for Stan model: anon_model_fb6547903c17ff209af2f1d3028c45ae.
# 4 chains, each with iter=2000; warmup=1000; thin=1;
# post-warmup draws per chain=1000, total post-warmup draws=4000.
#
#        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# c[1]   0.05  3.8e-4   0.02 4.2e-3   0.03   0.05   0.07    0.1   3919    1.0
# c[2]   0.23  2.3e-4   0.01    0.2   0.22   0.23   0.24   0.26   3132    1.0
# c[3]   0.19  3.8e-4   0.02   0.14   0.17   0.19    0.2   0.23   3428    1.0
# c[4]  -0.15  2.8e-4   0.02  -0.19  -0.17  -0.16  -0.14  -0.12   3675    1.0
# c[5]   0.13  3.5e-4   0.02   0.09   0.12   0.13   0.15   0.17   3293    1.0
# lp__  -9749    0.04   1.53  -9753  -9750  -9749  -9748  -9747   1813    1.0
#
# Samples were drawn using NUTS at Mon Sep  5 19:19:36 2022.
# No treedepth issues:  True
# No energy issues:  True
# No divergence issues:  True
# No effective sample size issues:  True
# No R issues:  True
# **** END **** B5 time_elapsed 58.95564783414205 minutes

# Inference for Stan model: anon_model_fb6547903c17ff209af2f1d3028c45ae.
# 4 chains, each with iter=2000; warmup=1000; thin=1;
# post-warmup draws per chain=1000, total post-warmup draws=4000.
#
#        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# c[1]   0.05  3.4e-4   0.02 3.2e-3   0.04   0.05   0.07    0.1   4862    1.0
# c[2]   0.23  2.0e-4   0.01   0.21   0.22   0.23   0.24   0.26   4345    1.0
# c[3]   0.17  3.6e-4   0.02   0.13   0.16   0.17   0.19   0.22   3806    1.0
# c[4]  -0.15  3.0e-4   0.02  -0.18  -0.16  -0.15  -0.13  -0.11   3516    1.0
# c[5]   0.19  4.7e-4   0.03   0.14   0.17   0.19   0.21   0.24   2965    1.0
# c[6]  -0.07  2.8e-4   0.02   -0.1  -0.08  -0.07  -0.06  -0.03   3705    1.0
# lp__  -9605    0.04   1.73  -9609  -9606  -9605  -9604  -9603   1859    1.0
#
# Samples were drawn using NUTS at Mon Sep  5 20:50:46 2022.
# No treedepth issues:  True
# No energy issues:  True
# No divergence issues:  True
# No effective sample size issues:  True
# No R issues:  True
# **** END **** B5 time_elapsed 72.61756316820781 minutes


# B6 - 4
# Inference for Stan model: anon_model_fb6547903c17ff209af2f1d3028c45ae.
# 4 chains, each with iter=2000; warmup=1000; thin=1;
# post-warmup draws per chain=1000, total post-warmup draws=4000.
#
#        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# c[1]   -0.1  3.9e-4   0.02  -0.14  -0.11  -0.09  -0.08  -0.05   3599    1.0
# c[2]   0.07  2.2e-4   0.01   0.04   0.06   0.07   0.07   0.09   3123    1.0
# c[3]   0.51  3.8e-4   0.02   0.47   0.49   0.51   0.52   0.54   2554    1.0
# c[4]  -0.18  2.5e-4   0.01   -0.2  -0.18  -0.17  -0.17  -0.15   2810    1.0
# lp__ -1.1e4    0.03   1.37 -1.1e4 -1.1e4 -1.1e4 -1.1e4 -1.1e4   1618    1.0
#
# Samples were drawn using NUTS at Sun Sep  4 19:16:00 2022.
# No treedepth issues:  True
# No energy issues:  True
# No divergence issues:  True
# No effective sample size issues:  True
# No R issues:  True
# **** END **** b6 time_elapsed 46.822215664386746 minutes


# B6 - 5
# Inference for Stan model: anon_model_fb6547903c17ff209af2f1d3028c45ae.
# 4 chains, each with iter=2000; warmup=1000; thin=1;
# post-warmup draws per chain=1000, total post-warmup draws=4000.
#
#        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
# c[1]   -0.1  4.3e-4   0.02  -0.14  -0.11   -0.1  -0.08  -0.05   3118    1.0
# c[2]   0.08  2.3e-4   0.01   0.05   0.07   0.08   0.09    0.1   2957    1.0
# c[3]    0.4  4.3e-4   0.02   0.35   0.38    0.4   0.41   0.44   2479    1.0
# c[4]  -0.27  2.8e-4   0.02   -0.3  -0.28  -0.27  -0.25  -0.24   3169    1.0
# c[5]   0.22  3.4e-4   0.02   0.18   0.21   0.22   0.23   0.26   3158    1.0
# lp__ -1.1e4    0.03   1.55 -1.1e4 -1.1e4 -1.1e4 -1.1e4 -1.1e4   2015    1.0
#
# Samples were drawn using NUTS at Sun Sep  4 16:55:37 2022.
# No treedepth issues:  True
# No energy issues:  True
# No divergence issues:  True
# No effective sample size issues:  True
# No R issues:  True
# **** END **** b6 time_elapsed 64.94734901984533 minutes
