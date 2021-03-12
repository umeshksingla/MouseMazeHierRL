import os, sys
import pystan
import pickle
import numpy as np

# Update directories for access
main_dir = 'C:/Users/kdilh/Documents/GitHub/MouseMaze/stan/'
#main_dir = '/sphere/mattar-lab/MouseMaze/stan/'
save_dir_fit = main_dir+'stan_results_TD0/'
if not os.path.exists(save_dir_fit):
    os.mkdir(save_dir_fit)
stan_file = main_dir + 'td_rew.stan'

# Loading nodes of mice trajectories
'''
TrajS   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)
          Matrix entries are node labels and extra space in the matrix is filled with -1
Qnodemap: 127 by 3 matrix of (node position, 3 possible node positions accessible from the current node)
          Matrix entries are node labels and extra space in the matrix is filled with nan
'''

TrajS = pickle.load(open('pre_reward_traj/rewMICE.p','rb')).astype(int)

N = 10                          # setting the number of rewarded mice
B = np.shape(TrajS)[1]          # setting the maximum number of bouts until the first reward was sampled
BL = np.shape(TrajS)[2]         # setting the maximum bout length until the first reward was sampled
S = 127                         # number of states/nodes in the maze
A = 3                           # number of actions available at each state
RT = 1                          # reward magnitude at the reward port (node 116)
Q0 = np.random.rand(N,S+1,A)    # initialized state-action values for each mouse
Q0[:, 127,:] = 0                # setting action-values of terminal state, 127 to 0
Q0[:, 116,:] = 0                # setting action-values of terminal state, 116 to 0
Qnodemap = pickle.load(open('Qnodemap.p','rb'))   # retrieving a list of next states for every node in the maze
Qnodemap[np.isnan(Qnodemap)] = -1                      # changing invalid states from nan to -1
Qnodemap = Qnodemap.astype(int)
Q0[:,Qnodemap==-1] = -1000      # setting action-values of invalid states to some extreme negative value
UB_stan = [1, 20, 1]            # upper bounds for alpha, beta and gamma where the lower bound for all parameters is 0

# Storing actions taken at each state, s to transition to state s'
TrajA = np.zeros(np.shape(TrajS)).astype(int)
for n in np.arange(N):
    for b in np.arange(B):
        for bl in np.arange(BL-1):
            if TrajS[n,b,bl+1] != -1 and TrajS[n,b,bl] != 127:
                TrajA[n,b,bl] = np.where(Qnodemap[TrajS[n,b,bl],:]==TrajS[n,b,bl+1])[0][0]+1

# Run STAN model
model_data = {'N': N,
              'B': B,
              'BL': BL,
              'S': S,
              'A': A,
              'RT': RT,
              'Q0': Q0,
              'Qnodemap': Qnodemap,
              'TrajS': TrajS,
              'TrajA': TrajA,
              'UB': UB_stan}

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