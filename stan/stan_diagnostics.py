import pickle
import os
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

#main_dir = r'C:\Users\kdilh\Documents\GitHub\RL_value\optimize_logs\STAN_subjectfit_explicit_alpha_beta_Q0_v1\fit0'
main_dir = '/sphere/mattar-lab/stan/'
save_dir_fit = main_dir+'stan_results_TD0/'
if not os.path.exists(save_dir_fit):
    print('Save directory not found')

pair_plot_dir = save_dir_fit+'pair_plots/'
os.mkdir(pair_plot_dir)

# Load and prepare fit data for plotting
fit = pickle.load(open(save_dir_fit+'/fit.pkl','rb'))['fit']
data = az.from_pystan(posterior=fit, prior=fit)

# Plot divergences for hyper parameters
# az.plot_pair(data, var_names=['alpha_mu_phi', 'beta_mu_phi','gamma_mu_phi'], divergences=True)
az.plot_pair(data, var_names=['alpha_mu_phi', 'beta_hyper_mean','gamma_mu_phi'], divergences=True)
plt.savefig(pair_plot_dir + 'hyperparam.png', bbox_inches='tight')

# Plot divergences for each mouse
for mouseID in np.arange(10):
    az.plot_pair(data, var_names=['alpha_sub_phi','beta_sub_phi','gamma_sub_phi'],
                 coords={'alpha_sub_phi_dim_0':mouseID, 'beta_sub_phi_dim_0':mouseID, 'gamma_sub_phi_dim_0':mouseID}, divergences=True)
    plt.savefig(pair_plot_dir + 'agent'+str(mouseID)+'.png', bbox_inches='tight')

