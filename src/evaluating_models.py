'''
Needs: set of best fit parameters in the form, sub_fits{0:[alpha_fit, beta_fit, gamma_fit, LL], 1:[]...., 9:[]}

Usage: After getting fitted parameters from model-fitting (like in STAN), use them to run model simulations
       and plot resulting trajectories with the fit results
'''

sub_fits = pickle.load(open(stan_results_dir+'TD0_cl_nonrp_real/best_sub_fits.p','rb'))
fit_group = 'Rew'
fit_group_data = stan_data_dir + 'real_traj/rewMICE_first_visit.p'
state_hist_AllMice, valid_bouts, _ = TD0_first_visit_generate_traj(sub_fits, fit_group, fit_group_data)

# Plotting to compare simulated and actual trajectory lengths
sim_lengths_all = {}
real_lengths_all = {}
rand_LL = {}
TrajS = pickle.load(open(fit_group_data,'rb'))
plt.figure()
for mouseID in np.arange(10):
    valid_boutID = np.where(TrajS[mouseID,:,2]!=InvalidState)[0]
    real_lengths = []
    sim_lengths = []
    for boutID, bout in enumerate(valid_boutID):
        end = np.where(TrajS[mouseID,bout]==InvalidState)[0][0]
        valid_traj = TrajS[mouseID,bout,0:end]
        random_choices = [val for val in valid_traj if val not in lv6_nodes]
        rand_LL[mouseID] = np.log(0.33) * len(random_choices) 
        real_lengths.extend([len(valid_traj)])
        sim_lengths.extend([len(state_hist_AllMice[mouseID][boutID])])
    real_lengths_all[mouseID] = real_lengths
    sim_lengths_all[mouseID] = sim_lengths
        
    plt.plot(np.arange(len(real_lengths_all[mouseID])), real_lengths_all[mouseID], 'r*', label='real')
    plt.plot(np.arange(len(real_lengths_all[mouseID])), sim_lengths_all[mouseID], 'g*', label='sim')
    if mouseID == 0:
        plt.legend()
plt.xlabel('Number of bouts till first reward')
plt.ylabel('Number of decisions/steps in bout')
plt.title('Number of steps in generated trajectories')