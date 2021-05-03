# Generating simulated data for a range of parameter values and saving these trajectories in the directory 'stan_simdata_dir'

# Set variables
file_name = 'set5_a0.05_b2_g0.7.p'
stan_simdata_dir = main_dir+'stan/pre_reward_traj/'
sim_data = False
save_traj = False
N = 10
TrajSize = 3000
TrajNo = 20

if sim_data:
    # Simulating data for parameter recovery
    da = 0.1
    db = 0.1
    dg = 0.1
    alpha_range = np.arange(0,1+da,da)
    beta_range = np.arange(0,1+db,db)
    gamma_range = np.arange(0,1+dg,dg)
    true_param = {}
    set_counter = 0
    for gamma in gamma_range:
        for beta in beta_range:
            for alpha in alpha_range:
                #alpha,beta,gamma = [0.2,10,0.2]
                print('Now simulating: ', set_counter, alpha, beta, gamma)
                true_param[set_counter] = [alpha, beta, gamma]

                best_sub_fits = {}
                for mouseID in np.arange(10):
                    best_sub_fits[mouseID] = [alpha,beta,gamma]

                state_hist_AllMice, valid_bouts, success = TD0_first_visit_generate_traj(best_sub_fits, 'Rew', "[#real_data]")

                simTrajS = np.ones((N,TrajNo,TrajSize), dtype=int) * InvalidState
                for mouseID in np.arange(N):
                    for boutID in np.arange(len(state_hist_AllMice[mouseID])):
                        simTrajS[mouseID,boutID,0:len(state_hist_AllMice[mouseID][boutID])] = state_hist_AllMice[mouseID][boutID]
                if success == 0:
                    print('Not saving set ', set_counter)
                elif success == 1:
                    if TD0_type == 'first_visit':
                        pickle.dump(simTrajS,open(stan_simdata_dir+'full_search_first_visit/set'+str(set_counter)+'.p','wb'))

                # Increment counter
                set_counter += 1

    # Save true parameter sets
    if first_visit_TD:
        pickle.dump(true_param,open(stan_simdata_dir+'full_search_first_visit/true_param.p','wb'))
    else:
        pickle.dump(true_param,open(stan_simdata_dir+'full_search/true_param.p','wb'))
        
    # Convert simulated trajectory from a dictionary to an array and save
    if save_traj:
        simTrajS = np.ones((N,TrajNo,TrajSize), dtype=int) * InvalidState
        for mouseID in np.arange(N):
            for boutID in np.arange(len(state_hist_AllMice[mouseID])):
                simTrajS[mouseID,boutID,0:len(state_hist_AllMice[mouseID][boutID])] = state_hist_AllMice[mouseID][boutID]
        pickle.dump(simTrajS,open(stan_simdata_dir+file_name,'wb'))